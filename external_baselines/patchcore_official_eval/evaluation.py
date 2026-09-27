"""Saved-prediction-only evaluation. No model/official/FAISS imports here."""
from pathlib import Path
import numpy as np
import torch
from .storage import read_json, json_write, csv_write, npz_write, start_stage, finish_stage
from .protocol import PROTOCOL, CATEGORIES, compatible
from .resources import Resources, environment


def calculate(rows, maps, masks, device, allow_cpu=False):
    from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import evaluate_fast
    from DINOv3.MADEqual.mvtec_task_decoupled.runner import image_metrics
    from DINOv3.MADEqual.normal_reconstruction.scoring import fixed_fpr_diagnostics
    result = evaluate_fast([r['label'] for r in rows], masks, maps,
                           device=torch.device(device), allow_cpu_fallback=allow_cpu)
    # MUST override evaluate_fast's map-max image metrics with official scores.
    result.update(image_metrics([r['label'] for r in rows], [r['image_score'] for r in rows]))
    # Existing AUPR columns contain average precision, not trapezoidal PR area.
    result['image_AP'] = result.pop('image_AUPR')
    result['pixel_AP'] = result.pop('pixel_AUPR')
    return result, fixed_fpr_diagnostics(maps, masks)


def summarize(metrics, operating, selected, dataset):
    from DINOv3.MADEqual.guided_validation.summary import mean_available
    if len(metrics) != len(selected) or {r['category'] for r in metrics} != set(selected):
        raise ValueError('Missing/duplicate category metrics')
    scope = 'dataset_macro' if set(selected) == set(CATEGORIES[dataset]) else 'selected_categories_macro_NOT_full_dataset'
    five = ('image_AUROC', 'image_AP', 'pixel_AUROC', 'pixel_AP', 'AUPRO_at_0p3')
    macro = [dict(dataset=dataset, scope=scope, category_count=len(selected),
                  **{k: mean_available([r[k] for r in metrics])[0] for k in five})]
    fixed = []
    for cap in (.01, .05):
        group = [r for r in operating if r['fpr_cap'] == cap]
        if len(group) != len(selected) or {r['category'] for r in group} != set(selected):
            raise ValueError('Missing/duplicate fixed FPR rows')
        row = dict(dataset=dataset, scope=scope, fpr_cap=cap, category_count=len(selected),
                   averaging='category_unweighted_excluding_NA')
        for key in ('actual_fpr', 'defect_pixel_recall', 'region_mean_coverage', 'small_region_mean_coverage'):
            row[key], row[key+'_valid_category_count'] = mean_available([r[key] for r in group])
        for key in ('region_count', 'small_region_count', 'small_region_image_count'):
            row[key+'_sum'] = sum(r[key] for r in group)
        fixed.append(row)
    return macro, fixed


def saved_inputs(category_dir, dataset_root=None):
    rows = read_json(category_dir / 'predict' / 'samples.json')
    if not rows or len({(r['category'], r['relative_path']) for r in rows}) != len(rows):
        raise ValueError('Missing/duplicate prediction keys')
    maps, masks, audits = [], [], []
    for row in rows:
        with np.load(category_dir / 'predict' / row['prediction_file'], allow_pickle=False) as data:
            if (str(data['relative_path']) != row['relative_path'] or str(data['category']) != row['category']
                    or float(data['image_score']) != row['image_score']):
                raise ValueError('Prediction identity/score mismatch')
            prediction = data['evaluation_map']
            if (data['input_map'].shape != (256, 256) or data['input_map'].dtype != np.float32
                    or prediction.shape != tuple(row['evaluation_hw']) or prediction.dtype != np.float32
                    or not np.isfinite(prediction).all() or not np.isfinite(data['input_map']).all()):
                raise ValueError('Invalid saved continuous map')
        gt = category_dir / 'ground_truth' / row['prediction_file']
        if gt.exists():
            with np.load(gt, allow_pickle=False) as data:
                if str(data['relative_path']) != row['relative_path'] or int(data['label']) != row['label']:
                    raise ValueError('Cached GT identity mismatch')
                mask = data['mask']
                source_foreground = int(data['source_foreground_pixels']) if 'source_foreground_pixels' in data else None
        else:
            if dataset_root is None:
                raise FileNotFoundError('First evaluation needs --dataset-root to snapshot GT; later evaluations are offline')
            from .data import evaluation_mask
            mask = evaluation_mask(row, dataset_root)
            source_foreground = 0
            if row['label']:
                from PIL import Image
                with Image.open(Path(dataset_root)/row['mask_path']) as im:
                    source_foreground = int(np.count_nonzero(np.asarray(im.convert('L')) > 0))
            npz_write(gt, mask=mask, relative_path=np.asarray(row['relative_path']), label=np.asarray(row['label']),
                      source_foreground_pixels=np.asarray(source_foreground))
        if mask.dtype != np.bool_ or mask.shape != prediction.shape:
            raise ValueError('Invalid cached binary GT')
        maps.append(prediction)
        masks.append(mask)
        audits.append(dict(category=row['category'], relative_path=row['relative_path'], label=row['label'],
                           mask_path=row['mask_path'], empty_evaluation_mask=not bool(mask.any()),
                           original_foreground_pixels=source_foreground,
                           original_anomalous_empty_mask=(bool(row['label'] and source_foreground == 0)
                                                         if source_foreground is not None else 'N/A'),
                           anomalous_empty_evaluation_mask=bool(row['label'] and not mask.any())))
    return rows, maps, masks, audits


def evaluate(args, selected):
    root = args.output_dir
    run = read_json(root / 'run.json')
    if run['protocol'] != PROTOCOL or run['dataset'] != args.dataset or run['benchmark']:
        raise ValueError('Formal run with identical protocol required for evaluation')
    device = 'cpu' if args.cpu_check else args.metric_device
    if not args.cpu_check and (torch.device(device).type != 'cuda' or torch.device(device).index is None
            or not torch.cuda.is_available() or torch.device(device).index >= torch.cuda.device_count()):
        raise RuntimeError('CUDA fast metrics required; no implicit CPU fallback')
    result_dir = root / 'evaluations' / args.evaluation_name
    identity = dict(protocol=PROTOCOL, dataset=args.dataset, categories=list(selected),
                    stage='evaluate', cpu_check=args.cpu_check)
    if not start_stage(result_dir, identity, args.skip_completed):
        return
    meter = Resources(result_dir, [device])
    json_write(result_dir/'environment.json', environment([device]))
    metrics, operating, audits = [], [], []
    try:
        for c in selected:
            meter.category = c
            path = root / c
            state = read_json(path/'predict'/'complete.json')
            compatible(state['identity'], args.dataset, c)
            if state['status'] != 'complete':
                raise ValueError('Incomplete predictions')
            with meter.measure('saved_predictions_and_gt_read'):
                rows, maps, masks, audit = saved_inputs(path, args.dataset_root)
                if len(rows) != state['image_count']:
                    raise ValueError('Incomplete saved predictions')
                audits.extend(audit)
            with meter.measure('metric_evaluation'):
                result, fixed = calculate(rows, maps, masks, device, args.cpu_check)
            metrics.append(dict(category=c, **result))
            operating.extend(dict(category=c, **r) for r in fixed)
            csv_write(result_dir/'category_metrics.csv', metrics)
            csv_write(result_dir/'fixed_fpr.csv', operating)
            csv_write(result_dir/'mask_audit.csv', audits)
            del maps, masks
        macro, fmacro = summarize(metrics, operating, selected, args.dataset)
        csv_write(result_dir/'macro_metrics.csv', macro)
        csv_write(result_dir/'fixed_fpr_macro.csv', fmacro)
        finish_stage(result_dir, identity, formal_cuda_metrics=not args.cpu_check)
    except Exception as e:
        json_write(result_dir/'failure.json', dict(type=type(e).__name__, message=str(e)))
        raise
