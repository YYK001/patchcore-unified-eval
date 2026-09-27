"""Serial, explicitly separated check / benchmark / fit / predict / evaluate."""
import gc
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
from .protocol import PROTOCOL, categories, compatible
from .storage import read_json, json_write, csv_write, npz_write, start_stage, finish_stage
from .resources import Resources, environment


def devices(args):
    if args.cpu_check:
        return dict(model='cpu', coreset='cpu', nn='cpu', metric='cpu')
    values = {name: str(torch.device(getattr(args, name+'_device'))) for name in ('model', 'coreset', 'nn', 'metric')}
    for name, value in values.items():
        d = torch.device(value)
        if d.type != 'cuda' or d.index is None or not torch.cuda.is_available() or d.index >= torch.cuda.device_count():
            raise RuntimeError(f'{name}: explicit available CUDA device required: {value}; no fallback')
    if values['model'] != values['nn'] or values['model'] != values['metric'] or values['model'] == values['coreset']:
        raise ValueError('This protocol requires model/NN/metric on one GPU and coreset on a distinct GPU')
    import faiss
    if not hasattr(faiss, 'StandardGpuResources'):
        raise RuntimeError('Install FAISS GPU: no formal CPU search fallback')
    return values


def make_nn(args, device):
    from .adapter import ChunkedFlatL2
    return ChunkedFlatL2(device, args.query_chunk, args.faiss_temp_mb)


def identity(args, category, stage, benchmark):
    return dict(protocol=PROTOCOL, dataset=args.dataset, category=category, stage=stage, benchmark=benchmark)


def prepare_run(args, benchmark):
    from .official import verify_source, verify_weights
    source = verify_source()
    out = args.output_dir
    if out.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('Run outputs must be outside the adapter source directory')
    if out.exists() and not (out/'run.json').is_file() and any(out.iterdir()):
        raise FileExistsError('Output directory is not an owned PatchCore run; choose a new directory')
    supplied = args.backbone_weights
    shared = out/'shared'/'wide_resnet50_2-95faca4d.pth'
    if (out/'run.json').exists():
        cfg = read_json(out/'run.json')
        if cfg['protocol'] != PROTOCOL or cfg['dataset'] != args.dataset or cfg['benchmark'] != benchmark:
            raise ValueError('Incompatible existing run')
        if cfg['dataset_root'] != str(args.dataset_root):
            raise ValueError('Dataset root differs from this run; use a new run directory')
        if verify_weights(shared) != cfg['backbone_sha256']:
            raise ValueError('Shared backbone differs from saved run')
        if supplied and verify_weights(supplied) != cfg['backbone_sha256']:
            raise ValueError('Supplied backbone differs from saved run')
    else:
        if args.command == 'predict':
            raise FileNotFoundError('predict requires an existing run and saved banks; will not fit')
        digest = verify_weights(supplied)
        shared.parent.mkdir(parents=True, exist_ok=True)
        # Copy once across all categories/stages. Keep original bytes for offline provenance.
        shutil.copyfile(supplied, shared)
        adapter_source = Path(__file__).resolve().parent
        shutil.copytree(adapter_source, out/'shared'/'adapter_source',
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', 'local_validation.json'))
        from .protocol import OFFICIAL_ROOT
        shutil.copytree(OFFICIAL_ROOT/'src', out/'shared'/'official_source'/'src')
        for filename in ('LICENSE', 'NOTICE', 'README.md'):
            shutil.copyfile(OFFICIAL_ROOT/filename, out/'shared'/'official_source'/filename)
        cfg = dict(protocol=PROTOCOL, dataset=args.dataset, dataset_root=str(args.dataset_root),
                   benchmark=benchmark, backbone_sha256=digest, source=source,
                   backbone_relative_path='shared/'+shared.name)
        json_write(out/'run.json', cfg)
    return shared


def stage_provenance(path, args, dev):
    json_write(path/'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    json_write(path/'environment.json', environment(list(dev.values())))
    json_write(path/'command.json', dict(argv=[sys.executable, '-m', 'external_baselines.patchcore_official_eval', *sys.argv[1:]],
                                        note='argv array preserves quoting; engineering parameters are in arguments.json'))


def fit_category(args, c, root, items, weights, dev, benchmark=False):
    from .adapter import seed_category, make_model, ChunkedApproximateGreedy, extract_to_memmap, save_selected_bank
    from .data import loader, metadata
    path = args.output_dir/c/'fit'
    ident = identity(args, c, 'fit', benchmark)
    manifest = [metadata(r, root, 'memory_fit_all_official_train_normal') for r in items]
    if path.exists() and args.skip_completed and (path/'complete.json').is_file():
        if read_json(path/'train_manifest.json') != manifest:
            raise ValueError('Training list/geometry changed; cannot skip')
        for filename in ('bank.npy', 'coreset_indices.npy', 'model.json'):
            if not (path/filename).is_file():
                raise FileNotFoundError(path/filename)
    if not start_stage(path, ident, args.skip_completed):
        return
    stage_provenance(path, args, dev)
    json_write(path/'train_manifest.json', manifest)
    meter = Resources(path, list(dev.values()), category=c)
    seed_category()
    try:
        with meter.measure('model_initialization'):
            # fit never needs a GPU0 FAISS index; CPU placeholder holds no bank.
            model = make_model(weights, dev['model'], make_nn(args, 'cpu'))
        features = extract_to_memmap(model, loader(items, args.batch_size, args.num_workers),
                                     len(items), path/'full_features.npy', meter.measure)
        del model
        gc.collect()
        if torch.cuda.is_available() and torch.device(dev['model']).type == 'cuda':
            with torch.cuda.device(dev['model']):
                torch.cuda.empty_cache()
        sampler = ChunkedApproximateGreedy(dev['coreset'], args.projection_chunk, args.distance_chunk)
        with meter.measure('coreset_projection'):
            reduced = sampler.project(features)
        with meter.measure('coreset_selection'):
            indices = sampler.select(reduced)
        with meter.measure('bank_save'):
            np.save(path/'coreset_indices.npy', indices, allow_pickle=False)
            save_selected_bank(features, indices, path/'bank.npy')
            json_write(path/'model.json', dict(protocol=PROTOCOL, shared_backbone='../../shared/'+weights.name,
                        feature_count=len(features), bank_rows=len(indices), bank_dimension=1024,
                        starting_points=sampler.start_points,
                        index='bank.npy is authoritative; rebuild exact FlatL2 on load (no coreset/extraction)',
                        train_image_count=len(items), benchmark=benchmark))
        # Bank/config are durable BEFORE any index, predictions, or evaluation.
        del reduced, features, sampler
        gc.collect()
        if torch.cuda.is_available() and torch.device(dev['coreset']).type == 'cuda':
            with torch.cuda.device(dev['coreset']):
                torch.cuda.empty_cache()
        finish_stage(path, ident, train_images=len(items), bank_rows=len(indices),
                     full_feature_cache='full_features.npy; may be manually removed after success')
    except Exception as e:
        json_write(path/'failure.json', dict(type=type(e).__name__, message=str(e)))
        raise


def predict_category(args, c, root, items, weights, dev, benchmark=False):
    from .adapter import seed_category, make_model, load_bank
    from .data import loader, metadata, map_to_evaluation
    fit = args.output_dir/c/'fit'
    fitted = read_json(fit/'complete.json')
    compatible(fitted['identity'], args.dataset, c, benchmark=benchmark)
    if fitted['status'] != 'complete':
        raise ValueError('Completed saved fit required')
    model_config = read_json(fit/'model.json')
    if model_config.get('protocol') != PROTOCOL or model_config.get('benchmark') != benchmark:
        raise ValueError('Saved model configuration differs from this protocol')
    saved_bank = np.load(fit/'bank.npy', mmap_mode='r', allow_pickle=False)
    saved_indices = np.load(fit/'coreset_indices.npy', mmap_mode='r', allow_pickle=False)
    if (saved_bank.shape != (model_config['bank_rows'], 1024) or saved_bank.dtype != np.float32
            or saved_indices.shape != (model_config['bank_rows'],)
            or model_config['bank_rows'] != int(model_config['feature_count']*.1)):
        raise ValueError('Saved bank/coreset/configuration dimensions disagree')
    del saved_bank, saved_indices
    path = args.output_dir/c/'predict'
    ident = identity(args, c, 'predict', benchmark)
    manifest = [metadata(r, root, 'test') for r in items]
    if path.exists() and args.skip_completed and (path/'complete.json').is_file():
        if read_json(path/'test_manifest.json') != manifest:
            raise ValueError('Test metadata/list changed; cannot skip')
        saved_rows = read_json(path/'samples.json')
        if len(saved_rows) != len(manifest) or any(not (path/r['prediction_file']).is_file() for r in saved_rows):
            raise ValueError('Missing prediction artifacts; cannot skip')
    if not start_stage(path, ident, args.skip_completed):
        return
    stage_provenance(path, args, dev)
    json_write(path/'test_manifest.json', manifest)
    meter = Resources(path, list(dev.values()), category=c)
    seed_category()
    try:
        with meter.measure('model_initialization'):
            model = make_model(weights, dev['model'], make_nn(args, dev['nn']))
        with meter.measure('index_build'):
            load_bank(model, fit/'bank.npy')
        rows = []
        for images in loader(items, args.batch_size, args.num_workers):
            with meter.measure('model_inference'):
                scores, maps = model.predict(images)
            with meter.measure('prediction_save'):
                for score, prediction in zip(scores, maps):
                    row = dict(manifest[len(rows)])
                    value = np.asarray(prediction, np.float32)
                    mapped = map_to_evaluation(value, row['original_hw'])
                    if not np.isfinite(score):
                        raise ValueError('Nonfinite official image score')
                    row.update(image_score=float(score), prediction_file=f'maps/{len(rows):06d}.npz')
                    npz_write(path/row['prediction_file'], category=np.asarray(c),
                              relative_path=np.asarray(row['relative_path']), image_score=np.asarray(float(score)),
                              input_map=value, evaluation_map=mapped)
                    rows.append(row)
                # Snapshot after EVERY batch. Evaluation failure cannot lose predictions.
                json_write(path/'samples.json', rows)
                csv_write(path/'sample_scores.csv', rows)
        if len(rows) != len(items):
            raise ValueError('Incomplete prediction count')
        del model
        gc.collect()
        if torch.cuda.is_available() and torch.device(dev['model']).type == 'cuda':
            with torch.cuda.device(dev['model']):
                torch.cuda.empty_cache()
        finish_stage(path, ident, image_count=len(rows))
    except Exception as e:
        json_write(path/'failure.json', dict(type=type(e).__name__, message=str(e)))
        raise


def check(args, selected, root, train, test, counts, weights, dev):
    from .data import metadata, evaluation_mask
    from .adapter import make_model
    path = args.output_dir/'checks'/('-'.join(selected))
    ident = dict(protocol=PROTOCOL, dataset=args.dataset, categories=list(selected), stage='check', cpu_check=args.cpu_check)
    if not start_stage(path, ident, args.skip_completed):
        return
    stage_provenance(path, args, dev)
    meter = Resources(path, list(dev.values()))
    try:
        rows, audits = [], []
        for c in selected:
            for role, items in (('memory_fit_all_official_train_normal', train[c]), ('test', test[c])):
                for rec in items:
                    row = metadata(rec, root, role)
                    rows.append(row)
                    if role == 'test':
                        mask = evaluation_mask(row, root)
                        source_foreground = 0
                        if row['label']:
                            from PIL import Image
                            with Image.open(root/row['mask_path']) as im:
                                source_foreground = int(np.count_nonzero(np.asarray(im.convert('L')) > 0))
                        audits.append(dict(category=c, relative_path=row['relative_path'], label=row['label'],
                                           empty_evaluation_mask=not bool(mask.any()),
                                           original_foreground_pixels=source_foreground,
                                           original_anomalous_empty_mask=bool(row['label'] and source_foreground == 0)))
        csv_write(path/'manifest.csv', rows)
        csv_write(path/'counts.csv', counts)
        csv_write(path/'mask_audit.csv', audits)
        with meter.measure('dependency_weight_device_probe'):
            model = make_model(weights, dev['model'], make_nn(args, dev['nn']))
            # Strict weight loading plus actual per-device allocation, exact NN kernel,
            # and existing metric imports. No training or test scores in check.
            for value in set(dev.values()):
                torch.zeros(1, device=value)
            nn = model.anomaly_scorer.nn_method
            nn.fit(np.zeros((2, 1024), np.float32))
            distances, _ = nn.run(1, np.ones((1, 1024), np.float32))
            np.testing.assert_allclose(distances, [[1024]], rtol=1e-6)
            from .evaluation import calculate
            calculate([dict(label=0, image_score=0), dict(label=1, image_score=1)],
                      [np.zeros((8, 8), np.float32), np.eye(8, dtype=np.float32)],
                      [np.zeros((8, 8), bool), np.eye(8, dtype=bool)],
                      dev['metric'], args.cpu_check)
            del model
        finish_stage(path, ident, dataset_counts=counts, dual_cuda_probe=not args.cpu_check,
                     benchmark_required=True, real_data_fit_not_performed=True)
    except Exception as e:
        json_write(path/'failure.json', dict(type=type(e).__name__, message=str(e)))
        raise


def run(args):
    selected = categories(args.dataset, args.category)
    # Evaluation deliberately has no official import, weights, dataset scan, model,
    # features or coreset dependency. First evaluation snapshots masks only.
    if args.command == 'evaluate':
        from .evaluation import evaluate
        return evaluate(args, selected)
    from .data import records
    dev = devices(args)
    benchmark = args.command == 'benchmark'
    if benchmark and len(selected) != 1:
        raise ValueError('Benchmark requires one explicit category')
    weights = prepare_run(args, benchmark)
    root, train, test, counts = records(args.dataset, args.dataset_root, selected)
    if args.command == 'check':
        return check(args, selected, root, train, test, counts, weights, dev)
    for c in selected:
        if args.command in ('fit', 'benchmark'):
            items = train[c][:args.benchmark_train_images] if benchmark else train[c]
            fit_category(args, c, root, items, weights, dev, benchmark)
        if args.command in ('predict', 'benchmark'):
            items = test[c][:args.benchmark_test_images] if benchmark else test[c]
            predict_category(args, c, root, items, weights, dev, benchmark)
        print(f'{args.command} {args.dataset}/{c} complete (benchmark={benchmark})', flush=True)
    # Counts are kept with every invocation, never interpreted as benchmark fit counts.
    for c in selected:
        stage = 'predict' if args.command in ('predict', 'benchmark') else 'fit'
        target = args.output_dir/c/stage/'data_counts.csv'
        if not target.exists():
            csv_write(target, [r for r in counts if r['category'] == c])
    if benchmark:
        import csv
        summary = []
        for phase in ('fit', 'predict'):
            with (args.output_dir/selected[0]/phase/'resources.csv').open() as f:
                resource_rows = list(csv.DictReader(f))
            for stage in dict.fromkeys(r['stage'] for r in resource_rows):
                group = [r for r in resource_rows if r['stage'] == stage]
                row = dict(phase=phase, stage=stage, calls=len(group),
                           seconds=sum(float(r['seconds']) for r in group))
                for key in group[0]:
                    if key.endswith('_peak_bytes') or key.endswith('_observed_max'):
                        values = [float(r[key]) for r in group if r.get(key)]
                        row[key] = max(values) if values else None
                summary.append(row)
        report = args.output_dir/'benchmark_report.json'
        if not report.exists():
            json_write(report, dict(status='complete', benchmark=True, formal_result=False,
                train_images=min(args.benchmark_train_images, len(train[selected[0]])),
                test_images=min(args.benchmark_test_images, len(test[selected[0]])),
                stages=summary, runtime_extrapolation='none; full coreset can be much slower',
                observation='PyTorch per logical GPU peaks; nvidia-smi physical UUID sampled maxima, not exact peaks'))
