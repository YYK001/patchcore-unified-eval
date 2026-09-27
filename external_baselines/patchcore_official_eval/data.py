"""Reuse existing split/label/mask mappings; model-facing dataset returns RGB only."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .protocol import BTAD_COUNTS, evaluation_hw


def records(dataset, root, selected):
    root = Path(root).expanduser().resolve()
    if dataset == 'btad':
        from DINOv3.MADEqual.btad_validation.dataset import records as btad_records
        root, train, test, _, _ = btad_records(root, selected, inspect_masks=False)
    elif dataset == 'visa':
        from DINOv3.MADEqual.visa_task_decoupled.dataset import visa_one_class_records
        root, train, test = visa_one_class_records(root, formal=True)
    else:
        from DINOv3.relation_reliability.datasets import mvtec_ad_records, _mvtec_root
        root = _mvtec_root(root)
        train, test = mvtec_ad_records(root, selected)
    train, test = ({c: group[c] for c in selected} for group in (train, test))
    counts = []
    for c in selected:
        if any(r.label != 0 for r in train[c]):
            raise ValueError('Normal-only training required')
        count = (len(train[c]), sum(r.label == 0 for r in test[c]), sum(r.label == 1 for r in test[c]))
        if dataset == 'btad' and count != BTAD_COUNTS[c]:
            raise ValueError(f'BTAD {c} count mismatch: {count} != {BTAD_COUNTS[c]}')
        keys = [Path(r.path).relative_to(root).as_posix() for r in train[c]+test[c]]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate/overlapping train-test keys')
        counts.append(dict(category=c, training_normal=count[0], test_normal=count[1], test_anomaly=count[2],
                           memory_normal=count[0], calibration_normal=0))
    return root, train, test, counts


def metadata(record, root, role):
    with Image.open(record.path) as im:
        hw = (im.height, im.width)
    return dict(category=record.category, relative_path=Path(record.path).relative_to(root).as_posix(),
                role=role, label=int(record.label),
                mask_path=Path(record.mask_path).relative_to(root).as_posix() if record.mask_path else None,
                original_hw=list(hw), input_hw=[256, 256], evaluation_hw=list(evaluation_hw(hw)),
                coordinate_rule='full-frame; original H//4,W//4')


class Images(torch.utils.data.Dataset):
    def __init__(self, items):
        # Discard labels/mask metadata at this boundary.
        self.paths = [str(r.path) for r in items]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from torchvision.transforms.functional import to_tensor, normalize
        with Image.open(self.paths[i]) as image:
            image = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            value = to_tensor(image)
        return normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def loader(items, batch, workers):
    return torch.utils.data.DataLoader(Images(items), batch_size=batch, num_workers=workers,
                                       shuffle=False, drop_last=False, pin_memory=False)


def map_to_evaluation(prediction, original_hw):
    value = np.asarray(prediction)
    if value.shape != (256, 256) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Official finite FP32 256x256 continuous map required')
    return torch.nn.functional.interpolate(torch.from_numpy(value)[None, None],
            size=evaluation_hw(original_hw), mode='bilinear', align_corners=False)[0, 0].numpy()


def evaluation_mask(row, dataset_root):
    from types import SimpleNamespace
    from DINOv3.MADEqual.mvtec_broad6_compose2.runner import _load_mask
    mask_path = str(Path(dataset_root) / row['mask_path']) if row['mask_path'] else None
    if mask_path:
        with Image.open(mask_path) as im:
            if (im.height, im.width) != tuple(row['original_hw']):
                raise ValueError('GT/image original geometry mismatch')
    record = SimpleNamespace(label=row['label'], mask_path=mask_path, path=row['relative_path'])
    return _load_mask(record, tuple(row['evaluation_hw']))
