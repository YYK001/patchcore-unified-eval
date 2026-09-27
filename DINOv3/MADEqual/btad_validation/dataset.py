"""Official BTAD train/ok, test/{ok,ko}, ground_truth/ko layout."""
from pathlib import Path

import numpy as np
from PIL import Image

from DINOv3.relation_reliability.datasets import Record, IMAGE_SUFFIXES, split_source_normal

CATEGORIES = ('01', '02', '03')


def find_root(path):
    supplied = Path(path).expanduser().resolve()
    candidates = (supplied, supplied/'BTech_Dataset_transformed', supplied/'BTAD'/'BTech_Dataset_transformed')
    found = [p for p in candidates if all((p/c/'train/ok').is_dir() for c in CATEGORIES)]
    if len(found) != 1:
        raise ValueError(f'expected one BTAD root containing 01/02/03 train/ok: {supplied}; found={found}')
    return found[0]


def images(folder):
    if not folder.is_dir(): raise FileNotFoundError(folder)
    result = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len({p.stem.casefold() for p in result}) != len(result):
        raise ValueError(f'duplicate sample/mask stem: {folder}')
    return result


def mask_info(path, image_hw):
    with Image.open(path) as im:
        if (im.height, im.width) != tuple(image_hw):
            raise ValueError(f'mask/image original geometry mismatch: {path}')
        value = np.asarray(im.convert('L'))
    codes = np.unique(value).tolist()
    if not set(codes) <= {0, 1, 255}:
        raise ValueError(f'unsupported BTAD binary mask encoding: {path}: {codes}')
    # Empty official annotations are reported, not removed or relabelled.
    return dict(values=codes, foreground_pixels=int(np.count_nonzero(value > 0)), empty=not bool((value > 0).any()))


def records(path, categories=CATEGORIES, inspect_masks=False):
    root = find_root(path); train = {}; test = {}; counts = []; manifest = []; errors = []; seen = set()
    for c in categories:
        if c not in CATEGORIES: raise ValueError(f'unknown BTAD category: {c}')
        train[c] = []; test[c] = []
        masks = {p.stem.casefold(): p for p in images(root/c/'ground_truth/ko')}
        used = set(); empty = []; codes = set()
        for split, kind in (('train', 'ok'), ('test', 'ok'), ('test', 'ko')):
            paths = images(root/c/split/kind)
            if not paths: errors.append(f'empty required split: {c}/{split}/{kind}')
            for p in paths:
                if p.resolve() in seen: raise ValueError(f'duplicate resolved image path: {p}')
                seen.add(p.resolve())
                mask = masks.get(p.stem.casefold()) if kind == 'ko' else None
                if kind == 'ko' and mask is None:
                    errors.append(f'missing mask: {p}'); continue
                with Image.open(p) as im: hw = (im.height, im.width)
                if min(hw)//4 <= 4: raise ValueError(f'image too small for frozen output/filter: {p}')
                if mask:
                    used.add(p.stem.casefold())
                    with Image.open(mask) as im:
                        if (im.height, im.width) != hw: raise ValueError(f'mask/image dimensions differ: {p} -> {mask}')
                    if inspect_masks:
                        info = mask_info(mask, hw); codes.update(info['values'])
                        if info['empty']: empty.append(mask.relative_to(root).as_posix())
                rec = Record(str(p.resolve()), int(kind == 'ko'), str(mask.resolve()) if mask else None,
                    c, 'btad', 'source' if split == 'train' else 'target', kind,
                    'source_normal' if split == 'train' else 'evaluation',
                    'not_evaluation' if split == 'train' else 'anomaly_mask' if mask else 'normal_zero_mask')
                (train[c] if split == 'train' else test[c]).append(rec)
        errors.extend(f'unmatched mask: {masks[k]}' for k in sorted(set(masks)-used))
        memory, cal = split_source_normal(train[c], 8)
        memory_paths = {r.path for r in memory}; cal_paths = {r.path for r in cal}
        if memory_paths & cal_paths: raise RuntimeError('normal split overlap')
        for split, items in (('train', train[c]), ('test', test[c])):
            for i,r in enumerate(items):
                with Image.open(r.path) as im: hw = (im.height, im.width)
                manifest.append(dict(category=c, split=split, image_index=i,
                    relative_path=Path(r.path).relative_to(root).as_posix(),
                    role=('calibration' if i%8 == 0 else 'memory') if split == 'train' else 'test',
                    label=r.label, mask=Path(r.mask_path).relative_to(root).as_posix() if r.mask_path else '',
                    height=hw[0], width=hw[1]))
        counts.append(dict(category=c, training_normal_count=len(train[c]), memory_source_count=len(memory),
            calibration_count=len(cal), test_normal_count=sum(r.label == 0 for r in test[c]),
            test_anomaly_count=sum(r.label == 1 for r in test[c]), missing_or_unmatched_masks=[e for e in errors if f'/{c}/' in e.replace('\\','/')],
            mask_values=sorted(codes) if inspect_masks else 'checked after predictions',
            empty_mask_paths=empty if inspect_masks else 'checked after predictions'))
    if errors: raise ValueError('BTAD input errors:\n'+'\n'.join(errors))
    return root, train, test, counts, manifest
