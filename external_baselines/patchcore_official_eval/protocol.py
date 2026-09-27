"""Frozen scientific configuration. This module uses only the standard library."""
from pathlib import Path

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / 'patchcore_official'
SOURCE = dict(url='https://github.com/amazon-science/patchcore-inspection',
              commit='fcaa92f124fb1ad74a7acf56726decd4b27cbcad', license='Apache-2.0')
PROTOCOL = dict(
    id='patchcore_official_fullframe256_seed42_v1', source=SOURCE,
    backbone='wide_resnet50_2', weights='Wide_ResNet50_2_Weights.IMAGENET1K_V1',
    layers=['layer2', 'layer3'], input_hw=[256, 256], patchsize=3, patchstride=1,
    pretrain_embed_dimension=1024, target_embed_dimension=1024,
    sampler='ApproximateGreedyCoresetSampler', coreset_ratio=0.1,
    projection_dimension=128, number_of_starting_points=10, anomaly_scorer_num_nn=1,
    seed=42, dtype='float32', amp=False, ensemble=False,
    preprocessing='RGB PIL bilinear direct resize 256x256; ImageNet normalization; no crop/augmentation',
    coordinates='official upsample then Gaussian sigma4; bilinear align_corners=False to original H//4,W//4',
    score='official image score; exact FAISS squared L2; no normalization/calibration',
    training='all official train normal; no calibration split',
    metrics='AP=average_precision; all test pixels; existing fast AUPRO 200 global linear thresholds FPR0.3 four-connected',
)
CATEGORIES = {
    'btad': ('01', '02', '03'),
    'mvtec': ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'),
    'visa': ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'),
}
BTAD_COUNTS = {'01': (400, 21, 49), '02': (399, 30, 200), '03': (1000, 400, 41)}


def categories(dataset, category):
    if category == 'all':
        return CATEGORIES[dataset]
    if category not in CATEGORIES[dataset]:
        raise ValueError(f'Unknown {dataset} category: {category}')
    return (category,)


def evaluation_hw(original_hw):
    h, w = (int(v) // 4 for v in original_hw)
    if min(h, w) < 1:
        raise ValueError('Original image must be at least 4x4')
    return h, w


def compatible(saved, dataset, category, *, benchmark=False):
    if (saved.get('protocol') != PROTOCOL or saved.get('dataset') != dataset
            or saved.get('category') != category or saved.get('benchmark') != benchmark):
        raise ValueError('Incompatible protocol/dataset/category/benchmark artifacts')
