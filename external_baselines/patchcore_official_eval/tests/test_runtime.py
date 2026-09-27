"""Server/compatible-environment tests. NOT executed during local delivery.

Synthetic backbone tests exercise upstream math without downloading weights.
PC_V1_WEIGHTS enables the real WRN50-2 comparison; PC_DUAL_T4=1 enables GPU tests.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('faiss')
pytest.importorskip('timm')
from external_baselines.patchcore_official_eval.adapter import (
    ChunkedFlatL2, ChunkedApproximateGreedy, make_model, load_bank,
    seed_category, extract_to_memmap, save_selected_bank,
)
from external_baselines.patchcore_official_eval.tests.reload_probe import TinyBackbone
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.sampler import ApproximateGreedyCoresetSampler


def upstream(backbone, nn, device='cpu'):
    model = PatchCore(torch.device(device))
    model.load(backbone=backbone.eval(), layers_to_extract_from=['layer2', 'layer3'],
               device=torch.device(device), input_shape=(3, 256, 256),
               pretrain_embed_dimension=1024, target_embed_dimension=1024,
               patchsize=3, patchstride=1, anomaly_score_num_nn=1, nn_method=nn)
    return model.eval()


def test_official_embeddings_scores_maps_and_fresh_reload(tmp_path):
    torch.set_num_threads(2)
    seed_category()
    backbone = TinyBackbone().eval()
    torch.save(backbone.state_dict(), tmp_path/'tiny.pth')
    adapted = make_model(None, 'cpu', ChunkedFlatL2('cpu', query_chunk=11), copy.deepcopy(backbone))
    reference = upstream(copy.deepcopy(backbone), FaissNN(False, 2))
    images = torch.randn(2, 3, 64, 80)
    with torch.no_grad():
        a = adapted._embed(images)
        b = np.asarray(reference._embed(images))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    bank = np.ascontiguousarray(a[::7])
    np.save(tmp_path/'bank.npy', bank)
    load_bank(adapted, tmp_path/'bank.npy')
    reference.anomaly_scorer.fit([bank])
    scores, maps = adapted.predict(images)
    rs, rm = reference.predict(images)
    np.testing.assert_allclose(scores, rs, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(maps, rm, rtol=1e-5, atol=1e-5)
    np.save(tmp_path/'images.npy', images.numpy())
    subprocess.run([sys.executable, '-m', 'external_baselines.patchcore_official_eval.tests.reload_probe',
                    str(tmp_path)], check=True)
    with np.load(tmp_path/'reloaded.npz') as saved:
        np.testing.assert_allclose(scores, saved['scores'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(maps, saved['maps'], rtol=1e-5, atol=1e-5)


def test_shared_projection_and_global_coreset(monkeypatch):
    torch.set_num_threads(2)
    seed_category()
    features = np.random.randn(83, 1024).astype(np.float32)
    projection_rng = torch.get_rng_state()
    mapper = torch.nn.Linear(1024, 128, bias=False)
    adapted = ChunkedApproximateGreedy('cpu', projection_chunk=13, distance_chunk=17)
    reduced = adapted.project(features, mapper=copy.deepcopy(mapper))
    reference = ApproximateGreedyCoresetSampler(.1, torch.device('cpu'), 10, 128)
    with torch.random.fork_rng(), torch.no_grad():
        torch.set_rng_state(projection_rng)
        full = reference._reduce_features(torch.from_numpy(features))
    np.testing.assert_allclose(reduced.numpy(), full.numpy(), rtol=3e-5, atol=2e-6)
    starts = np.arange(10)
    monkeypatch.setattr(np.random, 'choice', lambda *a, **kw: starts)
    # SAME projected values and starts isolates distance chunking from projection rounding.
    expected = reference._compute_greedy_coreset_indices(reduced)
    actual = adapted.select(reduced, starts.tolist())
    np.testing.assert_array_equal(actual, expected)
    # Official exact ties choose the first index repeatedly; do not silently deduplicate.
    tied = torch.zeros(30, 128)
    np.testing.assert_array_equal(adapted.select(tied, starts.tolist()),
                                  reference._compute_greedy_coreset_indices(tied))


def test_faiss_chunks_squared_distance():
    seed_category()
    bank = np.random.randn(39, 1024).astype(np.float32)
    query = np.random.randn(51, 1024).astype(np.float32)
    small, full = ChunkedFlatL2('cpu', 7), FaissNN(False, 2)
    small.fit(bank)
    full.fit(bank)
    a, ia = small.run(1, query)
    b, ib = full.run(1, query)
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-4)
    np.testing.assert_array_equal(ia, ib)
    small.fit(np.array([[0, 0]], np.float32))
    assert small.run(1, np.array([[3, 4]], np.float32))[0][0, 0] == 25


def test_non_square_nearest_gt_empty_masks_and_label_isolation(tmp_path):
    from PIL import Image
    from external_baselines.patchcore_official_eval.data import Images, evaluation_mask, map_to_evaluation
    seed_category()
    image = tmp_path/'image.png'
    Image.new('RGB', (509, 301), color=(20, 70, 130)).save(image)
    gt = np.zeros((301, 509), np.uint8)
    gt[10:50, 300:350] = 255
    Image.fromarray(gt).save(tmp_path/'gt.png')
    normal = SimpleNamespace(path=str(image), label=0, mask_path=None)
    abnormal = SimpleNamespace(path=str(image), label=1, mask_path=str(tmp_path/'nonexistent.png'))
    x, y = Images([normal])[0], Images([abnormal])[0]
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    model = make_model(None, 'cpu', ChunkedFlatL2('cpu'), TinyBackbone())
    a = extract_to_memmap(model, [x[None]], 1, tmp_path/'a.npy')
    b = extract_to_memmap(model, [y[None]], 1, tmp_path/'b.npy')
    np.testing.assert_array_equal(a, b)
    idx = np.arange(0, len(a), 10)
    save_selected_bank(a, idx, tmp_path/'bank.npy')
    load_bank(model, tmp_path/'bank.npy')
    sx, mx = model.predict(x[None])
    sy, my = model.predict(y[None])
    np.testing.assert_array_equal(sx, sy)
    np.testing.assert_array_equal(mx, my)
    mapped = map_to_evaluation(np.ones((256, 256), np.float32)*7, (301, 509))
    assert mapped.shape == (75, 127) and mapped.dtype == np.float32
    np.testing.assert_allclose(mapped, 7)  # no per-image normalization
    row = dict(label=1, mask_path='gt.png', original_hw=[301, 509], evaluation_hw=[75, 127], relative_path='image.png')
    mask = evaluation_mask(row, tmp_path)
    expected = np.asarray(Image.fromarray(gt).resize((127, 75), Image.Resampling.NEAREST)) > 0
    np.testing.assert_array_equal(mask, expected)
    Image.fromarray(np.zeros_like(gt)).save(tmp_path/'gt.png')
    assert not evaluation_mask(row, tmp_path).any()
    assert row['label'] == 1


def test_saved_only_evaluate_and_macro_na(tmp_path):
    from external_baselines.patchcore_official_eval.protocol import PROTOCOL
    from external_baselines.patchcore_official_eval.storage import json_write, npz_write
    from external_baselines.patchcore_official_eval.evaluation import summarize
    category = tmp_path/'01'
    rows = []
    # Image-score order deliberately opposite to map maxima; override must win.
    for i in range(3):
        mask = np.zeros((20, 30), bool)
        if i == 1:
            mask[4:8, 5:9] = True
        prediction = np.arange(600, dtype=np.float32).reshape(20, 30)/600 + (2 if i == 0 else 0)
        row = dict(category='01', relative_path=f'test/{i}.png', label=int(i > 0),
                   image_score=float(i), original_hw=[80, 120], input_hw=[256, 256],
                   evaluation_hw=[20, 30], mask_path='does-not-exist.png', prediction_file=f'maps/{i}.npz')
        rows.append(row)
        npz_write(category/'predict'/row['prediction_file'], category=np.asarray('01'),
                  relative_path=np.asarray(row['relative_path']), image_score=np.asarray(row['image_score']),
                  input_map=np.zeros((256, 256), np.float32), evaluation_map=prediction)
        npz_write(category/'ground_truth'/row['prediction_file'], relative_path=np.asarray(row['relative_path']),
                  label=np.asarray(row['label']), mask=mask)
    json_write(tmp_path/'run.json', dict(protocol=PROTOCOL, dataset='btad', benchmark=False))
    json_write(category/'predict'/'samples.json', rows)
    json_write(category/'predict'/'complete.json', dict(status='complete', image_count=3,
        identity=dict(protocol=PROTOCOL, dataset='btad', category='01', benchmark=False)))
    # No model, bank, weights, input images or original masks exist in this fixture.
    for name in ('first', 'recomputed'):
        subprocess.run([sys.executable, '-m', 'external_baselines.patchcore_official_eval', 'evaluate',
            '--dataset', 'btad', '--category', '01', '--output-dir', str(tmp_path),
            '--cpu-check', '--evaluation-name', name], check=True)
    import csv
    results = []
    for name in ('first', 'recomputed'):
        with (tmp_path/'evaluations'/name/'category_metrics.csv').open() as f:
            results.append(list(csv.DictReader(f)))
    assert results[0] == results[1]
    assert float(results[0][0]['image_AUROC']) == 1
    assert float(results[0][0]['image_AP']) == 1
    metrics = [dict(category='01', image_AUROC=.8, image_AP=.7, pixel_AUROC=.6, pixel_AP=.5, AUPRO_at_0p3=.4)]
    fixed = [dict(category='01', fpr_cap=cap, actual_fpr=0, defect_pixel_recall='N/A',
                  region_mean_coverage='N/A', small_region_mean_coverage='N/A', region_count=0,
                  small_region_count=0, small_region_image_count=0) for cap in (.01, .05)]
    macro, fmacro = summarize(metrics, fixed, ['01'], 'btad')
    assert 'NOT_full_dataset' in macro[0]['scope']
    assert fmacro[0]['small_region_mean_coverage'] == 'N/A'
    assert fmacro[0]['small_region_mean_coverage_valid_category_count'] == 0


@pytest.mark.skipif(not os.getenv('PC_V1_WEIGHTS'), reason='Requires original V1 weights, no downloads in test')
def test_real_wrn_official_equivalence():
    from torchvision.models import wide_resnet50_2
    from external_baselines.patchcore_official_eval.official import verify_weights
    path = os.environ['PC_V1_WEIGHTS']
    verify_weights(path)
    seed_category()
    torch.set_num_threads(2)
    adapted = make_model(path, 'cpu', ChunkedFlatL2('cpu', 31))
    backbone = wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    reference = upstream(backbone, FaissNN(False, 2))
    images = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        a, b = adapted._embed(images), np.asarray(reference._embed(images))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    bank = np.ascontiguousarray(a[::10])
    adapted.anomaly_scorer.nn_method.fit(bank)
    reference.anomaly_scorer.fit([bank])
    a, b = adapted.predict(images), reference.predict(images)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=2e-5)


@pytest.mark.skipif(os.getenv('PC_DUAL_T4') != '1', reason='Explicit dual-GPU server validation required')
def test_two_cuda_devices_and_gpu_faiss():
    device0 = os.getenv('PC_MODEL_DEVICE', 'cuda:0')
    device1 = os.getenv('PC_CORESET_DEVICE', 'cuda:1')
    assert device0 != device1 and torch.cuda.device_count() >= 2
    assert 'T4' in torch.cuda.get_device_name(torch.device(device0))
    assert 'T4' in torch.cuda.get_device_name(torch.device(device1))
    seed_category()
    values = np.random.randn(73, 1024).astype(np.float32)
    sampler = ChunkedApproximateGreedy(device1, 19, 23)
    reduced = sampler.project(values)
    state = np.random.get_state()
    actual = sampler.select(reduced)
    np.random.set_state(state)
    reference = ApproximateGreedyCoresetSampler(.1, torch.device(device1), 10, 128)
    expected = reference._compute_greedy_coreset_indices(reduced)
    np.testing.assert_array_equal(actual, expected)
    nn = ChunkedFlatL2(device0, 7)
    nn.fit(values[actual])
    assert nn.search_index.getDevice() == torch.device(device0).index
    queries = np.random.randn(41, 1024).astype(np.float32)
    d1, i1 = nn.run(1, queries)
    nn.query_chunk = 1000
    d2, i2 = nn.run(1, queries)
    np.testing.assert_allclose(d1, d2, rtol=2e-5, atol=2e-4)
    np.testing.assert_array_equal(i1, i2)
