"""Memory/device wrappers around upstream PatchCore; never an alternative model."""
from contextlib import nullcontext
import numpy as np
import torch
from .official import install_import_path

install_import_path()
import faiss
from patchcore.common import FaissNN
from patchcore.patchcore import PatchCore
from patchcore.sampler import ApproximateGreedyCoresetSampler


def seed_category():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ChunkedFlatL2(FaissNN):
    """Same exact Flat L2, explicit GPU resource lifetime and query/add chunks."""
    def __init__(self, device, query_chunk=4096, temp_mb=256, add_chunk=65536):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU/CUDA supported')
        if self.device.type == 'cuda' and self.device.index is None:
            raise ValueError('Explicit CUDA index required')
        if min(query_chunk, add_chunk, temp_mb) <= 0:
            raise ValueError('Chunk sizes/temp_mb must be positive')
        super().__init__(self.device.type == 'cuda', 4)
        self.query_chunk, self.add_chunk = query_chunk, add_chunk
        self.resources = None
        if self.on_gpu:
            if not hasattr(faiss, 'StandardGpuResources'):
                raise RuntimeError('FAISS GPU required; CPU fallback is forbidden')
            with torch.cuda.device(self.device):
                self.resources = faiss.StandardGpuResources()
                self.resources.setTempMemory(int(temp_mb) * 1024**2)

    def _create_index(self, dimension):
        if not self.on_gpu:
            return faiss.IndexFlatL2(dimension)
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = self.device.index
        cfg.useFloat16 = False
        with torch.cuda.device(self.device):
            return faiss.GpuIndexFlatL2(self.resources, dimension, cfg)

    def _index_to_gpu(self, index):
        if not self.on_gpu:
            return index
        options = faiss.GpuClonerOptions()
        options.useFloat16 = False
        with torch.cuda.device(self.device):
            return faiss.index_cpu_to_gpu(self.resources, self.device.index, index, options)

    def fit(self, features):
        if features.dtype != np.float32 or features.ndim != 2 or not len(features):
            raise ValueError('Nonempty FP32 bank required')
        self.reset_index()
        self.search_index = self._create_index(features.shape[1])
        for offset in range(0, len(features), self.add_chunk):
            block = np.ascontiguousarray(features[offset:offset+self.add_chunk])
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite bank')
            self.search_index.add(block)

    def run(self, n_nearest_neighbours, query_features, index_features=None):
        if index_features is not None:
            raise ValueError('This wrapper uses only the persistent fitted bank')
        if self.search_index is None:
            raise RuntimeError('Load a saved bank before predict; automatic fit is forbidden')
        if query_features.dtype != np.float32:
            raise ValueError('FP32 queries required')
        distances = np.empty((len(query_features), n_nearest_neighbours), np.float32)
        indices = np.empty(distances.shape, np.int64)
        for offset in range(0, len(query_features), self.query_chunk):
            end = min(offset + self.query_chunk, len(query_features))
            d, i = super().run(n_nearest_neighbours, np.ascontiguousarray(query_features[offset:end]))
            distances[offset:end], indices[offset:end] = d, i
        return distances, indices


class ChunkedApproximateGreedy(ApproximateGreedyCoresetSampler):
    """Official dense random Linear + approximate greedy, with bounded temporaries.

    All reduced features and a single N-vector live on coreset_device. Only
    distance temporaries are chunked. Global argmax returns the first tied row,
    exactly as upstream; no tie perturbation or forced deduplication is added.
    """
    def __init__(self, device, projection_chunk=8192, distance_chunk=65536):
        super().__init__(0.1, torch.device(device), 10, 128)
        if min(projection_chunk, distance_chunk) <= 0:
            raise ValueError('Positive chunks required')
        self.projection_chunk, self.distance_chunk = projection_chunk, distance_chunk

    @torch.no_grad()
    def project(self, features, mapper=None):
        if features.dtype != np.float32 or features.ndim != 2:
            raise ValueError('CPU FP32 feature matrix required')
        # Match upstream initialization on CPU, then move the ONE mapper to GPU.
        if mapper is None:
            mapper = torch.nn.Linear(features.shape[1], 128, bias=False)
        mapper = mapper.to(self.device).eval()
        reduced = torch.empty((len(features), 128), dtype=torch.float32, device=self.device)
        for offset in range(0, len(features), self.projection_chunk):
            end = min(offset+self.projection_chunk, len(features))
            block = torch.from_numpy(np.array(features[offset:end], copy=True)).to(self.device)
            reduced[offset:end] = mapper(block)
        return reduced

    @torch.no_grad()
    def select(self, features, start_points=None):
        if features.dtype != torch.float32 or features.device != self.device:
            raise ValueError('Projected features must be FP32 on coreset_device')
        if start_points is None:
            start_points = np.random.choice(len(features), min(10, len(features)), replace=False).tolist()
        self.start_points = list(start_points)
        anchors = torch.empty((len(features), 1), device=self.device, dtype=torch.float32)
        starts = features[start_points]
        for offset in range(0, len(features), self.distance_chunk):
            end = min(offset+self.distance_chunk, len(features))
            anchors[offset:end] = self._compute_batchwise_differences(features[offset:end], starts).mean(-1, keepdim=True)
        count = int(len(features) * self.percentage)
        if count < 1:
            raise ValueError('Too few features for the fixed 0.1 coreset')
        indices = np.empty(count, np.int64)
        for step in range(count):
            chosen = torch.argmax(anchors).item()
            indices[step] = chosen
            point = features[chosen:chosen+1]
            for offset in range(0, len(features), self.distance_chunk):
                end = min(offset+self.distance_chunk, len(features))
                distance = self._compute_batchwise_differences(features[offset:end], point)
                torch.minimum(anchors[offset:end], distance, out=anchors[offset:end])
            if step % 1000 == 0 or step+1 == count:
                print(f'coreset {step+1}/{count}', flush=True)
        return indices


class StreamingPatchCore(PatchCore):
    def _embed(self, images, detach=True, provide_patch_shapes=False):
        # Avoid upstream's list of one CPU ndarray per patch. Mathematical path
        # remains upstream _embed(detach=False); transfer the batch once.
        result = super()._embed(images, detach=False, provide_patch_shapes=provide_patch_shapes)
        if provide_patch_shapes:
            values, shapes = result
            return (values.detach().cpu().numpy() if detach else values), shapes
        return result.detach().cpu().numpy() if detach else result


def make_model(weights_path, model_device, nn, backbone=None):
    if backbone is None:
        from torchvision.models import wide_resnet50_2
        backbone = wide_resnet50_2(weights=None)
        backbone.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
    backbone.name = 'wideresnet50'
    backbone.requires_grad_(False).eval()
    model = StreamingPatchCore(torch.device(model_device))
    model.load(backbone=backbone, layers_to_extract_from=['layer2', 'layer3'],
               device=torch.device(model_device), input_shape=(3, 256, 256),
               pretrain_embed_dimension=1024, target_embed_dimension=1024,
               patchsize=3, patchstride=1, anomaly_score_num_nn=1, nn_method=nn)
    model.requires_grad_(False).eval()
    return model


def extract_to_memmap(model, batches, image_count, path, measure=None):
    """Only one batch plus the disk-backed complete feature array; no concatenate."""
    measure = measure or (lambda name: nullcontext())
    cache, offset = None, 0
    for images in batches:
        with measure('feature_extraction'), torch.no_grad():
            features = model._embed(images.to(model.device, dtype=torch.float32))
        with measure('cpu_cache_write'):
            if not np.isfinite(features).all():
                raise FloatingPointError('Nonfinite official training embedding')
            if cache is None:
                per_image, rem = divmod(len(features), len(images))
                if rem or features.shape[1] != 1024:
                    raise ValueError('Unexpected official embedding shape')
                cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                                  shape=(image_count*per_image, 1024))
            cache[offset:offset+len(features)] = features
            offset += len(features)
            cache.flush()
        del features
    if cache is None or offset != len(cache):
        raise ValueError('Incomplete feature extraction')
    return cache


def save_selected_bank(features, indices, path, chunk=8192):
    bank = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(indices), 1024))
    for offset in range(0, len(indices), chunk):
        bank[offset:offset+chunk] = features[indices[offset:offset+chunk]]
    bank.flush()
    del bank


def load_bank(model, path):
    bank = np.load(path, mmap_mode='r', allow_pickle=False)
    if bank.ndim != 2 or bank.shape[1] != 1024 or bank.dtype != np.float32:
        raise ValueError('Expected the saved FP32 1024-D bank')
    # A single 1024-D feature stream: official ConcatMerger is identity here.
    # Do not retain its otherwise unnecessary complete concatenate copy.
    model.anomaly_scorer.nn_method.fit(bank)
