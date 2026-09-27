from __future__ import annotations

import sys
from types import MethodType
from typing import Callable

import torch
import torch.nn.functional as F


def _patch_dinov3_torch_compat() -> None:
    """Allow the pinned DINOv3 source to run on PyTorch versions without one
    removed dynamo config key. This only restores the source's harmless cache
    limit setting; it does not alter model weights or attention math.
    """
    import torch._dynamo.config as dynamo_config

    key = "accumulated_cache_size_limit"
    config = getattr(dynamo_config, "_config", None)
    if isinstance(config, dict):
        # ConfigModule.__setattr__ validates against this mapping. `hasattr()`
        # alone is insufficient because some versions expose a fallback
        # __getattr__ without registering the key for assignment.
        config.setdefault(key, 1024)
        default = getattr(dynamo_config, "_default", None)
        if isinstance(default, dict):
            default.setdefault(key, 1024)
    elif not hasattr(dynamo_config, key):
        raise RuntimeError(
            "DINOv3 source expects torch._dynamo.config.accumulated_cache_size_limit, "
            "but this PyTorch runtime exposes no compatible config mapping"
        )

    # PyTorch 2.1's ConfigModule validates assignments against an internal
    # allow-list in addition to `_config`. Registering the key above is not
    # sufficient there, so install a narrowly scoped compatibility setter for
    # this one source-level cache option.
    config_type = type(dynamo_config)
    if not getattr(config_type, "_nsrm_accumulated_cache_compat", False):
        original_setattr = config_type.__setattr__

        def _compat_setattr(instance, name, value):
            if instance is dynamo_config and name == key:
                instance.__dict__[name] = value
                return
            return original_setattr(instance, name, value)

        config_type.__setattr__ = _compat_setattr
        config_type._nsrm_accumulated_cache_compat = True


def load_dinov3_vitb16(
    repo: str,
    checkpoint: str,
    device: torch.device,
    input_size: int = 512,
    use_half: bool = True,
):
    repo_path = str(repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    _patch_dinov3_torch_compat()
    from dinov3.hub.backbones import dinov3_vitb16

    model = dinov3_vitb16(
        pretrained=True,
        weights=str(checkpoint),
    )
    model.eval().to(device)
    if use_half and device.type == "cuda":
        model.half()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_patch_features(model, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    output = model.forward_features(image)
    patch = output["x_norm_patchtokens"]
    b, n, _ = patch.shape
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"DINOv3 patch token count is not square: {n}")
    return F.normalize(patch.float(), dim=-1), (side, side)


def _capture_and_call_original(self, qkv, attn_bias=None, rope=None):
    if attn_bias is not None:
        raise ValueError("NSRM attention capture does not support attention bias")
    batch, tokens, _ = qkv.shape
    original_qkv = qkv
    channels = self.qkv.in_features
    qkv = qkv.reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [value.transpose(1, 2) for value in (q, k, v)]
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(self.scale)
    weights = torch.softmax(logits, dim=-1)
    self._nsrm_last_attention = weights.detach()
    # Return the official implementation's output. The capture is a side
    # channel only; it must not alter the hidden states consumed by later blocks.
    return self._nsrm_original_compute_attention(original_qkv, attn_bias=attn_bias, rope=rope)


class AttentionCapture:
    def __init__(self, model, layer_index: int):
        self.model = model
        self.layer_index = int(layer_index)
        self.module = model.blocks[self.layer_index].attn
        self.original: Callable | None = None

    def __enter__(self):
        self.original = self.module.compute_attention
        self.module._nsrm_original_compute_attention = self.original
        self.module.compute_attention = MethodType(_capture_and_call_original, self.module)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            self.module.compute_attention = self.original
            if hasattr(self.module, "_nsrm_original_compute_attention"):
                del self.module._nsrm_original_compute_attention
        # The caller keeps its own tensor reference when it needs the
        # captured weights.  Do not leave the last full attention matrix
        # attached to the frozen model between stages.
        if hasattr(self.module, "_nsrm_last_attention"):
            del self.module._nsrm_last_attention
        return False

    @property
    def weights(self) -> torch.Tensor:
        if not hasattr(self.module, "_nsrm_last_attention"):
            raise RuntimeError("No attention was captured; run the model first")
        return self.module._nsrm_last_attention


class MultiAttentionCapture:
    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = tuple(sorted({int(index) for index in layer_indices}))
        self.modules = {index: model.blocks[index].attn for index in self.layer_indices}
        self.original = {}

    def __enter__(self):
        for index, module in self.modules.items():
            self.original[index] = module.compute_attention
            module._nsrm_original_compute_attention = self.original[index]
            module.compute_attention = MethodType(_capture_and_call_original, module)
        return self

    def __exit__(self, exc_type, exc, tb):
        for index, module in self.modules.items():
            if index in self.original:
                module.compute_attention = self.original[index]
                if hasattr(module, "_nsrm_original_compute_attention"):
                    del module._nsrm_original_compute_attention
            if hasattr(module, "_nsrm_last_attention"):
                del module._nsrm_last_attention
        return False

    @property
    def weights(self) -> dict[int, torch.Tensor]:
        missing = [index for index, module in self.modules.items() if not hasattr(module, "_nsrm_last_attention")]
        if missing:
            raise RuntimeError(f"No attention captured for layers: {missing}")
        return {index: module._nsrm_last_attention for index, module in self.modules.items()}


@torch.inference_mode()
def extract_attention(
    model,
    image: torch.Tensor,
    layer_index: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    with AttentionCapture(model, layer_index) as capture:
        output = model.forward_features(image)
        patch = output["x_norm_patchtokens"]
        weights = capture.weights
    tokens = int(patch.shape[1])
    side = int(tokens**0.5)
    if side * side != tokens:
        raise ValueError(f"DINOv3 patch token count is not square: {tokens}")
    return weights.float(), (side, side)


@torch.inference_mode()
def extract_features_attention_layers(
    model,
    image: torch.Tensor,
    layer_indices,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], tuple[int, int]]:
    with MultiAttentionCapture(model, layer_indices) as capture:
        output = model.forward_features(image)
        patch = output["x_norm_patchtokens"]
        weights = {index: value.float() for index, value in capture.weights.items()}
    tokens = int(patch.shape[1])
    side = int(tokens**0.5)
    if side * side != tokens:
        raise ValueError(f"DINOv3 patch token count is not square: {tokens}")
    return F.normalize(patch.float(), dim=-1), weights, (side, side)


@torch.inference_mode()
def extract_attention_layers(
    model,
    image: torch.Tensor,
    layer_indices,
) -> tuple[dict[int, torch.Tensor], tuple[int, int]]:
    _, weights, grid = extract_features_attention_layers(model, image, layer_indices)
    return weights, grid
