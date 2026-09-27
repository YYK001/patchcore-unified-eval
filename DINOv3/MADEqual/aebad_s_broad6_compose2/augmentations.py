from __future__ import annotations

import io
import json
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


SEED = 42
POLICY_NAME = "C12_BroadCompose2"
BROAD6 = (
    "AutoContrast",
    "Gamma",
    "Contrast",
    "WhiteBalance",
    "JPEG",
    "IlluminationField",
)


def normalize_relative_path(value: str) -> str:
    return value.replace("\\", "/")


def stable_seed(tag: str, relative_path: str, *, seed: int = SEED) -> int:
    path = normalize_relative_path(relative_path)
    return zlib.crc32(f"{seed}|{tag}|{path}".encode("utf-8")) & 0xFFFFFFFF


def choose_broad_compose2(relative_path: str, *, seed: int = SEED) -> tuple[str, str]:
    rng = np.random.default_rng(
        stable_seed(f"ASSIGN::{POLICY_NAME}", relative_path, seed=seed)
    )
    chosen = rng.choice(len(BROAD6), size=2, replace=False)
    return BROAD6[int(chosen[0])], BROAD6[int(chosen[1])]


def atomic_params(op: str, relative_path: str, *, seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(op, relative_path, seed=seed))
    if op == "AutoContrast":
        return {}
    if op == "Gamma":
        return {"gamma": float(rng.uniform(0.8, 1.25))}
    if op == "Contrast":
        return {"factor": float(rng.uniform(0.8, 1.2))}
    if op == "WhiteBalance":
        gains = rng.uniform(0.9, 1.1, size=3).astype(np.float64)
        gains /= float(np.prod(gains) ** (1.0 / 3.0))
        return {"r": float(gains[0]), "g": float(gains[1]), "b": float(gains[2])}
    if op == "JPEG":
        return {"quality": int(rng.integers(70, 96))}
    if op == "IlluminationField":
        return {
            "amplitude": float(rng.uniform(0.1, 0.2)),
            "angle": float(rng.uniform(0.0, 2.0 * np.pi)),
        }
    raise ValueError(f"unsupported Broad6 operation: {op}")


def apply_atomic(rgb: np.ndarray, op: str, params: dict[str, Any]) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Broad6 input must be an RGB array")
    if op == "AutoContrast":
        return np.asarray(ImageOps.autocontrast(Image.fromarray(value, mode="RGB")), dtype=np.uint8)
    if op == "Gamma":
        output = 255.0 * np.power(value.astype(np.float32) / 255.0, float(params["gamma"]))
        return np.clip(output, 0, 255).astype(np.uint8)
    if op == "Contrast":
        image = ImageEnhance.Contrast(Image.fromarray(value, mode="RGB")).enhance(
            float(params["factor"])
        )
        return np.asarray(image, dtype=np.uint8)
    if op == "WhiteBalance":
        gains = np.asarray([params["r"], params["g"], params["b"]], dtype=np.float32)
        return np.clip(value.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
    if op == "JPEG":
        buffer = io.BytesIO()
        Image.fromarray(value, mode="RGB").save(
            buffer, format="JPEG", quality=int(params["quality"])
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    if op == "IlluminationField":
        height, width = value.shape[:2]
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, height, dtype=np.float32),
            np.linspace(-1.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        angle = float(params["angle"])
        field = np.cos(angle) * xx + np.sin(angle) * yy
        denominator = float(np.max(np.abs(field)))
        if denominator > 0:
            field /= denominator
        gain = 1.0 + float(params["amplitude"]) * field
        return np.clip(
            value.astype(np.float32) * gain[:, :, None], 0, 255
        ).astype(np.uint8)
    raise ValueError(f"unsupported Broad6 operation: {op}")


@dataclass(frozen=True)
class Compose2Result:
    image: Image.Image
    op1: str
    op2: str
    params1: dict[str, Any]
    params2: dict[str, Any]

    def manifest_row(self, relative_path: str) -> dict[str, Any]:
        return {
            "relative_path": normalize_relative_path(relative_path),
            "op1": self.op1,
            "op2": self.op2,
            "recipe": f"{self.op1}>{self.op2}",
            "op1_params": json.dumps(self.params1, sort_keys=True, separators=(",", ":")),
            "op2_params": json.dumps(self.params2, sort_keys=True, separators=(",", ":")),
        }


def broad6_compose2(image: Image.Image, relative_path: str, *, seed: int = SEED) -> Compose2Result:
    path = normalize_relative_path(relative_path)
    op1, op2 = choose_broad_compose2(path, seed=seed)
    params1 = atomic_params(op1, path, seed=seed)
    params2 = atomic_params(op2, path, seed=seed)
    value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    value = apply_atomic(value, op1, params1)
    value = apply_atomic(value, op2, params2)
    return Compose2Result(
        image=Image.fromarray(value, mode="RGB"),
        op1=op1,
        op2=op2,
        params1=params1,
        params2=params2,
    )


def compose2_manifest_row(relative_path: str, *, seed: int = SEED) -> dict[str, Any]:
    path = normalize_relative_path(relative_path)
    op1, op2 = choose_broad_compose2(path, seed=seed)
    return {
        "relative_path": path,
        "op1": op1,
        "op2": op2,
        "recipe": f"{op1}>{op2}",
        "op1_params": json.dumps(
            atomic_params(op1, path, seed=seed), sort_keys=True, separators=(",", ":")
        ),
        "op2_params": json.dumps(
            atomic_params(op2, path, seed=seed), sort_keys=True, separators=(",", ":")
        ),
    }
