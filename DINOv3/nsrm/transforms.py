from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from .protocol import TRANSFORMS


def apply_transform(image: Image.Image, transform_id: int) -> Image.Image:
    if not 0 <= int(transform_id) < len(TRANSFORMS):
        raise ValueError(f"Unknown NSRM transform id: {transform_id}")
    spec = TRANSFORMS[int(transform_id)]
    name = str(spec["name"])
    image = image.convert("RGB")
    if name == "identity":
        return image
    if name.startswith("brightness-"):
        factor = float(spec["value"])
        array = np.asarray(image, dtype=np.float32) * factor
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if name.startswith("gamma-"):
        gamma = float(spec["value"])
        array = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(
            np.clip(np.power(array, gamma) * 255.0, 0, 255).astype(np.uint8), "RGB"
        )
    if name == "white-balance-warm" or name == "white-balance-cool":
        gains = np.asarray(spec["value"], dtype=np.float32).reshape(1, 1, 3)
        array = np.asarray(image, dtype=np.float32) * gains
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if name == "gaussian-blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(spec["value"])))
    if name == "jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(spec["value"]), subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    raise ValueError(f"Unsupported NSRM transform: {name}")
