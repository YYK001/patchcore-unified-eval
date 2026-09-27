"""Conditional normal-variation subspaces for DINOv2 anomaly detection.

This package intentionally does not modify the historical ``nvs.detection``
pipeline.  New experiments use the D0/D1/D2/D3 definitions exported here.
"""

from .pipeline import (
    CORE_METHODS,
    ConditionalNVSPipeline,
    FeatureSplit,
)
from .protocol import CalibrationState, ThreeWaySplit, split_three_way

__all__ = [
    "CORE_METHODS",
    "CalibrationState",
    "ConditionalNVSPipeline",
    "FeatureSplit",
    "ThreeWaySplit",
    "split_three_way",
]
