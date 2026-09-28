"""Workload implementations and their shared base classes."""

from .base import BaseOperation, BaseOperationInput
from .megatron_layer import MegatronLayer, MegatronLayerConfig, MegatronLayerInput

__all__ = [
    "BaseOperation",
    "BaseOperationInput",
    "MegatronLayer",
    "MegatronLayerConfig",
    "MegatronLayerInput",
]
