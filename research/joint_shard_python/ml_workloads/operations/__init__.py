"""Workload implementations and their shared base classes."""

from .barlow_twins import BarlowTwins, BarlowTwinsConfig, BarlowTwinsInput
from .base import BaseOperation, BaseOperationInput
from .context_attention import (
    ContextAttention,
    ContextAttentionConfig,
    ContextAttentionInput,
)
from .gated_mlp import GatedMLP, GatedMLPConfig, GatedMLPInput
from .graph_convolution import (
    GraphConvolution,
    GraphConvolutionConfig,
    GraphConvolutionInput,
)
from .kernel_linear_attention import (
    KernelLinearAttention,
    KernelLinearAttentionConfig,
    KernelLinearAttentionInput,
)
from .layernorm_linear import (
    LayerNormLinear,
    LayerNormLinearConfig,
    LayerNormLinearInput,
)
from .lora import LoRA, LoRAConfig, LoRAInput
from .megatron_layer import MegatronLayer, MegatronLayerConfig, MegatronLayerInput
from .moe_router import MoERouter, MoERouterConfig, MoERouterInput
from .output_projection_ce import (
    OutputProjectionCE,
    OutputProjectionCEConfig,
    OutputProjectionCEInput,
)
from .qkv_projection import QKVProjection, QKVProjectionConfig, QKVProjectionInput
from .reduce_dot import ReduceDot, ReduceDotConfig, ReduceDotInput

OPERATIONS: dict[str, tuple[type[BaseOperation], type]] = {
    "megatron_layer": (MegatronLayer, MegatronLayerConfig),
    "output_projection_ce": (OutputProjectionCE, OutputProjectionCEConfig),
    "context_attention": (ContextAttention, ContextAttentionConfig),
    "layernorm_linear": (LayerNormLinear, LayerNormLinearConfig),
    "reduce_dot": (ReduceDot, ReduceDotConfig),
    "lora": (LoRA, LoRAConfig),
    "qkv_projection": (QKVProjection, QKVProjectionConfig),
    "gated_mlp": (GatedMLP, GatedMLPConfig),
    "kernel_linear_attention": (KernelLinearAttention, KernelLinearAttentionConfig),
    "graph_convolution": (GraphConvolution, GraphConvolutionConfig),
    "barlow_twins": (BarlowTwins, BarlowTwinsConfig),
    "moe_router": (MoERouter, MoERouterConfig),
}

__all__ = [
    "OPERATIONS",
    "BaseOperation",
    "BaseOperationInput",
    "BarlowTwins",
    "BarlowTwinsConfig",
    "BarlowTwinsInput",
    "ContextAttention",
    "ContextAttentionConfig",
    "ContextAttentionInput",
    "GatedMLP",
    "GatedMLPConfig",
    "GatedMLPInput",
    "GraphConvolution",
    "GraphConvolutionConfig",
    "GraphConvolutionInput",
    "KernelLinearAttention",
    "KernelLinearAttentionConfig",
    "KernelLinearAttentionInput",
    "LayerNormLinear",
    "LayerNormLinearConfig",
    "LayerNormLinearInput",
    "LoRA",
    "LoRAConfig",
    "LoRAInput",
    "MegatronLayer",
    "MegatronLayerConfig",
    "MegatronLayerInput",
    "MoERouter",
    "MoERouterConfig",
    "MoERouterInput",
    "OutputProjectionCE",
    "OutputProjectionCEConfig",
    "OutputProjectionCEInput",
    "QKVProjection",
    "QKVProjectionConfig",
    "QKVProjectionInput",
    "ReduceDot",
    "ReduceDotConfig",
    "ReduceDotInput",
]
