"""Kernel attention baseline that explicitly forms the query-key matrix."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class KernelLinearAttentionConfig:
    batch: int = 2
    queries: int = 128
    sequence: int = 256
    feature_rank: int = 32
    value_dim: int = 64
    layout: str = "sequence"
    seed: int = 0


@dataclass(frozen=True)
class KernelLinearAttentionInput(BaseOperationInput):
    q: jax.Array
    k: jax.Array
    v: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.q, self.k, self.v)


class KernelLinearAttention(
    ShardedOperation[KernelLinearAttentionInput, KernelLinearAttentionConfig]
):
    """Use the fixed positive feature map phi(x) = max(x, 0) + 1."""

    layouts: ClassVar[tuple[str, ...]] = ("sequence", "queries", "replicated")

    def __init__(self, *, mesh: Mesh | None = None) -> None:
        super().__init__("kernel_linear_attention", mesh=mesh)

    def create_inputs(
        self, config: KernelLinearAttentionConfig
    ) -> KernelLinearAttentionInput:
        validate_positive_dimensions(
            batch=config.batch,
            queries=config.queries,
            sequence=config.sequence,
            feature_rank=config.feature_rank,
            value_dim=config.value_dim,
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        kv_spec = P(None, "model", None) if config.layout == "sequence" else P()
        return KernelLinearAttentionInput(
            q=build.random_normal(
                (config.batch, config.queries, config.feature_rank),
                P(None, "model", None) if config.layout == "queries" else P(),
            ),
            k=build.random_normal(
                (config.batch, config.sequence, config.feature_rank), kv_spec
            ),
            v=build.random_normal(
                (config.batch, config.sequence, config.value_dim), kv_spec
            ),
        )

    def _operation(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        q_features = jnp.maximum(q, 0) + 1
        k_features = jnp.maximum(k, 0) + 1
        kernel = q_features @ jnp.swapaxes(k_features, -1, -2)
        return (kernel @ v) / jnp.sum(kernel, axis=-1, keepdims=True)
