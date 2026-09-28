"""Full-score scaled dot-product attention over an unmasked context."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class ContextAttentionConfig:
    batch: int = 2
    queries: int = 64
    sequence: int = 256
    head_dim: int = 64
    value_dim: int = 64
    layout: str = "sequence"
    seed: int = 0


@dataclass(frozen=True)
class ContextAttentionInput(BaseOperationInput):
    q: jax.Array
    k: jax.Array
    v: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.q, self.k, self.v)


class ContextAttention(ShardedOperation[ContextAttentionInput, ContextAttentionConfig]):
    layouts: ClassVar[tuple[str, ...]] = ("sequence", "queries", "replicated")
    supports_full_tensor_boundary: ClassVar[bool] = True

    def __init__(
        self, *, mesh: Mesh | None = None, full_tensor_boundary: bool = False
    ) -> None:
        super().__init__(
            "context_attention", mesh=mesh, full_tensor_boundary=full_tensor_boundary
        )

    def create_inputs(self, config: ContextAttentionConfig) -> ContextAttentionInput:
        validate_positive_dimensions(
            batch=config.batch,
            queries=config.queries,
            sequence=config.sequence,
            head_dim=config.head_dim,
            value_dim=config.value_dim,
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        kv_spec = P(None, "model", None) if config.layout == "sequence" else P()
        return ContextAttentionInput(
            q=build.random_normal(
                (config.batch, config.queries, config.head_dim),
                P(None, "model", None) if config.layout == "queries" else P(),
            ),
            k=build.random_normal(
                (config.batch, config.sequence, config.head_dim), kv_spec
            ),
            v=build.random_normal(
                (config.batch, config.sequence, config.value_dim), kv_spec
            ),
        )

    def _operation(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        k, v = (
            self.constrain_replication_if_enabled(k),
            self.constrain_replication_if_enabled(v),
        )
        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(
            jnp.asarray(q.shape[-1], dtype=q.dtype)
        )
        return jax.nn.softmax(scores, axis=-1) @ v
