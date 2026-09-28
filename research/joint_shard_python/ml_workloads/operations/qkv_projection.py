"""Three independent attention projections, without a fused QKV weight."""

from dataclasses import dataclass
from typing import ClassVar

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class QKVProjectionConfig:
    tokens: int = 128
    hidden: int = 256
    projection: int = 256
    layout: str = "output"
    seed: int = 0


@dataclass(frozen=True)
class QKVProjectionInput(BaseOperationInput):
    x: jax.Array
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.wq, self.wk, self.wv)


class QKVProjection(ShardedOperation[QKVProjectionInput, QKVProjectionConfig]):
    layouts: ClassVar[tuple[str, ...]] = ("output", "hidden", "tokens", "replicated")

    def __init__(self, *, mesh: Mesh | None = None) -> None:
        super().__init__("qkv_projection", mesh=mesh)

    def create_inputs(self, config: QKVProjectionConfig) -> QKVProjectionInput:
        validate_positive_dimensions(
            tokens=config.tokens, hidden=config.hidden, projection=config.projection
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        x_spec = {
            "output": P(),
            "hidden": P(None, "model"),
            "tokens": P("model", None),
            "replicated": P(),
        }[config.layout]
        w_spec = {
            "output": P(None, "model"),
            "hidden": P("model", None),
            "tokens": P(),
            "replicated": P(),
        }[config.layout]

        def weight() -> jax.Array:
            return build.random_normal(
                (config.hidden, config.projection), w_spec, scale=config.hidden**-0.5
            )

        return QKVProjectionInput(
            build.random_normal((config.tokens, config.hidden), x_spec),
            weight(),
            weight(),
            weight(),
        )

    def _operation(
        self, x: jax.Array, wq: jax.Array, wk: jax.Array, wv: jax.Array
    ) -> tuple[jax.Array, ...]:
        return (x @ wq, x @ wk, x @ wv)
