"""Sequence reduction followed by projection: the original algebraic control."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class ReduceDotConfig:
    batch: int = 2
    sequence: int = 128
    hidden: int = 256
    output: int = 64
    layout: str = "sequence"
    seed: int = 0


@dataclass(frozen=True)
class ReduceDotInput(BaseOperationInput):
    x: jax.Array
    weight: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.weight)


class ReduceDot(ShardedOperation[ReduceDotInput, ReduceDotConfig]):
    layouts: ClassVar[tuple[str, ...]] = ("sequence", "hidden", "replicated")

    def __init__(self, *, mesh: Mesh | None = None) -> None:
        super().__init__("reduce_dot", mesh=mesh)

    def create_inputs(self, config: ReduceDotConfig) -> ReduceDotInput:
        validate_positive_dimensions(
            batch=config.batch,
            sequence=config.sequence,
            hidden=config.hidden,
            output=config.output,
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        x_spec = {
            "sequence": P(None, "model", None),
            "hidden": P(None, None, "model"),
            "replicated": P(),
        }[config.layout]
        return ReduceDotInput(
            x=build.random_normal(
                (config.batch, config.sequence, config.hidden), x_spec
            ),
            weight=build.random_normal(
                (config.hidden, config.output),
                P("model", None) if config.layout == "hidden" else P(),
                scale=config.hidden**-0.5,
            ),
        )

    def _operation(self, x: jax.Array, weight: jax.Array) -> jax.Array:
        return jnp.sum(x, axis=1) @ weight
