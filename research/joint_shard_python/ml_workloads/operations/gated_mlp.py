"""Separate gate and value projections in a SwiGLU feed-forward block."""

from dataclasses import dataclass
from typing import ClassVar

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class GatedMLPConfig:
    tokens: int = 128
    hidden: int = 256
    ffn: int = 512
    output: int = 256
    layout: str = "ffn"
    seed: int = 0


@dataclass(frozen=True)
class GatedMLPInput(BaseOperationInput):
    x: jax.Array
    gate: jax.Array
    up: jax.Array
    down: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.gate, self.up, self.down)


class GatedMLP(ShardedOperation[GatedMLPInput, GatedMLPConfig]):
    layouts: ClassVar[tuple[str, ...]] = ("ffn", "hidden", "tokens", "replicated")

    def __init__(self, *, mesh: Mesh | None = None) -> None:
        super().__init__("gated_mlp", mesh=mesh)

    def create_inputs(self, config: GatedMLPConfig) -> GatedMLPInput:
        validate_positive_dimensions(
            tokens=config.tokens,
            hidden=config.hidden,
            ffn=config.ffn,
            output=config.output,
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        x_spec = {
            "ffn": P(),
            "hidden": P(None, "model"),
            "tokens": P("model", None),
            "replicated": P(),
        }[config.layout]
        w_spec = {
            "ffn": P(None, "model"),
            "hidden": P("model", None),
            "tokens": P(),
            "replicated": P(),
        }[config.layout]
        return GatedMLPInput(
            x=build.random_normal((config.tokens, config.hidden), x_spec),
            gate=build.random_normal(
                (config.hidden, config.ffn), w_spec, scale=config.hidden**-0.5
            ),
            up=build.random_normal(
                (config.hidden, config.ffn), w_spec, scale=config.hidden**-0.5
            ),
            down=build.random_normal(
                (config.ffn, config.output),
                P("model", None) if config.layout == "ffn" else P(),
                scale=config.ffn**-0.5,
            ),
        )

    def _operation(
        self, x: jax.Array, gate: jax.Array, up: jax.Array, down: jax.Array
    ) -> jax.Array:
        gate_values = x @ gate
        activated = gate_values * jax.lax.logistic(gate_values)
        return (activated * (x @ up)) @ down
