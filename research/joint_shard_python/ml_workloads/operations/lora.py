"""LoRA baseline that constructs a dense effective weight before projection."""

from dataclasses import dataclass
from typing import ClassVar

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class LoRAConfig:
    tokens: int = 128
    hidden: int = 256
    output: int = 256
    rank: int = 16
    layout: str = "output"
    seed: int = 0


@dataclass(frozen=True)
class LoRAInput(BaseOperationInput):
    x: jax.Array
    weight: jax.Array
    a: jax.Array
    b: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.weight, self.a, self.b)


class LoRA(ShardedOperation[LoRAInput, LoRAConfig]):
    """Compute X(W + AB), with unit adapter scaling and no dropout."""

    layouts: ClassVar[tuple[str, ...]] = ("output", "hidden", "tokens", "replicated")

    def __init__(self, *, mesh: Mesh | None = None) -> None:
        super().__init__("lora", mesh=mesh)

    def create_inputs(self, config: LoRAConfig) -> LoRAInput:
        validate_positive_dimensions(
            tokens=config.tokens,
            hidden=config.hidden,
            output=config.output,
            rank=config.rank,
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
        return LoRAInput(
            x=build.random_normal((config.tokens, config.hidden), x_spec),
            weight=build.random_normal(
                (config.hidden, config.output), w_spec, scale=config.hidden**-0.5
            ),
            a=build.random_normal(
                (config.hidden, config.rank),
                P("model", None) if config.layout == "hidden" else P(),
                scale=config.hidden**-0.5,
            ),
            b=build.random_normal(
                (config.rank, config.output),
                P(None, "model") if config.layout == "output" else P(),
                scale=config.rank**-0.5,
            ),
        )

    def _operation(
        self, x: jax.Array, weight: jax.Array, a: jax.Array, b: jax.Array
    ) -> jax.Array:
        return x @ (weight + a @ b)
