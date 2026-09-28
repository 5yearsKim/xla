"""Centered-variance LayerNorm with affine parameters followed by a linear map."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import (
    ShardedInputBuilder,
    ShardedOperation,
    validate_positive_dimensions,
    validate_positive_epsilon,
)
from .base import BaseOperationInput


@dataclass(frozen=True)
class LayerNormLinearConfig:
    tokens: int = 128
    hidden: int = 256
    output: int = 128
    layout: str = "hidden"
    seed: int = 0


@dataclass(frozen=True)
class LayerNormLinearInput(BaseOperationInput):
    x: jax.Array
    scale: jax.Array
    bias: jax.Array
    weight: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.scale, self.bias, self.weight)


class LayerNormLinear(ShardedOperation[LayerNormLinearInput, LayerNormLinearConfig]):
    layouts: ClassVar[tuple[str, ...]] = ("hidden", "tokens", "replicated")
    supports_full_tensor_boundary: ClassVar[bool] = True

    def __init__(
        self,
        *,
        epsilon: float = 1e-5,
        mesh: Mesh | None = None,
        full_tensor_boundary: bool = False,
    ) -> None:
        validate_positive_epsilon(epsilon)
        self.epsilon = epsilon
        super().__init__(
            "layernorm_linear", mesh=mesh, full_tensor_boundary=full_tensor_boundary
        )

    def create_inputs(self, config: LayerNormLinearConfig) -> LayerNormLinearInput:
        validate_positive_dimensions(
            tokens=config.tokens, hidden=config.hidden, output=config.output
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        hidden = config.layout == "hidden"
        x_spec = (
            P(None, "model")
            if hidden
            else P("model", None)
            if config.layout == "tokens"
            else P()
        )
        return LayerNormLinearInput(
            x=build.random_normal((config.tokens, config.hidden), x_spec),
            scale=build.place_on_mesh(
                np.ones(config.hidden, dtype=np.float32), P("model") if hidden else P()
            ),
            bias=build.place_on_mesh(
                np.zeros(config.hidden, dtype=np.float32), P("model") if hidden else P()
            ),
            weight=build.random_normal(
                (config.hidden, config.output),
                P("model", None) if hidden else P(),
                scale=config.hidden**-0.5,
            ),
        )

    def _operation(
        self, x: jax.Array, scale: jax.Array, bias: jax.Array, weight: jax.Array
    ) -> jax.Array:
        x = self.constrain_replication_if_enabled(x)
        centered = x - jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered / jnp.sqrt(variance + self.epsilon)
        return (normalized * scale + bias) @ weight
