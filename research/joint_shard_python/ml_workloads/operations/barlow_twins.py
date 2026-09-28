"""Barlow Twins loss through the complete feature cross-correlation matrix."""

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
class BarlowTwinsConfig:
    samples: int = 128
    features: int = 1024
    layout: str = "features"
    seed: int = 0


@dataclass(frozen=True)
class BarlowTwinsInput(BaseOperationInput):
    a: jax.Array
    b: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.a, self.b)


class BarlowTwins(ShardedOperation[BarlowTwinsInput, BarlowTwinsConfig]):
    """Population normalization, diagonal-to-one penalty, and off-diagonal penalty."""

    layouts: ClassVar[tuple[str, ...]] = ("features", "samples", "replicated")

    def __init__(
        self,
        *,
        epsilon: float = 1e-5,
        off_diagonal_weight: float = 0.005,
        mesh: Mesh | None = None,
    ) -> None:
        validate_positive_epsilon(epsilon)
        if not np.isfinite(off_diagonal_weight) or off_diagonal_weight < 0:
            raise ValueError("off_diagonal_weight must be finite and nonnegative")
        self.epsilon = epsilon
        self.off_diagonal_weight = off_diagonal_weight
        super().__init__("barlow_twins", mesh=mesh)

    def create_inputs(self, config: BarlowTwinsConfig) -> BarlowTwinsInput:
        validate_positive_dimensions(samples=config.samples, features=config.features)
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        spec = {
            "features": P(None, "model"),
            "samples": P("model", None),
            "replicated": P(),
        }[config.layout]
        return BarlowTwinsInput(
            build.random_normal((config.samples, config.features), spec),
            build.random_normal((config.samples, config.features), spec),
        )

    def _operation(self, a: jax.Array, b: jax.Array) -> jax.Array:
        def normalize(x: jax.Array) -> jax.Array:
            centered = x - jnp.mean(x, axis=0, keepdims=True)
            variance = jnp.mean(centered * centered, axis=0, keepdims=True)
            return centered / jnp.sqrt(variance + self.epsilon)

        a, b = normalize(a), normalize(b)
        correlation = (a.T @ b) / a.shape[0]
        feature_ids = np.arange(a.shape[1], dtype=np.int32)
        diagonal = jax.lax.gather(
            correlation,
            jnp.asarray(np.column_stack((feature_ids, feature_ids))),
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0, 1),
                start_index_map=(0, 1),
            ),
            slice_sizes=(1, 1),
            mode="promise_in_bounds",
        )
        diagonal_loss = jnp.sum((diagonal - 1) ** 2)
        off_diagonal_loss = jnp.sum(correlation * correlation) - jnp.sum(
            diagonal * diagonal
        )
        return diagonal_loss + self.off_diagonal_weight * off_diagonal_loss
