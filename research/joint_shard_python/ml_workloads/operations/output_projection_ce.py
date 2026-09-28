"""Complete vocabulary projection followed by stable cross-entropy."""

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
    gather_columns_per_row,
    validate_positive_dimensions,
)
from .base import BaseOperationInput


@dataclass(frozen=True)
class OutputProjectionCEConfig:
    tokens: int = 128
    hidden: int = 256
    vocabulary: int = 1024
    layout: str = "vocabulary"
    seed: int = 0


@dataclass(frozen=True)
class OutputProjectionCEInput(BaseOperationInput):
    x: jax.Array
    weight: jax.Array
    targets: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.weight, self.targets)


class OutputProjectionCE(
    ShardedOperation[OutputProjectionCEInput, OutputProjectionCEConfig]
):
    """Return mean token loss; keep the full logits computation in the source."""

    layouts: ClassVar[tuple[str, ...]] = ("vocabulary", "tokens", "replicated")
    supports_full_tensor_boundary: ClassVar[bool] = True

    def __init__(
        self, *, mesh: Mesh | None = None, full_tensor_boundary: bool = False
    ) -> None:
        super().__init__(
            "output_projection_ce", mesh=mesh, full_tensor_boundary=full_tensor_boundary
        )

    def create_inputs(
        self, config: OutputProjectionCEConfig
    ) -> OutputProjectionCEInput:
        validate_positive_dimensions(
            tokens=config.tokens, hidden=config.hidden, vocabulary=config.vocabulary
        )
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        tokens = config.layout == "tokens"
        return OutputProjectionCEInput(
            x=build.random_normal(
                (config.tokens, config.hidden), P("model", None) if tokens else P()
            ),
            weight=build.random_normal(
                (config.hidden, config.vocabulary),
                P(None, "model") if config.layout == "vocabulary" else P(),
                scale=config.hidden**-0.5,
            ),
            targets=build.place_on_mesh(
                build.rng.integers(
                    config.vocabulary, size=config.tokens, dtype=np.int32
                ),
                P("model") if tokens else P(),
            ),
        )

    def _operation(
        self, x: jax.Array, weight: jax.Array, targets: jax.Array
    ) -> jax.Array:
        logits = self.constrain_replication_if_enabled(x @ weight)
        shifted = logits - jax.lax.stop_gradient(
            jnp.max(logits, axis=-1, keepdims=True)
        )
        log_probs = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
        target_log_probs = gather_columns_per_row(log_probs, targets[:, None])
        return -jnp.mean(target_log_probs)
