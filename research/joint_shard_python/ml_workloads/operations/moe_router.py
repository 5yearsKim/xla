"""Global expert-logit computation and stable global top-k routing."""

from dataclasses import dataclass
from typing import ClassVar

import jax
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
class MoERouterConfig:
    tokens: int = 128
    hidden: int = 256
    experts: int = 64
    layout: str = "experts"
    seed: int = 0


@dataclass(frozen=True)
class MoERouterInput(BaseOperationInput):
    x: jax.Array
    weight: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.weight)


class MoERouter(ShardedOperation[MoERouterInput, MoERouterConfig]):
    """Return (global expert IDs, weights); ties favor smaller expert IDs."""

    layouts: ClassVar[tuple[str, ...]] = ("experts", "tokens", "replicated")

    def __init__(
        self,
        *,
        top_k: int = 2,
        weight_normalization: str = "selected",
        mesh: Mesh | None = None,
    ) -> None:
        validate_positive_dimensions(top_k=top_k)
        if weight_normalization not in ("selected", "full"):
            raise ValueError("weight_normalization must be 'selected' or 'full'")
        self.top_k = top_k
        self.weight_normalization = weight_normalization
        super().__init__("moe_router", mesh=mesh)

    def create_inputs(self, config: MoERouterConfig) -> MoERouterInput:
        validate_positive_dimensions(
            tokens=config.tokens, hidden=config.hidden, experts=config.experts
        )
        if self.top_k > config.experts:
            raise ValueError("top_k must not exceed experts")
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        return MoERouterInput(
            x=build.random_normal(
                (config.tokens, config.hidden),
                P("model", None) if config.layout == "tokens" else P(),
            ),
            weight=build.random_normal(
                (config.hidden, config.experts),
                P(None, "model") if config.layout == "experts" else P(),
                scale=config.hidden**-0.5,
            ),
        )

    def _operation(self, x: jax.Array, weight: jax.Array) -> tuple[jax.Array, ...]:
        logits = x @ weight
        scores, expert_ids = jax.lax.top_k(logits, self.top_k, is_stable=True)
        if self.weight_normalization == "selected":
            weights = jax.nn.softmax(scores, axis=-1)
        else:
            probabilities = jax.nn.softmax(logits, axis=-1)
            weights = gather_columns_per_row(probabilities, expert_ids)
        return (expert_ids, weights)
