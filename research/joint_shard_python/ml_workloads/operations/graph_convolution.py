"""Dense graph aggregation before feature projection and activation."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ._common import ShardedInputBuilder, ShardedOperation, validate_positive_dimensions
from .base import BaseOperationInput


@dataclass(frozen=True)
class GraphConvolutionConfig:
    graphs: int = 4
    nodes: int = 128
    hidden: int = 256
    output: int = 64
    edge_probability: float = 0.05
    layout: str = "nodes"
    seed: int = 0


@dataclass(frozen=True)
class GraphConvolutionInput(BaseOperationInput):
    adjacency: jax.Array
    x: jax.Array
    weight: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.adjacency, self.x, self.weight)


class GraphConvolution(ShardedOperation[GraphConvolutionInput, GraphConvolutionConfig]):
    """A is a fixed row-normalized dense adjacency with self edges."""

    layouts: ClassVar[tuple[str, ...]] = ("nodes", "independent_graphs", "replicated")

    def __init__(self, *, mesh: Mesh | None = None) -> None:
        super().__init__("graph_convolution", mesh=mesh)

    def create_inputs(self, config: GraphConvolutionConfig) -> GraphConvolutionInput:
        validate_positive_dimensions(
            graphs=config.graphs,
            nodes=config.nodes,
            hidden=config.hidden,
            output=config.output,
        )
        if (
            not np.isfinite(config.edge_probability)
            or not 0 <= config.edge_probability <= 1
        ):
            raise ValueError("edge_probability must be finite and between zero and one")
        build = ShardedInputBuilder(self.mesh, config.seed, config.layout, self.layouts)
        spec = {
            "nodes": P(None, "model", None),
            "independent_graphs": P("model", None, None),
            "replicated": P(),
        }[config.layout]
        adjacency = (
            build.rng.random((config.graphs, config.nodes, config.nodes))
            < config.edge_probability
        ).astype(np.float32)
        indices = np.arange(config.nodes)
        adjacency[:, indices, indices] = 1
        adjacency /= adjacency.sum(axis=-1, keepdims=True)
        return GraphConvolutionInput(
            adjacency=build.place_on_mesh(adjacency, spec),
            x=build.random_normal((config.graphs, config.nodes, config.hidden), spec),
            weight=build.random_normal(
                (config.hidden, config.output), scale=config.hidden**-0.5
            ),
        )

    def _operation(
        self, adjacency: jax.Array, x: jax.Array, weight: jax.Array
    ) -> jax.Array:
        return jnp.maximum((adjacency @ x) @ weight, 0)
