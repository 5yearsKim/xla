"""Input construction shared by original-computation workloads."""

from typing import ClassVar

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .base import BaseOperation, ConfigT, InputT


def validate_positive_dimensions(**dimensions: int) -> None:
    """Reject invalid dimensions before allocation or tracing."""
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def validate_positive_epsilon(epsilon: float) -> None:
    """Reject a nonfinite or nonpositive normalization epsilon."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")


def gather_columns_per_row(array: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather the specified valid columns separately for each matrix row."""
    return jax.lax.gather(
        array,
        indices[..., None],
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        ),
        slice_sizes=(1, 1),
        mode="promise_in_bounds",
    )


class ShardedInputBuilder:
    """Seeded host initialization with validated device placement."""

    def __init__(
        self, mesh: Mesh, seed: int, layout: str, layouts: tuple[str, ...]
    ) -> None:
        if layout not in layouts:
            raise ValueError(f"layout must be one of {layouts}, got {layout!r}")
        self.mesh = mesh
        self.rng = np.random.default_rng(seed)

    def place_on_mesh(self, array: np.ndarray, spec: P = P()) -> jax.Array:
        """Validate partition divisibility and place an array on the mesh."""
        for dimension, axis in enumerate(spec):
            if axis is not None:
                partitions = self.mesh.shape[axis]
                if array.shape[dimension] % partitions:
                    raise ValueError(
                        f"Sharded dimension {dimension} of shape {array.shape} "
                        f"must be divisible by mesh size {partitions}"
                    )
        return jax.device_put(array, NamedSharding(self.mesh, spec))

    def random_normal(
        self, shape: tuple[int, ...], spec: P = P(), *, scale: float = 1.0
    ) -> jax.Array:
        """Create seeded float32 normal samples and place them on the mesh."""
        array = self.rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
        return self.place_on_mesh(array, spec)


class ShardedOperation(BaseOperation[InputT, ConfigT]):
    """Use input layouts while leaving intermediate and output layouts inferred."""

    layouts: ClassVar[tuple[str, ...]] = ("replicated",)
    supports_full_tensor_boundary: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        *,
        mesh: Mesh | None = None,
        full_tensor_boundary: bool = False,
    ) -> None:
        super().__init__(name)
        self.mesh = (
            mesh
            if mesh is not None
            else Mesh(np.asarray(jax.devices()), axis_names=("model",))
        )
        if self.mesh.axis_names != ("model",):
            raise ValueError("Workloads require a one-dimensional mesh named 'model'")
        self.full_tensor_boundary = full_tensor_boundary

    def constrain_replication_if_enabled(self, array: jax.Array) -> jax.Array:
        """Apply a replication constraint when full_tensor_boundary is enabled."""
        if self.full_tensor_boundary:
            return jax.lax.with_sharding_constraint(
                array, NamedSharding(self.mesh, P())
            )
        return array
