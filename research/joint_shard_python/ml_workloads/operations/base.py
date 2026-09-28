"""Shared interfaces for workloads that produce lowered JAX IR."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
from jax.sharding import Sharding


@dataclass(frozen=True)
class BaseOperationInput(ABC):
    """Input arrays for an operation, in the order its function accepts them."""

    @abstractmethod
    def arrays(self) -> tuple[jax.Array, ...]:
        """Return JAX arrays, including their device shardings."""


InputT = TypeVar("InputT", bound=BaseOperationInput)
ConfigT = TypeVar("ConfigT")


class BaseOperation(ABC, Generic[InputT, ConfigT]):
    """Compile a workload using the shardings attached to its inputs."""

    def __init__(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError("Operation name must be a valid identifier")
        self.name = name

    @abstractmethod
    def create_inputs(self, config: ConfigT) -> InputT:
        """Build workload inputs from operation-specific configuration."""

    @abstractmethod
    def _operation(self, *arrays: jax.Array) -> jax.Array:
        """Compute the workload from its input arrays."""

    def compile(
        self,
        inputs: InputT,
        *,
        out_sharding: Sharding | None = None,
    ) -> jax.stages.Compiled:
        """Lower and compile, triggering any configured XLA dumps."""
        return jax.jit(self._operation, out_shardings=out_sharding).lower(
            *inputs.arrays()
        ).compile()
