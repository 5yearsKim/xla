"""Shared interfaces for workloads that produce lowered JAX IR."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
from jax.sharding import Sharding

OperationOutput = jax.Array | tuple[jax.Array, ...]
OutputSharding = Sharding | tuple[Sharding | None, ...] | None


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
        self.output_sharding: OutputSharding = None

    @abstractmethod
    def create_inputs(self, config: ConfigT) -> InputT:
        """Build workload inputs from operation-specific configuration."""

    @abstractmethod
    def _operation(self, *arrays: jax.Array) -> OperationOutput:
        """Compute the workload from its input arrays."""

    def lower(
        self,
        inputs: InputT,
        *,
        out_sharding: OutputSharding = None,
    ) -> jax.stages.Lowered:
        """Lower the original computation before backend optimization."""
        return jax.jit(self._operation, out_shardings=out_sharding).lower(
            *inputs.arrays()
        )

    def compile(
        self,
        inputs: InputT,
        *,
        out_sharding: OutputSharding = None,
    ) -> jax.stages.Compiled:
        """Lower and compile, triggering any configured XLA dumps."""
        return self.lower(inputs, out_sharding=out_sharding).compile()

    def execute(self, inputs: InputT) -> OperationOutput:
        """Compile and execute with the operation's output contract."""
        return self.compile(inputs, out_sharding=self.output_sharding)(*inputs.arrays())
