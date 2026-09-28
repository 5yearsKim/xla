"""Compile exported selected MLIR directly with prebuilt XLA/PJRT.

The low-level bindings and JAX array helpers are pinned in pyproject.toml.
No tracing, propagation, or search happens here.
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jaxlib
import numpy as np
from jax._src.interpreters import mlir
from jaxlib import xla_client as xc
from jaxlib.mlir import ir


@dataclass(frozen=True)
class Port:
    shape: tuple[int, ...]
    dtype: np.dtype
    sharding: str


def _port(type_, attrs):
    tensor = ir.RankedTensorType(type_)
    # Initial research fixtures use f32. Fail explicitly for other types rather
    # than silently changing their input values or precision.
    if str(tensor.element_type) != "f32":
        raise ValueError("the prebuilt runner currently supports f32 ports")
    if any(size < 0 for size in tensor.shape):
        raise ValueError("ports must have static shapes")
    if "mhlo.sharding" not in attrs:
        raise ValueError("expected xla_input.mlir with mhlo.sharding on every port")
    return Port(
        tuple(tensor.shape),
        np.dtype("float32"),
        ir.StringAttr(attrs["mhlo.sharding"]).value,
    )


class SelectedExecutable:
    """Global NumPy inputs/outputs, with JAX handling device shard transfers."""

    def __init__(self, exported_mlir: str, *, platform="cpu", dump_dir=None):
        if jax.__version__ != "0.11.2" or jaxlib.__version__ != "0.11.2":
            raise RuntimeError(
                "run 'uv sync' in joint_shard_executor (requires JAX/jaxlib 0.11.2)"
            )
        with mlir.make_ir_context():
            module = ir.Module.parse(exported_mlir)
            attributes = module.operation.attributes
            missing = [
                name
                for name in ("mhlo.num_partitions", "mhlo.num_replicas")
                if name not in attributes
            ]
            if missing:
                raise ValueError(
                    "expected exported xla_input.mlir; missing "
                    + ", ".join(missing)
                    + ". Generate it using the optimizer's --output-dir option; "
                    "selected.mlir still contains unexported Shardy IR."
                )
            self.partitions = ir.IntegerAttr(attributes["mhlo.num_partitions"]).value
            replicas = ir.IntegerAttr(attributes["mhlo.num_replicas"]).value
            if replicas != 1 or self.partitions < 1:
                raise ValueError("expected one replica and positive partitions")
            main = next(
                op
                for op in module.body.operations
                if op.operation.name == "func.func"
                and ir.StringAttr(op.attributes["sym_name"]).value == "main"
            )
            self.inputs = tuple(
                _port(t, a)
                for t, a in zip(
                    main.type.inputs,
                    ir.ArrayAttr(main.attributes["arg_attrs"])
                    if main.type.inputs
                    else (),
                )
            )
            self.outputs = tuple(
                _port(t, a)
                for t, a in zip(
                    main.type.results,
                    ir.ArrayAttr(main.attributes["res_attrs"])
                    if main.type.results
                    else (),
                )
            )
        if platform == "cpu":
            self.client = xc.make_cpu_client(num_devices=self.partitions)
        elif platform == "gpu":
            from jax._src import xla_bridge

            self.client = xla_bridge.get_backend("gpu")
        else:
            raise ValueError("platform must be cpu or gpu")
        if self.client.device_count() != self.client.local_device_count():
            raise ValueError("only single-host execution is supported")
        self.devices = tuple(self.client.local_devices()[: self.partitions])
        if len(self.devices) != self.partitions:
            raise ValueError(f"need {self.partitions} addressable {platform} devices")
        options = xc.CompileOptions()
        options.num_replicas = 1
        options.num_partitions = self.partitions
        build = options.executable_build_options
        build.num_replicas = 1
        build.num_partitions = self.partitions
        build.use_spmd_partitioning = True
        # The upstream exporter has already lowered the manual body to native
        # StableHLO. Standard GSPMD preserves it instead of repartitioning it.
        build.use_shardy_partitioner = False
        options.device_assignment = xc.DeviceAssignment.create(
            np.array([[d.id for d in self.devices]], dtype=np.int32)
        )
        if dump_dir:
            build.debug_options.xla_dump_to = str(Path(dump_dir).resolve() / "xla")
            build.debug_options.xla_dump_hlo_as_text = True
        self.executable = self.client.compile_and_load(
            exported_mlir, self.devices, options
        )

    def random_inputs(self, seed=42):
        rng = np.random.default_rng(seed)
        return [
            np.asarray(rng.uniform(-0.5, 0.5, port.shape), dtype=port.dtype)
            for port in self.inputs
        ]

    def _sharding(self, port):
        return xc.GSPMDSharding(self.devices, xc.HloSharding.from_string(port.sharding))

    def prepare_inputs(self, inputs: list[np.ndarray]) -> list[jax.Array]:
        """Upload global inputs using the exact exported boundary shardings."""
        if len(inputs) != len(self.inputs):
            raise ValueError(f"expected {len(self.inputs)} arguments")
        arguments = []
        for index, (value, port) in enumerate(zip(inputs, self.inputs)):
            value = np.asarray(value)
            if value.shape != port.shape or value.dtype != port.dtype:
                raise ValueError(
                    f"arg{index} must have shape {port.shape} and dtype {port.dtype}"
                )
            sharding = self._sharding(port)
            indices = sharding.addressable_devices_indices_map(port.shape)
            arrays = [jax.device_put(value[indices[d]], d) for d in self.devices]
            arguments.append(
                jax.make_array_from_single_device_arrays(port.shape, sharding, arrays)
            )
        return arguments

    def execute_device(self, arguments: list[jax.Array]) -> list[list[jax.Array]]:
        """Execute with prepared inputs and return device-resident output shards."""
        return self.executable.execute_sharded(
            arguments
        ).disassemble_into_single_device_arrays()

    def materialize_outputs(self, shards: list[list[jax.Array]]) -> list[np.ndarray]:
        return [
            np.asarray(
                jax.make_array_from_single_device_arrays(
                    port.shape, self._sharding(port), arrays
                )
            )
            for port, arrays in zip(self.outputs, shards)
        ]

    def execute(self, inputs, *, repeats=1):
        if repeats < 1:
            raise ValueError("repeats must be positive")
        arguments = self.prepare_inputs(inputs)
        for _ in range(repeats):
            shards = self.execute_device(arguments)
            # Materialization also waits for completion before another repeat.
            outputs = self.materialize_outputs(shards)
        return outputs
