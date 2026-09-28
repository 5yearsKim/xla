"""Numerical execution of unpartitioned rewrite candidates on one CPU."""

import re

import jax
import numpy as np
from jax._src import compiler, xla_bridge
from jax._src.interpreters import mlir
from jaxlib.mlir import ir


def execute_unsharded(text: str, arrays: tuple[jax.Array, ...]) -> list[jax.Array]:
    """Remove placement only; compile the full arithmetic on a single CPU.

    Use pre-Shardy candidate IR. This does not execute the explicit collectives
    or measure the multi-device selected program's latency.
    """
    with mlir.make_ir_context():
        module = ir.Module.parse(text)
        for function in module.body.operations:
            if function.operation.name != "func.func":
                continue
            for block in function.regions[0].blocks:
                for op in list(block.operations):
                    if op.operation.name == "sdy.sharding_constraint":
                        op.results[0].replace_all_uses_with(op.operands[0])
                        op.operation.erase()
        text = str(module)
    text = re.sub(
        r"mhlo.num_partitions = \d+ : i32", "mhlo.num_partitions = 1 : i32", text
    )
    text = re.sub(r"^\s*sdy.mesh .*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\{sdy.sharding = #sdy.sharding<@mesh, \[.*?\]>\}", "", text)
    backend = xla_bridge.get_backend("cpu")
    device = backend.devices()[0]
    with mlir.make_ir_context():
        module = ir.Module.parse(text)
        options = compiler.get_compile_options(
            num_replicas=1, num_partitions=1, backend=backend
        )
        executable = backend.compile_and_load(str(module), [device], options)
    return executable.execute(
        [jax.device_put(np.asarray(array), device) for array in arrays]
    )
