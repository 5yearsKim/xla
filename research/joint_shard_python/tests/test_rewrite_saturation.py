"""Lower real originals, rewrite through the C++ engine, and execute on CPU.

Build //research/joint_shard/tools:parse_stablehlo before running these tests.
JOINT_SHARD_REWRITE_BINARY overrides the binary; JOINT_SHARD_REWRITE_REPORT_DIR
optionally retains original/replacement MLIR and a JSON saturation summary.
"""

import json
import os
import re
import subprocess
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import mlir
from jaxlib.mlir import ir

# Import this first to configure the four CPU devices before importing JAX.
from test_operations import host_arrays, reference, small_config

from ml_workloads import OPERATIONS
from ml_workloads.rewrite_execution import execute_unsharded

PROJECT = Path(__file__).resolve().parents[1]
BINARY = Path(
    os.environ.get(
        "JOINT_SHARD_REWRITE_BINARY",
        str(
            PROJECT.parent.parent
            / "bazel-bin/research/joint_shard/tools/parse_stablehlo"
        ),
    )
)
if "JOINT_SHARD_REWRITE_BINARY" in os.environ and not BINARY.is_file():
    raise RuntimeError(f"Rewrite binary does not exist: {BINARY}")


def execute_mlir(text: str, arrays: tuple[jax.Array, ...]) -> list[jax.Array]:
    """Check values with full inputs on one CPU; this is not a sharding benchmark.

    The engine verifies the module with the original mesh/annotations. For this
    numerical oracle only, remove argument placement annotations and the mesh.
    No computation or arithmetic operation is substituted.
    """
    return execute_unsharded(text, arrays)


def dot_shapes(text: str) -> list[tuple[int, ...]]:
    with mlir.make_ir_context():
        module = ir.Module.parse(text)
        return [
            tuple(ir.RankedTensorType(op.results[0].type).shape)
            for function in module.body.operations
            if function.operation.name == "func.func"
            for block in function.regions[0].blocks
            for op in block.operations
            if op.operation.name == "stablehlo.dot_general"
        ]


@unittest.skipUnless(BINARY.is_file(), "Build the C++ parse_stablehlo tool first")
class RewriteSaturationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.artifacts = Path(
            os.environ.get("JOINT_SHARD_REWRITE_REPORT_DIR", cls.temporary.name)
        )
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        cls.summary: list[dict[str, Any]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        (cls.artifacts / "summary.json").write_text(json.dumps(cls.summary, indent=2))
        cls.temporary.cleanup()

    def rewrite_text(
        self,
        label: str,
        original: str,
        policy: str,
        extra_options: tuple[str, ...] = (),
    ) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]], dict[str, int]]:
        path = self.artifacts / f"{label}.original.mlir"
        path.write_text(original)
        result = subprocess.run(
            [
                str(BINARY),
                str(path),
                f"--numerical-policy={policy}",
                "--rewrite-report",
                *extra_options,
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        (self.artifacts / f"{label}.report.txt").write_text(result.stderr)
        (self.artifacts / f"{label}.rewritten.mlir").write_text(result.stdout)
        self.assertEqual(result.returncode, 0, msg=f"{label}: {result.stderr}")
        runs = [
            {"stop": stop, "iterations": int(iterations), "nodes": int(nodes)}
            for stop, iterations, nodes in re.findall(
                r"run \d+ stop=(\S+) iterations=(\d+) nodes=(\d+)", result.stderr
            )
        ]
        self.assertTrue(runs, msg=label)
        for run in runs:
            self.assertIn(run["stop"], {"saturated", "iteration-limit", "match-limit"})
            self.assertLessEqual(run["iterations"], 10)
            self.assertLess(run["nodes"], 10000)
        applied = {
            rule: int(count)
            for rule, count in re.findall(r"rule (\S+).*?applied=(\d+)", result.stderr)
            if int(count)
        }
        return result, runs, applied

    def rewrite_and_check(
        self,
        name: str,
        config: Any,
        *,
        policy: str = "relaxed",
        variant: str = "forward",
        gradient: bool = False,
    ) -> dict[str, Any]:
        operation_type, _ = OPERATIONS[name]
        operation = operation_type()
        inputs = operation.create_inputs(config)
        arrays = inputs.arrays()
        if gradient:
            indices = tuple(
                i
                for i, array in enumerate(arrays)
                if np.issubdtype(array.dtype, np.floating)
            )

            def objective(*args: jax.Array) -> jax.Array:
                output = operation._operation(*args)
                return sum(
                    jnp.sum(leaf)
                    for leaf in jax.tree.leaves(output)
                    if np.issubdtype(leaf.dtype, np.floating)
                )

            function = jax.grad(objective, argnums=indices)
            original = jax.jit(function).lower(*arrays).as_text("stablehlo")
            expected = jax.jit(function)(*arrays)
        else:
            original = operation.lower(inputs).as_text("stablehlo")
            expected = (
                operation.execute(inputs)
                if name == "megatron_layer"
                else reference(name, host_arrays(arrays))
            )
        layout = getattr(config, "layout", "tensor_parallel")
        label = f"{name}.{variant}.{layout}.{policy}"
        result, runs, applied = self.rewrite_text(label, original, policy)
        actual = execute_mlir(result.stdout, arrays)
        expected_leaves = jax.tree.leaves(expected)
        self.assertEqual(len(actual), len(expected_leaves), msg=label)
        for value, expected_value in zip(actual, expected_leaves, strict=True):
            self.assertEqual(value.shape, expected_value.shape, msg=label)
            if np.issubdtype(value.dtype, np.integer):
                np.testing.assert_array_equal(value, expected_value, err_msg=label)
            else:
                np.testing.assert_allclose(
                    value,
                    expected_value,
                    rtol=3e-4 if gradient else 3e-5,
                    atol=3e-4 if gradient else 3e-5,
                    err_msg=label,
                )
        row = {
            "operation": name,
            "variant": variant,
            "layout": layout,
            "policy": policy,
            "dimensions": {
                field.name: getattr(config, field.name)
                for field in fields(config)
                if isinstance(getattr(config, field.name), int)
            },
            "runs": runs,
            "original_dot_shapes": dot_shapes(original),
            "rewritten_dot_shapes": dot_shapes(result.stdout),
            "applied_rules": applied,
            "boundaries": re.findall(r"^boundary (.*)$", result.stderr, re.MULTILINE),
        }
        self.summary.append(row)
        if policy == "relaxed" and not gradient:
            targets = {
                "reduce_dot": "project-before-sequence-sum",
                "lora": "expand-lora-projection",
                "graph_convolution": "reassociate-batched-shared-weight-right",
                "kernel_linear_attention": "kernel-attention-summaries-transposed-key-nested-broadcast",
                "context_attention": "dot-divide-broadcast-nested",
            }
            if name in targets:
                self.assertIn(targets[name], applied, msg=f"{label}: {result.stderr}")
        if name == "context_attention" and not gradient:
            self.assertFalse(
                any(rule.startswith("kernel-attention-summaries") for rule in applied)
            )
            if policy == "strict":
                self.assertFalse(
                    any(rule.startswith("dot-divide-broadcast") for rule in applied)
                )
        return row

    def test_all_originals_all_layouts_strict_and_relaxed(self) -> None:
        for name, (operation_type, config_type) in OPERATIONS.items():
            for layout in getattr(operation_type, "layouts", (None,)):
                config = small_config(
                    config_type, **({"layout": layout} if layout else {})
                )
                for policy in ("strict", "relaxed"):
                    with self.subTest(operation=name, layout=layout, policy=policy):
                        row = self.rewrite_and_check(name, config, policy=policy)
                        if name in ("reduce_dot", "lora", "graph_convolution"):
                            self.assertTrue(
                                all(run["stop"] == "saturated" for run in row["runs"])
                            )

    def test_dimensions_that_reverse_intermediate_size_tradeoffs(self) -> None:
        variants = {
            "reduce_dot": ({"hidden": 32, "output": 4}, {"hidden": 4, "output": 32}),
            "graph_convolution": (
                {"hidden": 32, "output": 4},
                {"hidden": 4, "output": 32},
            ),
            "lora": (
                {"tokens": 4, "hidden": 32, "output": 16},
                {"tokens": 32, "hidden": 8, "output": 4},
            ),
            "kernel_linear_attention": (
                {"queries": 32, "feature_rank": 4},
                {"queries": 4, "feature_rank": 16},
            ),
        }
        for name, shapes in variants.items():
            _, config_type = OPERATIONS[name]
            for index, dimensions in enumerate(shapes):
                with self.subTest(operation=name, dimensions=dimensions):
                    row = self.rewrite_and_check(
                        name,
                        small_config(config_type, **dimensions),
                        variant=f"shape{index}",
                    )
                    selected = row["rewritten_dot_shapes"]
                    if name == "lora":
                        materialized_weight = (
                            dimensions["hidden"],
                            dimensions["output"],
                        )
                        self.assertEqual(materialized_weight in selected, bool(index))
                    elif name == "graph_convolution":
                        # Tests use small_config's graph/node sizes.
                        config = small_config(config_type, **dimensions)
                        aggregation = (config.graphs, config.nodes, config.hidden)
                        self.assertEqual(aggregation in selected, bool(index))
                    elif name == "kernel_linear_attention":
                        config = small_config(config_type, **dimensions)
                        scores = (config.batch, config.queries, config.sequence)
                        self.assertEqual(scores in selected, bool(index))

    def test_backward_programs(self) -> None:
        for name, (_, config_type) in OPERATIONS.items():
            if name == "megatron_layer":
                continue
            with self.subTest(operation=name):
                self.rewrite_and_check(
                    name, small_config(config_type), gradient=True, variant="backward"
                )

    def test_gram_norm_shape_controls_and_gradients(self) -> None:
        def norm(a: jax.Array, b: jax.Array) -> jax.Array:
            correlation = a.T @ b
            return jnp.sum(correlation * correlation)

        def sample_norm(a: jax.Array, b: jax.Array) -> jax.Array:
            return jnp.sum((a @ a.T) * (b @ b.T))

        rng = np.random.default_rng(31)
        for samples, features_a, features_b in ((4, 16, 16), (16, 4, 4), (4, 8, 12)):
            a = jnp.asarray(rng.normal(size=(samples, features_a)), dtype=jnp.float32)
            b = jnp.asarray(rng.normal(size=(samples, features_b)), dtype=jnp.float32)
            for gradient in (False, True):
                function = jax.grad(norm, argnums=(0, 1)) if gradient else norm
                expected = (
                    jax.grad(sample_norm, argnums=(0, 1))(a, b)
                    if gradient
                    else np.sum(
                        (
                            np.asarray(a, dtype=np.float64).T
                            @ np.asarray(b, dtype=np.float64)
                        )
                        ** 2
                    )
                )
                original = jax.jit(function).lower(a, b).as_text("stablehlo")
                for policy in ("strict", "relaxed"):
                    label = f"gram_norm.{samples}.{features_a}.{features_b}.{gradient}.{policy}"
                    result, runs, applied = self.rewrite_text(label, original, policy)
                    for actual, reference_value in zip(
                        execute_mlir(result.stdout, (a, b)),
                        jax.tree.leaves(expected),
                        strict=True,
                    ):
                        np.testing.assert_allclose(
                            actual, reference_value, rtol=3e-5, atol=3e-5, err_msg=label
                        )
                    if not gradient:
                        gram_applied = any(
                            rule.startswith("feature-gram-to-sample-gram")
                            for rule in applied
                        )
                        self.assertEqual(gram_applied, policy == "relaxed")
                        selected = dot_shapes(result.stdout)
                        use_sample = policy == "relaxed" and samples < min(
                            features_a, features_b
                        )
                        self.assertEqual((samples, samples) in selected, use_sample)
                        self.assertEqual(
                            (features_a, features_b) in selected, not use_sample
                        )
                        self.assertTrue(all(run["stop"] == "saturated" for run in runs))
                    self.summary.append(
                        {
                            "operation": "gram_norm",
                            "variant": "backward" if gradient else "forward",
                            "policy": policy,
                            "dimensions": {
                                "samples": samples,
                                "features_a": features_a,
                                "features_b": features_b,
                            },
                            "runs": runs,
                            "applied_rules": applied,
                            "original_dot_shapes": dot_shapes(original),
                            "rewritten_dot_shapes": dot_shapes(result.stdout),
                        }
                    )

    def test_attention_selects_division_shape_and_permission_controls(self) -> None:
        operation_type, config_type = OPERATIONS["context_attention"]
        for value_dim in (4, 16):
            config = small_config(config_type, value_dim=value_dim)
            operation = operation_type()
            arrays = operation.create_inputs(config).arrays()
            original = jax.jit(operation._operation).lower(*arrays).as_text("stablehlo")
            expected = reference("context_attention", host_arrays(arrays))
            for enabled in (False, True):
                label = f"attention_division.{value_dim}.{enabled}"
                result, runs, applied = self.rewrite_text(
                    label,
                    original,
                    "relaxed",
                    (f"--allow-dot-division={str(enabled).lower()}",),
                )
                np.testing.assert_allclose(
                    execute_mlir(result.stdout, arrays)[0],
                    expected,
                    rtol=3e-5,
                    atol=3e-5,
                )
                self.assertEqual(
                    any(rule.startswith("dot-divide-broadcast") for rule in applied),
                    enabled,
                )
                with mlir.make_ir_context():
                    module = ir.Module.parse(result.stdout)
                    function = next(
                        op
                        for op in module.body.operations
                        if op.operation.name == "func.func"
                    )
                    return_op = list(function.regions[0].blocks[0].operations)[-1]
                    producer = return_op.operands[0].owner
                    self.assertEqual(
                        producer.name,
                        "stablehlo.divide"
                        if enabled and value_dim < config.sequence
                        else "stablehlo.dot_general",
                    )
                self.summary.append(
                    {
                        "operation": "context_attention",
                        "variant": "division_control",
                        "policy": "relaxed",
                        "division_enabled": enabled,
                        "dimensions": {
                            "sequence": config.sequence,
                            "value_dim": value_dim,
                        },
                        "runs": runs,
                        "applied_rules": applied,
                    }
                )


if __name__ == "__main__":
    unittest.main()
