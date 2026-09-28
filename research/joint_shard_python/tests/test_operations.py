"""Numerical, sharding, and original-computation checks on four CPU devices."""

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

# Device count must be set before importing JAX. Preserve caller flags.
test_flags = os.environ.get("XLA_FLAGS", "").split()
if not any(
    flag.startswith("--xla_force_host_platform_device_count=") for flag in test_flags
):
    test_flags.append("--xla_force_host_platform_device_count=4")
os.environ["XLA_FLAGS"] = " ".join(test_flags)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ml_workloads import (  # noqa: E402
    OPERATIONS,
    BarlowTwins,
    BarlowTwinsConfig,
    ContextAttention,
    ContextAttentionConfig,
    LayerNormLinear,
    LayerNormLinearConfig,
    MoERouter,
    MoERouterConfig,
    OutputProjectionCE,
    OutputProjectionCEConfig,
)

SMALL_DIMENSIONS = {
    "batch": 2,
    "tokens": 4,
    "sequence": 8,
    "hidden": 8,
    "output": 4,
    "vocabulary": 12,
    "queries": 12,
    "head_dim": 4,
    "value_dim": 4,
    "rank": 4,
    "projection": 4,
    "ffn": 12,
    "feature_rank": 4,
    "graphs": 4,
    "nodes": 8,
    "samples": 4,
    "features": 8,
    "experts": 8,
}


def small_config(config_type: type, **overrides: Any) -> Any:
    values = {
        field.name: SMALL_DIMENSIONS[field.name]
        for field in fields(config_type)
        if field.name in SMALL_DIMENSIONS
    }
    return config_type(**(values | overrides))


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def reference(name: str, arrays: list[np.ndarray], **options: Any) -> Any:
    """Independent float64 references; use explicit off-diagonal masking for BT."""
    match name:
        case "output_projection_ce":
            x, weight, targets = arrays
            logits = x @ weight
            shifted = logits - logits.max(axis=-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            return -log_probs[np.arange(x.shape[0]), targets].mean()
        case "context_attention":
            q, k, v = arrays
            return softmax((q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])) @ v
        case "layernorm_linear":
            x, scale, bias, weight = arrays
            centered = x - x.mean(axis=-1, keepdims=True)
            normalized = centered / np.sqrt(
                (centered**2).mean(axis=-1, keepdims=True)
                + options.get("epsilon", 1e-5)
            )
            return (normalized * scale + bias) @ weight
        case "reduce_dot":
            x, weight = arrays
            return x.sum(axis=1) @ weight
        case "lora":
            x, weight, a, b = arrays
            return x @ (weight + a @ b)
        case "qkv_projection":
            x, wq, wk, wv = arrays
            return (x @ wq, x @ wk, x @ wv)
        case "gated_mlp":
            x, gate, up, down = arrays
            gate_values = x @ gate
            return ((gate_values / (1 + np.exp(-gate_values))) * (x @ up)) @ down
        case "kernel_linear_attention":
            q, k, v = arrays
            kernel = (np.maximum(q, 0) + 1) @ (np.maximum(k, 0) + 1).swapaxes(-1, -2)
            return (kernel @ v) / kernel.sum(axis=-1, keepdims=True)
        case "graph_convolution":
            adjacency, x, weight = arrays
            return np.maximum((adjacency @ x) @ weight, 0)
        case "barlow_twins":
            a, b = arrays

            def normalize(x: np.ndarray) -> np.ndarray:
                centered = x - x.mean(axis=0, keepdims=True)
                return centered / np.sqrt(
                    (centered**2).mean(axis=0, keepdims=True)
                    + options.get("epsilon", 1e-5)
                )

            correlation = normalize(a).T @ normalize(b) / a.shape[0]
            off_diagonal = correlation.copy()
            np.fill_diagonal(off_diagonal, 0)
            return np.sum((np.diag(correlation) - 1) ** 2) + options.get(
                "off_diagonal_weight", 0.005
            ) * np.sum(off_diagonal**2)
        case "moe_router":
            x, weight = arrays
            logits = x @ weight
            ids = np.argsort(-logits, axis=-1, kind="stable")[
                :, : options.get("top_k", 2)
            ]
            if options.get("weight_normalization", "selected") == "selected":
                weights = softmax(np.take_along_axis(logits, ids, axis=-1))
            else:
                weights = np.take_along_axis(softmax(logits), ids, axis=-1)
            return (ids.astype(np.int32), weights)
        case _:
            raise AssertionError(f"No reference for {name}")


def host_arrays(arrays: tuple[jax.Array, ...]) -> list[np.ndarray]:
    return [
        np.asarray(
            array,
            dtype=np.float64
            if np.issubdtype(array.dtype, np.floating)
            else array.dtype,
        )
        for array in arrays
    ]


def float_leaves(output: Any) -> list[Any]:
    return [
        leaf
        for leaf in jax.tree.leaves(output)
        if np.issubdtype(leaf.dtype, np.floating)
    ]


class OperationsTest(unittest.TestCase):
    def assert_outputs_close(self, actual: Any, expected: Any) -> None:
        actual_leaves, expected_leaves = (
            jax.tree.leaves(actual),
            jax.tree.leaves(expected),
        )
        self.assertEqual(len(actual_leaves), len(expected_leaves))
        for actual_leaf, expected_leaf in zip(
            actual_leaves, expected_leaves, strict=True
        ):
            if np.issubdtype(actual_leaf.dtype, np.integer):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
            else:
                np.testing.assert_allclose(
                    actual_leaf, expected_leaf, rtol=3e-5, atol=3e-5
                )

    def test_all_layouts_match_independent_references(self) -> None:
        for name, (operation_type, config_type) in OPERATIONS.items():
            if name == "megatron_layer":
                continue
            operation = operation_type()
            reference_arrays = None
            for layout in operation.layouts:
                with self.subTest(operation=name, layout=layout):
                    inputs = operation.create_inputs(
                        small_config(config_type, layout=layout)
                    )
                    arrays = host_arrays(inputs.arrays())
                    if reference_arrays is None:
                        reference_arrays = arrays
                    else:
                        for actual, expected in zip(
                            arrays, reference_arrays, strict=True
                        ):
                            np.testing.assert_array_equal(actual, expected)
                    output = operation.execute(inputs)
                    self.assert_outputs_close(output, reference(name, arrays))
                    for array in inputs.arrays():
                        self.assertEqual(
                            array.sharding.mesh.shape["model"], len(jax.devices())
                        )
                    if layout != "replicated":
                        self.assertTrue(
                            any(
                                not array.is_fully_replicated
                                for array in inputs.arrays()
                            )
                        )

    def test_gradients_against_numpy_directional_differences(self) -> None:
        rng = np.random.default_rng(7)
        for name, (operation_type, config_type) in OPERATIONS.items():
            if name == "megatron_layer":
                continue
            with self.subTest(operation=name):
                operation = operation_type()
                inputs = operation.create_inputs(small_config(config_type))
                arrays = inputs.arrays()
                host = host_arrays(arrays)
                float_indices = tuple(
                    i
                    for i, array in enumerate(arrays)
                    if np.issubdtype(array.dtype, np.floating)
                )
                cotangents = [
                    rng.normal(size=leaf.shape)
                    for leaf in float_leaves(reference(name, host))
                ]

                def objective(*args: jax.Array) -> jax.Array:
                    output = float_leaves(operation._operation(*args))
                    return sum(
                        jnp.sum(value * jnp.asarray(cotangent, dtype=value.dtype))
                        for value, cotangent in zip(output, cotangents, strict=True)
                    )

                gradients = jax.jit(jax.grad(objective, argnums=float_indices))(*arrays)
                directions = {
                    i: rng.normal(scale=0.02, size=host[i].shape) for i in float_indices
                }
                derivative = sum(
                    np.sum(np.asarray(grad) * directions[i])
                    for i, grad in zip(float_indices, gradients, strict=True)
                )
                step = 1e-4

                def perturbed_objective(sign: int) -> float:
                    perturbed = [
                        array + sign * step * directions[i]
                        if i in directions
                        else array
                        for i, array in enumerate(host)
                    ]
                    leaves = float_leaves(reference(name, perturbed))
                    return float(
                        sum(
                            np.sum(value * cotangent)
                            for value, cotangent in zip(leaves, cotangents, strict=True)
                        )
                    )

                finite_difference = (
                    perturbed_objective(1) - perturbed_objective(-1)
                ) / (2 * step)
                self.assertTrue(np.isfinite(derivative))
                np.testing.assert_allclose(
                    derivative, finite_difference, rtol=3e-3, atol=5e-5
                )

    def test_original_dot_order_and_intermediate_shapes(self) -> None:
        expected = {
            "output_projection_ce": [(4, 12)],
            "context_attention": [(2, 12, 8), (2, 12, 4)],
            "layernorm_linear": [(4, 4)],
            "reduce_dot": [(2, 4)],
            "lora": [(8, 4), (4, 4)],
            "qkv_projection": [(4, 4)] * 3,
            "gated_mlp": [(4, 12), (4, 12), (4, 4)],
            "kernel_linear_attention": [(2, 12, 8), (2, 12, 4)],
            "graph_convolution": [(4, 8, 8), (4, 8, 4)],
            "barlow_twins": [(8, 8)],
            "moe_router": [(4, 8)],
        }
        for name, shapes in expected.items():
            with self.subTest(operation=name):
                operation_type, config_type = OPERATIONS[name]
                operation = operation_type()
                inputs = operation.create_inputs(small_config(config_type))
                jaxpr = jax.make_jaxpr(operation._operation)(*inputs.arrays()).jaxpr
                dots = [
                    eqn for eqn in jaxpr.eqns if eqn.primitive.name == "dot_general"
                ]
                self.assertEqual([dot.outvars[0].aval.shape for dot in dots], shapes)
                original = operation.lower(inputs).as_text("stablehlo")
                self.assertEqual(original.count("stablehlo.dot_general"), len(shapes))
                self.assertNotIn("sdy.sharding_constraint", original)
                self.assertNotIn("call @", original)
                if name == "reduce_dot":
                    self.assertEqual(dots[0].invars[0].aval.shape, (2, 8))

    def test_explicit_replication_is_opt_in(self) -> None:
        for operation_type, config_type in (
            (OutputProjectionCE, OutputProjectionCEConfig),
            (ContextAttention, ContextAttentionConfig),
            (LayerNormLinear, LayerNormLinearConfig),
        ):
            operation = operation_type(full_tensor_boundary=True)
            inputs = operation.create_inputs(small_config(config_type))
            with self.subTest(operation=operation.name):
                self.assertIn(
                    "sdy.sharding_constraint",
                    operation.lower(inputs).as_text("stablehlo"),
                )
                self.assert_outputs_close(
                    operation.execute(inputs),
                    reference(operation.name, host_arrays(inputs.arrays())),
                )

    def test_stable_cross_entropy_at_extreme_logits(self) -> None:
        operation = OutputProjectionCE()
        inputs = operation.create_inputs(small_config(OutputProjectionCEConfig))
        inputs = replace(inputs, weight=inputs.weight * 1000)
        output = operation.execute(inputs)
        self.assertTrue(np.isfinite(output))
        self.assert_outputs_close(
            output, reference(operation.name, host_arrays(inputs.arrays()))
        )

    def test_layernorm_large_offset_small_variance(self) -> None:
        operation = LayerNormLinear(epsilon=1e-4)
        inputs = operation.create_inputs(small_config(LayerNormLinearConfig))
        offsets = np.asarray(
            [-0.5, -0.375, -0.25, -0.125, 0.125, 0.25, 0.375, 0.5], dtype=np.float32
        )
        x = np.broadcast_to(10000 + offsets, (4, 8)).copy()
        inputs = replace(inputs, x=jax.device_put(x, inputs.x.sharding))
        self.assert_outputs_close(
            operation.execute(inputs),
            reference(
                operation.name, host_arrays(inputs.arrays()), epsilon=operation.epsilon
            ),
        )

    def test_barlow_constant_features_and_custom_penalty(self) -> None:
        operation = BarlowTwins(epsilon=1e-3, off_diagonal_weight=0.1)
        inputs = operation.create_inputs(small_config(BarlowTwinsConfig))
        self.assert_outputs_close(
            operation.execute(inputs),
            reference(
                operation.name,
                host_arrays(inputs.arrays()),
                epsilon=operation.epsilon,
                off_diagonal_weight=operation.off_diagonal_weight,
            ),
        )
        constant = replace(inputs, a=jnp.ones_like(inputs.a), b=jnp.ones_like(inputs.b))
        self.assertEqual(float(operation.execute(constant)), 8.0)

    def test_moe_global_ids_ties_and_normalization(self) -> None:
        for mode in ("selected", "full"):
            for k in (1, 3, 8):
                operation = MoERouter(top_k=k, weight_normalization=mode)
                inputs = operation.create_inputs(small_config(MoERouterConfig))
                with self.subTest(mode=mode, top_k=k):
                    output = operation.execute(inputs)
                    self.assert_outputs_close(
                        output,
                        reference(
                            operation.name,
                            host_arrays(inputs.arrays()),
                            top_k=k,
                            weight_normalization=mode,
                        ),
                    )
                    tied = replace(inputs, weight=jnp.zeros_like(inputs.weight))
                    ids, weights = operation.execute(tied)
                    np.testing.assert_array_equal(
                        ids, np.broadcast_to(np.arange(k), (4, k))
                    )
                    np.testing.assert_allclose(
                        weights, 1 / (k if mode == "selected" else 8)
                    )

    def test_validation(self) -> None:
        for name, (operation_type, config_type) in OPERATIONS.items():
            if name == "megatron_layer":
                continue
            operation = operation_type()
            config = small_config(config_type)
            with self.subTest(operation=name):
                with self.assertRaisesRegex(ValueError, "layout"):
                    operation.create_inputs(replace(config, layout="unknown"))
                dimension = next(
                    field.name
                    for field in fields(config)
                    if field.name in SMALL_DIMENSIONS
                )
                with self.assertRaisesRegex(ValueError, "positive"):
                    operation.create_inputs(replace(config, **{dimension: 0}))
        with self.assertRaisesRegex(ValueError, "divisible"):
            OutputProjectionCE().create_inputs(OutputProjectionCEConfig(vocabulary=7))
        for epsilon in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                LayerNormLinear(epsilon=epsilon)
            with self.assertRaises(ValueError):
                BarlowTwins(epsilon=epsilon)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            BarlowTwins(off_diagonal_weight=-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            MoERouter(top_k=0)
        with self.assertRaisesRegex(ValueError, "exceed"):
            MoERouter(top_k=9).create_inputs(small_config(MoERouterConfig))
        with self.assertRaisesRegex(ValueError, "weight_normalization"):
            MoERouter(weight_normalization="unknown")
        operation_type, config_type = OPERATIONS["graph_convolution"]
        with self.assertRaisesRegex(ValueError, "edge_probability"):
            operation_type().create_inputs(
                small_config(config_type, edge_probability=2.0)
            )

    def test_cli_exports_originals_and_handles_tuple_outputs(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as dump_dir:
            for name in OPERATIONS:
                _, config_type = OPERATIONS[name]
                config = small_config(config_type)
                arguments = [
                    sys.executable,
                    str(project / "main.py"),
                    "--operation",
                    name,
                    "--output-dir",
                    dump_dir,
                    "--execute",
                ]
                for field in fields(config):
                    arguments.extend(
                        [
                            "--" + field.name.replace("_", "-"),
                            str(getattr(config, field.name)),
                        ]
                    )
                result = subprocess.run(
                    arguments,
                    cwd=project,
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("Output", result.stdout)
                original = Path(dump_dir, name, "original.mlir")
                self.assertTrue(original.is_file())
                self.assertIn("stablehlo.dot_general", original.read_text())


if __name__ == "__main__":
    unittest.main()
