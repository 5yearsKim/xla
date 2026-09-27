"""Numerical tests for the two-artifact optimizer and standalone runner."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prebuilt_runner import SelectedExecutable

WORKSPACE = Path(__file__).resolve().parents[2]
OPTIMIZER = WORKSPACE / "bazel-bin/research/joint_shard/tools/summarize_regions"
FIXTURES = WORKSPACE / "research/joint_shard/testdata"
RUNNER = Path(__file__).resolve().with_name("run.py")


def optimize(fixture, directory):
    command = [
        str(OPTIMIZER),
        str(FIXTURES / fixture),
        "--optimize-dag",
        "--compute-work-per-us=1000",
        "--collective-latency-us=0",
        "--output-dir",
        str(directory),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return (directory / "selected.mlir").read_text(), (
        directory / "xla_input.mlir"
    ).read_text()


class ExecutionTest(unittest.TestCase):
    def compare(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)

    def test_optimizer_selected_chain_and_residual(self):
        for name in ("chain_3.mlir", "residual_block.mlir"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                selected, xla_input = optimize(name, Path(directory))
                self.assertIn("sdy.all_gather", selected)
                with self.assertRaisesRegex(ValueError, "expected exported xla_input"):
                    SelectedExecutable(selected)
                self.assertIn("SPMDFullToShardShape", xla_input)
                self.assertNotIn("sdy.all_gather", xla_input)
                program = SelectedExecutable(xla_input)
                s, x, w, v = program.random_inputs(456)
                a = (s * x) @ w
                expected = np.tanh(a) @ v
                if name == "residual_block.mlir":
                    expected = a + expected
                self.compare(program.execute([s, x, w, v], repeats=2)[0], expected)

    def test_cli_inputs_outputs_and_dumps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _ = optimize("chain_3.mlir", root / "plan")
            rng = np.random.default_rng(789)
            scalar = np.array(0.3, dtype=np.float32)
            a = rng.uniform(-0.5, 0.5, (8, 16)).astype(np.float32)
            b = rng.uniform(-0.5, 0.5, (16, 4)).astype(np.float32)
            w = rng.uniform(-0.5, 0.5, (4, 4)).astype(np.float32)
            np.savez(root / "inputs.npz", arg0=scalar, arg1=a, arg2=b, arg3=w)
            command = [sys.executable, str(RUNNER), str(root / "plan/xla_input.mlir")]
            subprocess.run(
                command + ["--compile-only"], capture_output=True, text=True, check=True
            )
            subprocess.run(
                command
                + [
                    "--inputs",
                    str(root / "inputs.npz"),
                    "--dump-dir",
                    str(root / "dump"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            with np.load(root / "dump/outputs.npz") as outputs:
                self.compare(outputs["result0"], np.tanh((scalar * a) @ b) @ w)
            self.assertTrue(any((root / "dump/xla").glob("*after_optimizations.txt")))


if __name__ == "__main__":
    unittest.main()
