"""Arm isolation, charged adapters, and aggressive LayerNorm regression checks."""

import tempfile
import unittest
from pathlib import Path

from experiments.four_arm import ARMS, TOOLS, layernorm_stress, run_case


@unittest.skipUnless(
    (TOOLS / "workload_ablation").is_file() and (TOOLS / "parse_stablehlo").is_file(),
    "Build workload_ablation and parse_stablehlo first",
)
class FourArmTest(unittest.TestCase):
    def test_arm_isolation_and_selected_computations(self) -> None:
        for name, layout in (
            ("reduce_dot", "sequence"),
            ("graph_convolution", "nodes"),
            ("kernel_linear_attention", "sequence"),
            ("layernorm_linear", "hidden"),
        ):
            with (
                self.subTest(operation=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                row = run_case(name, layout, False, Path(directory), quick=True)
                arms = row["arms"]
                self.assertEqual(arms["baseline"]["candidate_id"], 0)
                self.assertEqual(arms["sharding_only"]["candidate_id"], 0)
                for arm in ("baseline", "rewrite_only"):
                    self.assertEqual(
                        arms[arm]["internal_inputs"], row["external_inputs"]
                    )
                    self.assertEqual(arms[arm]["input_adapters_cost"]["total_us"], 0)
                self.assertFalse(row["layout_truncated"])
                self.assertFalse(row["failures"])
                self.assertTrue(row["backward"]["pass"])
                for arm in ARMS:
                    self.assertTrue(arms[arm]["numerical_pass"])
                    self.assertEqual(arms[arm]["cost"]["unknown"], 0)
                    self.assertGreaterEqual(arms[arm]["modeled_bytes_per_device"], 0)
                    self.assertLessEqual(
                        arms["combined"]["cost"]["total_us"],
                        arms[arm]["cost"]["total_us"] + 1e-9,
                    )
                if name == "layernorm_linear":
                    self.assertGreater(
                        row["rules"]["centered-square-to-raw-moments-keepdims"][
                            "applied"
                        ],
                        0,
                    )
                if name == "reduce_dot":
                    self.assertIn(
                        "witness-project-before-sequence-sum", row["candidate_names"]
                    )
                if name == "kernel_linear_attention":
                    self.assertTrue(
                        any(
                            candidate.startswith("witness-kernel-attention-summaries")
                            for candidate in row["candidate_names"]
                        )
                    )

    def test_explicit_opt_in_exposes_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = layernorm_stress(Path(directory), TOOLS / "parse_stablehlo")
        self.assertFalse(result["False"]["raw_moment_applied"])
        self.assertTrue(result["True"]["raw_moment_applied"])
        self.assertEqual(result["False"]["variance"], result["original_variance"])
        self.assertNotEqual(result["True"]["variance"], result["original_variance"])

    def test_layout_cap_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = run_case(
                "reduce_dot",
                "sequence",
                False,
                Path(directory),
                quick=True,
                max_layouts=1,
            )
        self.assertEqual(row["layout_count"], 1)
        self.assertTrue(row["layout_truncated"])
        self.assertEqual(
            row["arms"]["baseline"]["cost"], row["arms"]["sharding_only"]["cost"]
        )
        self.assertEqual(
            row["arms"]["rewrite_only"]["cost"], row["arms"]["combined"]["cost"]
        )
