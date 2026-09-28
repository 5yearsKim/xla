"""Four-arm cost experiments on original operations, with CPU numerical checks.

Run from joint_shard_python: uv run python -m experiments.four_arm --output-dir PATH
Build //research/joint_shard/tools:{workload_ablation,parse_stablehlo} first.
"""

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Configure before JAX initializes. The experiment uses four CPU devices for
# input annotations and one CPU for unpartitioned arithmetic validation.
flags = os.environ.get("XLA_FLAGS", "").split()
if not any(
    flag.startswith("--xla_force_host_platform_device_count=") for flag in flags
):
    flags.append("--xla_force_host_platform_device_count=4")
os.environ["XLA_FLAGS"] = " ".join(flags)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from ml_workloads import OPERATIONS  # noqa: E402
from ml_workloads.rewrite_execution import execute_unsharded  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT.parent.parent / "bazel-bin/research/joint_shard/tools"
ARMS = ("baseline", "rewrite_only", "sharding_only", "combined")
WORKLOADS = (
    "reduce_dot",
    "graph_convolution",
    "kernel_linear_attention",
    "layernorm_linear",
)


def dimensions(name: str, reverse: bool, quick: bool) -> dict[str, int]:
    if name in ("reduce_dot", "graph_convolution", "layernorm_linear"):
        large, small = (32, 8) if quick else (1024, 128)
        result = {
            "hidden": small if reverse else large,
            "output": large if reverse else small,
        }
        if name == "reduce_dot":
            result |= {"batch": 4, "sequence": 16 if quick else 128}
        elif name == "graph_convolution":
            result |= {"graphs": 4, "nodes": 16 if quick else 128}
        else:
            result |= {"tokens": 8 if quick else 32}
        return result
    large, small = (32, 8) if quick else (256, 16)
    return {
        "batch": 4,
        "sequence": 16 if quick else 128,
        "queries": small if reverse else large,
        "feature_rank": large if reverse else small,
        "value_dim": 8 if quick else 32,
    }


def reference(name: str, arrays: tuple[jax.Array, ...]) -> np.ndarray:
    values = [np.asarray(array, dtype=np.float64) for array in arrays]
    if name == "reduce_dot":
        x, w = values
        return x.sum(axis=1) @ w
    if name == "graph_convolution":
        a, x, w = values
        return np.maximum((a @ x) @ w, 0)
    if name == "kernel_linear_attention":
        q, k, v = values
        scores = (np.maximum(q, 0) + 1) @ (np.maximum(k, 0) + 1).swapaxes(-1, -2)
        return (scores @ v) / scores.sum(axis=-1, keepdims=True)
    x, scale, bias, w = values
    centered = x - x.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt(
        (centered * centered).mean(axis=-1, keepdims=True) + 1e-5
    )
    return (normalized * scale + bias) @ w


def invoke(
    binary: Path, args: list[str], path: Path
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=180
    )
    path.write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{binary.name} failed: {result.stderr}")
    return result


def run_case(
    name: str,
    layout: str,
    reverse: bool,
    directory: Path,
    *,
    quick: bool = False,
    binary: Path = TOOLS / "workload_ablation",
    rewrite_binary: Path = TOOLS / "parse_stablehlo",
    cost_options: tuple[str, ...] = (),
    max_layouts: int = 256,
    aggressive_raw_moments: bool = True,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    operation_type, config_type = OPERATIONS[name]
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]), ("model",))
    if mesh.size != 4:
        raise RuntimeError("four CPU devices are required")
    config = config_type(**dimensions(name, reverse, quick), layout=layout, seed=31)
    operation = operation_type(mesh=mesh)
    inputs = operation.create_inputs(config)
    arrays = inputs.arrays()
    # Explicit output contract, identical across all arms. Dense graph outputs
    # remain node/graph partitioned; token-parallel LN outputs remain partitioned.
    output_spec = P()
    if name == "graph_convolution":
        output_spec = (
            P(None, "model", None)
            if layout == "nodes"
            else P("model", None, None)
            if layout == "independent_graphs"
            else P()
        )
    elif name == "layernorm_linear" and layout == "tokens":
        output_spec = P("model", None)
    elif name == "kernel_linear_attention" and layout == "queries":
        output_spec = P(None, "model", None)
    source = directory / "original.mlir"
    source.write_text(
        operation.lower(inputs, out_sharding=NamedSharding(mesh, output_spec)).as_text(
            "stablehlo"
        )
    )
    aggressive = (
        ["--allow-raw-moments"]
        if name == "layernorm_linear" and aggressive_raw_moments
        else []
    )
    result = invoke(
        binary,
        [
            str(source),
            f"--output-dir={directory}",
            "--numerical-policy=relaxed",
            f"--max-layouts={max_layouts}",
            *aggressive,
            *cost_options,
        ],
        directory / "search.log",
    )
    row = json.loads(result.stdout)
    row |= {
        "operation": name,
        "layout": layout,
        "reverse": reverse,
        "config": asdict(config),
        "raw_moments_enabled": name == "layernorm_linear" and aggressive_raw_moments,
    }
    reordered = tuple(arrays[i] for i in row["argument_order"])
    expected = reference(name, arrays)
    np.savez(
        directory / "inputs.npz",
        **{f"arg{i}": np.asarray(value) for i, value in enumerate(reordered)},
    )
    np.save(directory / "reference.npy", expected)
    for arm in ARMS:
        actual = np.asarray(
            execute_unsharded(
                (directory / f"{arm}.candidate.mlir").read_text(), reordered
            )[0]
        )
        np.testing.assert_allclose(
            actual, expected, rtol=3e-4, atol=3e-4, err_msg=f"{name}/{layout}/{arm}"
        )
        row["arms"][arm]["max_abs_error"] = float(np.max(np.abs(actual - expected)))
        row["arms"][arm]["numerical_pass"] = True
    baseline = row["arms"]["baseline"]["cost"]["total_us"]
    combined = row["arms"]["combined"]["cost"]["total_us"]
    row["combined_beats_both"] = all(
        combined < row["arms"][arm]["cost"]["total_us"] - 1e-9
        for arm in ("rewrite_only", "sharding_only")
    )
    row["combined_vs_baseline"] = baseline / combined if combined else None
    # Check a real backward program separately. This is not autodiff through
    # explicit Shardy collectives or a measured training benchmark.
    function = jax.grad(
        lambda *args: jnp.sum(operation._operation(*args)),
        argnums=tuple(range(len(arrays))),
    )
    backward = directory / "backward.original.mlir"
    backward.write_text(jax.jit(function).lower(*arrays).as_text("stablehlo"))
    rewritten = invoke(
        rewrite_binary,
        [str(backward), "--numerical-policy=relaxed", "--rewrite-report", *aggressive],
        directory / "backward.log",
    )
    (directory / "backward.rewritten.mlir").write_text(rewritten.stdout)
    expected_gradient = jax.tree.leaves(function(*arrays))
    actual_gradient = execute_unsharded(rewritten.stdout, arrays)
    errors = []
    for actual, target in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(actual, target, rtol=5e-4, atol=5e-4)
        errors.append(float(np.max(np.abs(np.asarray(actual) - np.asarray(target)))))
    row["backward"] = {
        "pass": True,
        "max_abs_errors": errors,
        "stop_reasons": re.findall(r"run \d+ stop=(\S+)", rewritten.stderr),
    }
    (directory / "validated.json").write_text(json.dumps(row, indent=2))
    return row


def layernorm_stress(directory: Path, rewrite_binary: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    x = jnp.asarray(np.tile(1e4 + np.linspace(0, 0.1, 8, dtype=np.float32), (4, 1)))

    def variance(x: jax.Array) -> jax.Array:
        centered = x - jnp.mean(x, axis=-1, keepdims=True)
        return jnp.mean(centered * centered, axis=-1)

    original = jax.jit(variance).lower(x).as_text("stablehlo")
    source = directory / "variance.original.mlir"
    source.write_text(original)
    reference_value = np.asarray(variance(x)).tolist()
    result: dict[str, Any] = {
        "original_variance": reference_value,
        "input_offset": 10000,
    }
    for enabled in (False, True):
        rewritten = invoke(
            rewrite_binary,
            [
                str(source),
                "--numerical-policy=relaxed",
                "--rewrite-report",
                f"--allow-raw-moments={str(enabled).lower()}",
            ],
            directory / f"variance.{enabled}.log",
        )
        (directory / f"variance.{enabled}.mlir").write_text(rewritten.stdout)
        result[str(enabled)] = {
            "variance": np.asarray(
                execute_unsharded(rewritten.stdout, (x,))[0]
            ).tolist(),
            "raw_moment_applied": any(
                int(count) > 0
                for count in re.findall(
                    r"rule centered-square-to-raw-moments\S*.*?applied=(\d+)",
                    rewritten.stderr,
                )
            ),
        }
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    return result


def write_report(
    rows: list[dict[str, Any]],
    stress: dict[str, Any],
    path: Path,
    controls: list[dict[str, Any]] | None = None,
) -> None:
    wins = sum(row["combined_beats_both"] for row in rows)
    text = [
        "# LayerNorm and four-arm experiment",
        "",
        f"{wins} of {len(rows)} cases meet the strict criterion: combined estimated cost is lower than both rewrite-only and sharding-only. These are bounded-search results under an illustrative cost model.",
        "",
        "Costs are illustrative Shardy estimates, not measured accelerator latency. All arms preserve the same external input/output layouts; input conversions and output-contract collectives are charged.",
        "",
        "The four arms are: original arithmetic with original layouts; candidate arithmetic with original layouts; original arithmetic with searched layouts; and candidate arithmetic with searched layouts. External contracts stay fixed in every arm.",
        "",
        "Search covers original/compute/depth/memory candidates plus one checked direct application of each relevant target rule. The latter includes compute-expensive alternatives that the profile extractors can omit. Layout search covers replicated or one full `model` axis on each divisible input dimension. It does not search all possible intermediate layouts or all e-graph expressions.",
        "",
        "Arithmetic of every selected candidate was compiled and executed on one CPU against float64 NumPy references. Selected collective IR was verified, reconstructed with adapters, and exported to XLA. Original backward programs were rewritten and numerically compared with JAX gradients. The separate executor benchmark validates distributed execution and synchronized latency; those results belong in ACCELERATOR.md rather than the cost table below.",
        "",
        "LayerNorm raw moments require `--numerical-policy=relaxed --allow-raw-moments`. The flag is off in every engine preset; this experiment opts in unless `--no-raw-moments` is supplied. JSON records the permission for each case. Other workloads use ordinary relaxed algebra.",
        "",
        f"Cost parameters: `{rows[0]['cost_parameters'] if rows else {}}`. These rates are illustrative and are not hardware calibration.",
        "",
        "| Workload | Layout | Dimensions | Baseline us | Rewrite us | Sharding us | Combined us | Combined beats both |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        costs = [row["arms"][arm]["cost"]["total_us"] for arm in ARMS]
        shape = ",".join(
            f"{k}={v}"
            for k, v in row["config"].items()
            if k not in ("layout", "seed", "edge_probability")
        )
        text.append(
            f"| {row['operation']} | {row['layout']} | {shape} | "
            + " | ".join(f"{cost:.4f}" for cost in costs)
            + f" | {'yes' if row['combined_beats_both'] else 'no'} |"
        )
    text += [
        "",
        "## Interpretation",
        "",
        "A selected rewrite or lower communication payload does not by itself meet the strict combined-arm criterion. Ties with rewrite-only or sharding-only are reported as failures of that criterion.",
        "",
        "For each LayerNorm hidden-sharding case, inspect both traffic and collective counts: the combined winner can gather X and perform fewer reductions, increasing bytes while reducing modeled latency. Input gathers are included in the cost. This is algebraic raw-moment exposure followed by Shardy lowering, rather than a manually introduced distributed LayerNorm operator.",
        "",
        "Reduce-dot explicitly retains a project-first rule witness even when compute extraction prefers reduce-first. Projecting every sequence element adds computation; communication savings can be insufficient under these rates. Dense graph convolution and kernel attention also have controls without remote-feature or sequence-summary communication. Dense graph results do not establish sparse neighbor-exchange savings.",
    ]
    if controls:
        text += [
            "",
            "## LayerNorm flag-off controls",
            "",
            "These runs repeat the LayerNorm cases with raw moments disabled. Other relaxed rules and the same layout search remain enabled. All four selected computations and each original backward program are numerically checked again.",
            "",
            "| Layout | Dimensions | Enabled combined us | Disabled combined us | Disabled raw-moment applications |",
            "|---|---|---:|---:|---:|",
        ]
        for control in controls:
            enabled = next(
                row
                for row in rows
                if row["operation"] == control["operation"]
                and row["layout"] == control["layout"]
                and row["reverse"] == control["reverse"]
            )
            applied = sum(
                stats["applied"]
                for name, stats in control["rules"].items()
                if name.startswith("centered-square-to-raw-moments")
            )
            text.append(
                f"| {control['layout']} | {'reversed' if control['reverse'] else 'favorable'} | {enabled['arms']['combined']['cost']['total_us']:.4f} | {control['arms']['combined']['cost']['total_us']:.4f} | {applied} |"
            )
    text += [
        "",
        "## Selected communication and search",
        "",
        "Bytes are per-device traffic under the existing ring/payload model, including adapters. Logical payload diagnostics in JSON use a different global tensor-size convention.",
        "",
        "| Case | Arm | Candidate | Traffic bytes/device | Input adapter us | Collectives |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        for arm in ARMS:
            selected = row["arms"][arm]
            text.append(
                f"| {row['operation']}/{row['layout']}/{'reversed' if row['reverse'] else 'favorable'} | {arm} | {selected['candidate']} | {selected['modeled_bytes_per_device']:.0f} | {selected['input_adapters_cost']['total_us']:.4f} | {selected['collectives']} |"
            )
    limits = [
        f"{row['operation']}/{row['layout']}/{'reversed' if row['reverse'] else 'favorable'}: {row['saturation']}"
        for row in rows
        if row["saturation"]["stop"] != "saturated"
        or row["layout_truncated"]
        or row["extraction_profiles_skipped"]
        or row["extraction_limited"]
    ]
    text += [
        "",
        f"{len(rows)} main cases, {len(rows) * 4} arms; {len(controls or [])} additional flag-off controls. Combined strictly beats both single interventions in {wins} main cases.",
        "",
        "Search limits:",
        "",
    ]
    text += [f"- {limit}" for limit in limits] or [
        "- No reported saturation/layout/candidate-cap limits."
    ]
    text += [
        "",
        "## Aggressive variance stress result",
        "",
        "A variance-only lowering exposes cancellation directly, without relying on which LayerNorm candidate the cost model selects.",
        "",
        "```json",
        json.dumps(stress, indent=2),
        "```",
        "",
        "Detailed metrics, layouts, rule application counts, selected IR, XLA inputs, numerical errors and backward logs are retained beside this report in per-case directories.",
    ]
    path.write_text("\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quick", action="store_true", help="Small shapes for regression checks"
    )
    parser.add_argument("--operation", choices=WORKLOADS, action="append")
    parser.add_argument("--binary", type=Path, default=TOOLS / "workload_ablation")
    parser.add_argument(
        "--rewrite-binary", type=Path, default=TOOLS / "parse_stablehlo"
    )
    parser.add_argument("--max-layouts", type=int, default=256)
    parser.add_argument(
        "--raw-moments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Explicitly opt LayerNorm into raw moments (default for this experiment)",
    )
    parser.add_argument(
        "--layernorm-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat LayerNorm with the aggressive flag off",
    )
    parser.add_argument("--compute-work-per-us", type=float, default=1e6)
    parser.add_argument("--bandwidth-bytes-per-us", type=float, default=50000)
    parser.add_argument("--collective-latency-us", type=float, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.operation or WORKLOADS:
        for layout in OPERATIONS[name][0].layouts:
            for reverse in (False, True):
                label = f"{name}.{layout}.{'reversed' if reverse else 'favorable'}"
                print(f"Running {label}", flush=True)
                rows.append(
                    run_case(
                        name,
                        layout,
                        reverse,
                        args.output_dir / label,
                        quick=args.quick,
                        binary=args.binary,
                        rewrite_binary=args.rewrite_binary,
                        max_layouts=args.max_layouts,
                        aggressive_raw_moments=args.raw_moments,
                        cost_options=(
                            f"--compute-work-per-us={args.compute_work_per_us}",
                            f"--bandwidth-bytes-per-us={args.bandwidth_bytes_per_us}",
                            f"--collective-latency-us={args.collective_latency_us}",
                        ),
                    )
                )
                (args.output_dir / "summary.json").write_text(
                    json.dumps(rows, indent=2)
                )
    controls = []
    if args.raw_moments and args.layernorm_controls:
        for row in rows:
            if row["operation"] != "layernorm_linear":
                continue
            label = f"layernorm_linear.{row['layout']}.{'reversed' if row['reverse'] else 'favorable'}"
            print(f"Running flag-off control {label}", flush=True)
            controls.append(
                run_case(
                    row["operation"],
                    row["layout"],
                    row["reverse"],
                    args.output_dir / "layernorm_controls" / label,
                    quick=args.quick,
                    binary=args.binary,
                    rewrite_binary=args.rewrite_binary,
                    max_layouts=args.max_layouts,
                    aggressive_raw_moments=False,
                    cost_options=(
                        f"--compute-work-per-us={args.compute_work_per_us}",
                        f"--bandwidth-bytes-per-us={args.bandwidth_bytes_per_us}",
                        f"--collective-latency-us={args.collective_latency_us}",
                    ),
                )
            )
            (args.output_dir / "layernorm_controls.json").write_text(
                json.dumps(controls, indent=2)
            )
    stress = layernorm_stress(args.output_dir / "layernorm_stress", args.rewrite_binary)
    write_report(rows, stress, args.output_dir / "REPORT.md", controls)
    print(f"Report: {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
