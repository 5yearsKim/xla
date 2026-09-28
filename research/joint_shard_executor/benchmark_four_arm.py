"""Synchronized execution of the exact four-arm XLA artifacts, without reselection."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# Share the host with other jobs without reserving most device memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from prebuilt_runner import SelectedExecutable  # noqa: E402

ARMS = ("baseline", "rewrite_only", "sharding_only", "combined")


def wait(shards: list[list[jax.Array]]) -> None:
    for output in shards:
        for shard in output:
            shard.block_until_ready()


def benchmark_case(
    directory: Path, *, platform: str, warmup: int, repeats: int
) -> dict[str, Any]:
    with np.load(directory / "inputs.npz", allow_pickle=False) as data:
        inputs = [data[f"arg{i}"] for i in range(len(data.files))]
    reference = np.load(directory / "reference.npy", allow_pickle=False)
    programs = {
        arm: SelectedExecutable(
            (directory / f"{arm}.xla_input.mlir").read_text(), platform=platform
        )
        for arm in ARMS
    }
    arguments = {
        arm: program.prepare_inputs(inputs) for arm, program in programs.items()
    }
    for arrays in arguments.values():
        for array in arrays:
            array.block_until_ready()
    for _ in range(warmup):
        for arm, program in programs.items():
            wait(program.execute_device(arguments[arm]))
    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    outputs = {}
    # Rotate arm order each round, keeping compilation and transfers outside
    # the timed interval. Each sample includes host dispatch and device wait.
    for repeat in range(repeats):
        order = ARMS[repeat % len(ARMS) :] + ARMS[: repeat % len(ARMS)]
        for arm in order:
            start = time.perf_counter_ns()
            shards = programs[arm].execute_device(arguments[arm])
            wait(shards)
            samples[arm].append((time.perf_counter_ns() - start) / 1000)
            outputs[arm] = shards
    result = {}
    for arm, program in programs.items():
        actual = program.materialize_outputs(outputs[arm])[0]
        np.testing.assert_allclose(actual, reference, rtol=3e-4, atol=3e-4)
        result[arm] = {
            "program_sha256": hashlib.sha256(
                (directory / f"{arm}.xla_input.mlir").read_bytes()
            ).hexdigest(),
            "median_us": float(np.median(samples[arm])),
            "min_us": min(samples[arm]),
            "p95_us": float(np.percentile(samples[arm], 95)),
            "samples_us": samples[arm],
            "numerical_pass": True,
            "max_abs_error": float(np.max(np.abs(actual - reference))),
        }
    result["combined_lower_median_than_both"] = all(
        result["combined"]["median_us"] < result[arm]["median_us"]
        for arm in ("rewrite_only", "sharding_only")
    )
    result["distinct_from_both"] = all(
        result["combined"]["program_sha256"] != result[arm]["program_sha256"]
        for arm in ("rewrite_only", "sharding_only")
    )
    result["combined_beats_both"] = (
        result["combined_lower_median_than_both"] and result["distinct_from_both"]
    )
    return result


def write_report(result: dict[str, Any], path: Path) -> None:
    successful = [case for case in result["cases"].values() if "error" not in case]
    text = [
        "# Synchronized selected-program execution",
        "",
        f"Platform: {result['platform']}; JAX {result['jax_version']}; warmup {result['warmup']}, timed rounds {result['repeats']}. Visible GPUs: `{result['cuda_visible_devices']}`.",
        "",
        f"Runtime flags: `XLA_FLAGS={result['xla_flags']}`, `NVIDIA_TF32_OVERRIDE={result['nvidia_tf32_override']}`. Numerical tolerance: rtol=atol=0.0003.",
        "",
        "These are executions of the exact exported four-device programs, including selected input adapters and output collectives. There is no rewrite or layout reselection in the executor. Each successful arm passes a distributed output check against the saved independent float64 reference.",
        "",
        "Timing includes Python/PJRT host dispatch and synchronized completion of all output shards. Compilation, input upload and output download are excluded. Arms rotate order each round. Medians are single-run observations, not statistically established speedups or device-only kernel times; shared GPU load can affect comparisons. The snapshot records existing device allocations.",
        "",
        f"{len(successful)} successful cases; {len(result['cases']) - len(successful)} failed cases. Combined has lower median than both single interventions and a distinct exported artifact in {sum(case['combined_beats_both'] for case in successful)} successful cases. Identical-artifact comparisons are excluded because different timings for the same program are noise.",
        "",
        "| Case | Baseline median us | Rewrite median us | Sharding median us | Combined median us | Lower than both, distinct artifacts |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, case in result["cases"].items():
        if "error" in case:
            text += [f"| {label} | failed | | | | |"]
            continue
        text.append(
            f"| {label} | "
            + " | ".join(f"{case[arm]['median_us']:.3f}" for arm in ARMS)
            + f" | {'yes' if case['combined_beats_both'] else 'no'} |"
        )
    failures = {
        label: case["error"]
        for label, case in result["cases"].items()
        if "error" in case
    }
    text += [
        "",
        "Every sample, p95, minimum, numerical error, and any failures are retained in `accelerator.json`.",
        "",
        "GPU snapshot:",
        "",
        "```text",
        result.get("gpu_snapshot", "CPU execution"),
        "```",
    ]
    if failures:
        text += ["", "Failures:", "", "```json", json.dumps(failures, indent=2), "```"]
    path.write_text("\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="four_arm output directory")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--case", action="append", help="Exact case directory name")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    result: dict[str, Any] = {
        "platform": args.platform,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "jax_version": jax.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "timing": "host dispatch plus synchronized device completion; no compile or transfer",
        "cases": {},
    }
    if args.platform == "gpu":
        try:
            result["gpu_snapshot"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            result["gpu_snapshot"] = "unavailable"
    rows = json.loads((args.directory / "summary.json").read_text())
    for row in rows:
        label = f"{row['operation']}.{row['layout']}.{'reversed' if row['reverse'] else 'favorable'}"
        if args.case and label not in args.case:
            continue
        print(f"Benchmarking {label}", flush=True)
        try:
            result["cases"][label] = benchmark_case(
                args.directory / label,
                platform=args.platform,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        except Exception as error:
            # Preserve a failed distributed validation as evidence and continue
            # independent cases. Never record an unsuccessful run as a speedup.
            result["cases"][label] = {"error": str(error)}
            print(f"Failed: {error}", flush=True)
        (args.directory / "accelerator.json").write_text(json.dumps(result, indent=2))
    write_report(result, args.directory / "ACCELERATOR.md")


if __name__ == "__main__":
    main()
