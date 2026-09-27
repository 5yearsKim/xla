#!/usr/bin/env python3
"""Compile and execute an optimizer-exported xla_input.mlir with prebuilt PJRT."""

import argparse
import sys
from pathlib import Path

import numpy as np

from prebuilt_runner import SelectedExecutable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="XLA-ready StableHLO MLIR")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inputs", type=Path, help="NPZ with global arg0, arg1, ...")
    parser.add_argument(
        "--outputs", type=Path, help="Save global result0, result1, ... as NPZ"
    )
    parser.add_argument("--dump-dir", type=Path, help="Save standard XLA HLO dumps")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.compile_only and (args.inputs or args.outputs):
        parser.error("--inputs and --outputs require execution")
    program = SelectedExecutable(
        args.input.read_text(), platform=args.platform, dump_dir=args.dump_dir
    )
    print(f"Compiled on {args.platform}: {program.partitions} partitions")
    if args.compile_only:
        return 0
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as data:
            expected = [f"arg{i}" for i in range(len(program.inputs))]
            if set(data.files) != set(expected):
                raise ValueError(f"input NPZ must contain exactly {expected}")
            inputs = [data[key] for key in expected]
    else:
        inputs = program.random_inputs(args.seed)
    outputs = program.execute(inputs, repeats=args.repeats)
    destination = args.outputs
    if destination is None and args.dump_dir:
        destination = args.dump_dir / "outputs.npz"
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            np.savez(file, **{f"result{i}": x for i, x in enumerate(outputs)})
    print(f"Executed {args.repeats} time(s): {len(outputs)} global outputs")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"run: {error}", file=sys.stderr)
        sys.exit(1)
