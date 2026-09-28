"""Compile the Megatron layer and save XLA dumps under lowered/."""

import argparse
import os
from pathlib import Path


def configure_xla_flags(dump_dir: Path) -> None:
    """Set XLA and StableHLO dump flags before importing JAX."""
    flags = os.environ.get("XLA_FLAGS", "").split()
    flags = [
        flag for flag in flags
        if not flag.startswith(("--xla_dump_to=", "--xla_dump_hlo_pass_re="))
    ]
    if not any(flag.startswith("--xla_force_host_platform_device_count=") for flag in flags):
        flags.append("--xla_force_host_platform_device_count=4")
    flags.extend((
        f"--xla_dump_to={dump_dir}",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_module_re=jit__operation",
        "--xla_dump_hlo_pass_re=shardy|spmd-partitioner",
    ))
    os.environ["XLA_FLAGS"] = " ".join(flags)
    os.environ["JAX_DUMP_IR_TO"] = str(dump_dir)
    os.environ["JAX_DUMP_IR_MODES"] = "stablehlo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "lowered",
        help="Parent directory for operation-specific XLA dumps",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Also execute the compiled layer"
    )
    args = parser.parse_args()

    dump_dir = (args.output_dir / "megatron_layer").resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    configure_xla_flags(dump_dir)

    from ml_workloads import MegatronLayer, MegatronLayerConfig

    layer = MegatronLayer()
    config = MegatronLayerConfig(
        batch=args.batch,
        sequence=args.sequence,
        hidden=args.hidden,
        ffn=args.ffn,
        seed=args.seed,
    )
    inputs = layer.create_inputs(config)
    compiled = layer.compile(inputs, out_sharding=layer.output_sharding)
    print(f"XLA dumps saved under {dump_dir}")

    if args.execute:
        output = compiled(*inputs.arrays())
        output.block_until_ready()
        print(f"Output shape: {output.shape}")
        print(f"Output sharding: {output.sharding}")


if __name__ == "__main__":
    main()
