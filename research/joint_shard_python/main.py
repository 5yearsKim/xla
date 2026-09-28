"""Lower an original ML workload and save StableHLO and XLA dumps."""

import argparse
import os
from dataclasses import fields
from pathlib import Path


def configure_xla_flags(dump_dir: Path) -> None:
    """Set XLA and StableHLO dump flags before importing JAX."""
    flags = os.environ.get("XLA_FLAGS", "").split()
    flags = [
        flag
        for flag in flags
        if not flag.startswith(("--xla_dump_to=", "--xla_dump_hlo_pass_re="))
    ]
    if not any(
        flag.startswith("--xla_force_host_platform_device_count=") for flag in flags
    ):
        flags.append("--xla_force_host_platform_device_count=4")
    flags.extend(
        (
            f"--xla_dump_to={dump_dir}",
            "--xla_dump_hlo_as_text",
            "--xla_dump_hlo_module_re=jit__operation",
            "--xla_dump_hlo_pass_re=shardy|spmd-partitioner",
        )
    )
    os.environ["XLA_FLAGS"] = " ".join(flags)
    os.environ["JAX_DUMP_IR_TO"] = str(dump_dir)
    os.environ["JAX_DUMP_IR_MODES"] = "stablehlo"


def add_common_arguments(
    parser: argparse.ArgumentParser, *, operations: tuple[str, ...] | None = None
) -> None:
    parser.add_argument(
        "--operation",
        default="megatron_layer",
        choices=operations,
        help="Original workload to lower (default: megatron_layer)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "lowered",
        help="Parent directory for operation-specific XLA dumps",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Also execute the compiled workload"
    )


def main() -> None:
    # Parse the dump destination first: JAX must see these flags before import.
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_common_arguments(preliminary)
    selected, _ = preliminary.parse_known_args()
    if not selected.operation.isidentifier():
        preliminary.error("operation must be a valid identifier")
    dump_dir = (selected.output_dir / selected.operation).resolve()
    configure_xla_flags(dump_dir)

    import jax

    from ml_workloads import OPERATIONS

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(parser, operations=tuple(OPERATIONS))
    if selected.operation not in OPERATIONS:
        parser.error(
            f"unknown operation {selected.operation!r}; choose from {tuple(OPERATIONS)}"
        )
    operation_type, config_type = OPERATIONS[selected.operation]
    for field in fields(config_type):
        options = {"choices": operation_type.layouts} if field.name == "layout" else {}
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
            help=f"{field.name.replace('_', ' ')} (default: {field.default})",
            **options,
        )
    if getattr(operation_type, "supports_full_tensor_boundary", False):
        parser.add_argument(
            "--full-tensor-boundary",
            action="store_true",
            help="Explicitly replicate the report's intermediate (a rewrite boundary)",
        )
    if selected.operation in ("layernorm_linear", "barlow_twins"):
        parser.add_argument("--epsilon", type=float, default=1e-5)
    if selected.operation == "barlow_twins":
        parser.add_argument("--off-diagonal-weight", type=float, default=0.005)
    if selected.operation == "moe_router":
        parser.add_argument("--top-k", type=int, default=2)
        parser.add_argument(
            "--weight-normalization", choices=("selected", "full"), default="selected"
        )
    args = parser.parse_args()

    constructor_options = {
        name: getattr(args, name)
        for name in (
            "full_tensor_boundary",
            "epsilon",
            "off_diagonal_weight",
            "top_k",
            "weight_normalization",
        )
        if hasattr(args, name)
    }
    try:
        operation = operation_type(**constructor_options)
        config = config_type(
            **{field.name: getattr(args, field.name) for field in fields(config_type)}
        )
        inputs = operation.create_inputs(config)
    except ValueError as error:
        parser.error(str(error))

    dump_dir.mkdir(parents=True, exist_ok=True)
    lowered = operation.lower(inputs, out_sharding=operation.output_sharding)
    # This is the original program, before backend fusion/reassociation.
    original_path = dump_dir / "original.mlir"
    original_path.write_text(lowered.as_text("stablehlo"))
    compiled = lowered.compile()
    print(f"Original StableHLO saved to {original_path}")
    print(f"XLA dumps saved under {dump_dir}")

    if args.execute:
        output = compiled(*inputs.arrays())
        jax.block_until_ready(output)
        leaves = jax.tree.leaves(output)
        for index, array in enumerate(leaves):
            label = "Output" if len(leaves) == 1 else f"Output {index}"
            print(f"{label} shape: {array.shape}")
            print(f"{label} sharding: {array.sharding}")


if __name__ == "__main__":
    main()
