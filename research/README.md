# Joint rewrite and sharding projects

These projects connect through MLIR files:

```text
JAX workload → joint_shard_python → StableHLO MLIR
            → joint_shard → xla_input.mlir
            → joint_shard_executor + runtime inputs → computed results
```

| Project | Role | Input | Output |
|---|---|---|---|
| [joint_shard_python](joint_shard_python/README.md) | Generates a Megatron-layer workload using JAX | Workload configuration and input shardings | StableHLO MLIR and XLA HLO dumps |
| [joint_shard](joint_shard/README.md) | Optimizes computation rewrites and sharding choices | StableHLO MLIR, mesh/sharding constraints, and search options | In chain/DAG mode, `selected.mlir` and `xla_input.mlir` |
| [joint_shard_executor](joint_shard_executor/README.md) | Compiles and executes the optimized program using prebuilt XLA/PJRT | `xla_input.mlir` and runtime arrays (generated or supplied as NPZ) | Computed arrays and optional output NPZ/compiler dumps |

`selected.mlir` records the chosen global StableHLO + Shardy program;
`xla_input.mlir` is its XLA-ready export and the executor's input.
The executor does not rerun optimizer search or Shardy propagation.

The handoffs are manual. `joint_shard_python` currently provides a specific
workload, rather than a general exporter for arbitrary JAX code; it can also
compile and optionally execute that workload directly through JAX.

See the [optimizer guide](misc/REGION_OPTIMIZER.md) for search details and the
project READMEs above for commands.
