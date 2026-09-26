# StableHLO, egg-c, and Shardy prototype

Build from the XLA workspace root:

```sh
bazel build //research/joint_shard:run_shardy
bazel-bin/research/joint_shard/run_shardy \
  research/joint_shard/testdata/dot_general.mlir \
--case=all --round-trip --dump-dir=/tmp/joint_shard
```

Alternatively, from the workspace root:

```sh
bazel run //research/joint_shard:run_shardy -- \
  "$PWD/research/joint_shard/testdata/dot_general.mlir" \
  --case=all --round-trip --dump-dir=/tmp/joint_shard
```

`bazel run` changes the working directory to the target's runfiles directory.
The bundled samples are included in `data`; for other input files, pass an
absolute path as above.

The driver clones the parsed module for each case. Omit `--round-trip` to run
Shardy directly on the input. `--case=all` runs the three matmul cases below.

| Case | LHS dimensions | RHS dimensions | Result dimensions |
|---|---|---|---|
| compatible | data, unsharded | unsharded, model | data, model |
| gather | data, unsharded | unsharded, model | data, unsharded |
| contracting | data, model | model, unsharded | data, unsharded |

All boundary dimensions are closed. The mesh is `data=2, model=2`: four logical
partitions, not two runtime replicas and two partitions. Unused axes are
implicitly replicated. Boundaries are function argument/result attributes;
operation result shardings are propagated by Shardy.

Observed output for the supplied matmul after collective conversion:

| Case | Explicit reshards before conversion | Final communication |
|---|---|---|
| compatible | 0 | none |
| gather | 1 | one `sdy.all_gather` |
| contracting | 0 | one `sdy.all_reduce` |

Run the existing rewritten elementwise example:

```sh
bazel-bin/research/joint_shard/run_shardy \
  research/joint_shard/testdata/add_mul.mlir \
  --case=elementwise --round-trip --dump-dir=/tmp/joint_shard/elementwise
```

`--round-trip` applies only add commutativity. The demonstration extractor
prefers an add with the smaller subtree on the left; it is not a communication
cost model. The single-dot sample has no applicable rewrite, but its dot is
imported, extracted, and rebuilt before entering Shardy.

`--stop-after=propagation|reshards|collectives` selects the last stage (default:
collectives). Verified snapshots are written as:

```text
00_exported.mlir
01_boundary_shardings.mlir
02_propagated.mlir
03_explicit_reshards.mlir
04_collectives.mlir
```

For `--case=all`, each case gets a separate dump subdirectory. Stdout reports
communication operation counts for each stage and prints the final module.
Shardy's propagation pipeline includes preparation and export cleanup and can
already insert boundary reshards. Full explicit-reshard insertion can also
insert reduction collectives. Therefore zero reshards does not imply zero
communication; inspect collective kinds too.

Output still uses global tensor shapes and Shardy collectives (`sdy.all_gather`,
`sdy.all_reduce`, etc.). It is verified compiler IR, not yet device-local
executable code. No automatic sharding search, pruning, or communication cost
model is implemented.

The bridge supports arguments, add, multiply, exp, and rank-2 `dot_general`
with contracting dimensions `[1] x [0]` and no batching. Dot types and all
attributes are interned into context-owned descriptors shared by importer and
exporter. Descriptor IDs are scoped to one round trip, not a persistent format.
