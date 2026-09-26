# StableHLO, egg-c, and Shardy prototype

The project uses component-based Bazel packages, with headers beside their
implementations:

```text
joint_shard/
  BUILD.bazel       # aggregate integration-test suite
  bridge/           # StableHLO ↔ egg-c import/export and descriptors
  transforms/       # rewriting between fixed sharding points
  sharding/         # Shardy pipeline and snapshots
  tools/            # command-line executables
  tests/            # cross-component C++ integration tests
  testdata/         # MLIR fixtures
  egg-c/            # e-graph engine, with its own package structure
```

Each component has its own `BUILD.bazel`. Libraries are visible only within
`joint_shard`; tools and integration tests depend on those component targets.
Component-specific unit tests can live beside the implementation they test.
New source files belong in their component package, with explicit `srcs`/`hdrs`.

Build and test from the XLA workspace root:

```sh
bazel build //research/joint_shard/tools:run_shardy //research/joint_shard/tools:parse_stablehlo
bazel test //research/joint_shard:tests
bazel-bin/research/joint_shard/tools/run_shardy \
  research/joint_shard/testdata/dot_general.mlir \
  --round-trip --dump-dir=/tmp/joint_shard
```

Inputs provide their own `sdy.mesh` definitions and sharding annotations. The
runner preserves them instead of generating boundary layouts. All non-maximal,
non-empty meshes must have the same device count; `mhlo.num_partitions`, when
present, must match. An explicit partitioning mesh is required.

An absent sharding is unconstrained. Closed dimensions, including empty `{}`
dimensions, specify fixed layouts. Open `?` dimensions permit further
propagation. Native Shardy attributes preserve mesh references, sub-axes,
replication, and priorities without a second sharding representation.

## Rewriting around fixed points

`--round-trip` rewrites supported unannotated computations between preserved
operations. An operation carrying `sdy.sharding` is a boundary, including a
partially open annotation. Explicit `sdy.sharding_constraint` operations are also
boundaries. Their results enter the e-graph as opaque leaves; their operands are
independent rewrite roots. The annotated operation itself never enters the
rewrite graph, so rewriting cannot move across or remove the fixed point.
Function argument/result attributes remain on the original function.

Preserved operations are visited even when unused, keeping dangling constraints
and unused annotated intermediates. Use-scoped constraints retain their original
uses. Unsupported operations are also boundaries, allowing supported regions of
larger modules to be rewritten without importing every operation. Rewriting
supports single-block functions and leaves nested regions intact.

The current bridge imports add, multiply, exp, and rank-2 `dot_general` with
contracting dimensions `[1] x [0]` and no batching. Add/multiply/exp with other
attributes are preserved because their exporter paths cannot carry those
attributes. Dot types and attributes use context-owned descriptors shared by
importer and exporter. Descriptor IDs are scoped to one region round trip.

Only add commutativity is enabled. The demonstration extractor prefers an add
with the smaller subtree on the left; it is not a communication cost model.
Separate region roots may rebuild shared unconstrained computations separately.
No automatic sharding search, pruning, or communication cost model is implemented.

To inspect rewriting without running propagation:

```sh
bazel-bin/research/joint_shard/tools/parse_stablehlo \
  research/joint_shard/testdata/add_mul.mlir
```

## Propagation and communication

The pipeline verifies the annotated module, propagates shardings, inserts
explicit reshards, then converts them to collectives. Reshards are not inserted
unconditionally at function entry: Shardy resolves incompatible layouts where
conversion is needed. The propagation pipeline includes preparation and export
cleanup and can already insert boundary reshards. Remaining explicit constraints
are converted to reshards before full reshard insertion.

`--stop-after=propagation|reshards|collectives` selects the last stage (default:
collectives). Verified snapshots are:

```text
00_input.mlir
01_propagated.mlir
02_explicit_reshards.mlir
03_collectives.mlir
```

`00_input` is the module entering Shardy, after rewriting when enabled. Stdout
reports communication operation counts for each stage and prints the final
module. Full explicit-reshard insertion can also insert reduction collectives;
zero reshards does not imply zero communication.

The annotated matmul samples use `data=2, model=2`:

| File | LHS dimensions | RHS dimensions | Result dimensions | Expected communication |
|---|---|---|---|---|
| `dot_general.mlir` | data, unsharded | unsharded, model | data, model | none |
| `dot_gather.mlir` | data, unsharded | unsharded, model | data, unsharded | all-gather |
| `dot_contracting.mlir` | data, model | model, unsharded | data, unsharded | all-reduce |

For other input files with `bazel run`, pass an absolute path because Bazel
changes the working directory to the target's runfiles directory.

Output uses global tensor shapes and Shardy collectives (`sdy.all_gather`,
`sdy.all_reduce`, etc.). It is verified compiler IR, not device-local executable
code.
