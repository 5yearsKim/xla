# StableHLO, egg-c, and Shardy prototype

The TensorLang implementation and extension contracts are in
[TENSORLANG.md](TENSORLANG.md). The bridge uses structural nodes with typed
operator attributes;
typed e-class analysis checks tensor types, and ordinary rewrite patterns match
complete operator attributes. TensorNode implements the language outside egg-c;
the engine stores it directly and supports custom search/application callbacks.

The project uses component-based Bazel packages, with headers beside their
implementations:

```text
joint_shard/
  BUILD.bazel       # aggregate integration-test suite
  bridge/           # TensorLang nodes and StableHLO ↔ egg-c import/export
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

This project uses C++20. Build with `--config=joint_shard`: the root `.bazelrc`
scopes C++20 source flags to joint_shard and selects the hermetic GCC 12 / glibc
2.35 sysroot, because XLA's default GCC 8 library lacks C++20 headers. Dependencies
retain C++17 source flags and use the same selected sysroot. This configuration
requires glibc 2.35 or newer to run; it does not change other XLA builds unless
selected. Engine CMake and standalone Bazel also require C++20. The engine checks
external node and analysis APIs with `Language` and `AnalysisFor` concepts.

Build and test from the XLA workspace root:

```sh
bazel build --config=joint_shard //research/joint_shard/tools:run_shardy //research/joint_shard/tools:parse_stablehlo
bazel test --config=joint_shard //research/joint_shard:tests
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
outputs of shared rewrite sessions within each supported island. The annotated operation itself never enters the
rewrite graph, so rewriting cannot move across or remove the fixed point.
Function argument/result attributes remain on the original function.

Preserved operations are visited even when unused, keeping dangling constraints
and unused annotated intermediates. Use-scoped constraints retain their original
uses. Unsupported operations are also boundaries, allowing supported regions of
larger modules to be rewritten without importing every operation. Rewriting
supports single-block functions, imports canonical reducer bodies, and leaves
other nested regions intact.

The bridge now covers general/batched dots, common elementwise operations,
constants, transpose, static reshape, broadcast, and canonical reductions.
Semantic parameters live in native typed attrs. Shared standalone inference
validates proposed expressions before insertion/export. Derived `result_layout`
and `xla_shape` hints are recomputed downstream; unknown attrs remain boundaries.

The default pipeline loads embedded `tensor.rules` and appends attribute-aware
rules. Strict numerical semantics are the default; floating algebra requires
`--numerical-policy=relaxed` or explicit numerical permissions. Every captured
operator occurrence is checked, and a complete RHS is validated before mutation.
Search limits and rejection reasons are exposed through `--rewrite-report`.
Contiguous supported islands share one graph and export cache across all outputs.

`run_shardy --round-trip` compares the original with one rewritten candidate;
`--candidates=4` adds depth and memory extraction profiles. Candidates run on
independent clones and are scored by logical collective payload, then estimated
compute work. Ties and unknown estimates retain the original. This is a bounded
comparison with a coarse global-payload proxy, not topology-aware sharding search.
See [TENSORLANG.md](TENSORLANG.md) for the contracts and remaining extensions.

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

`00_input` is the module entering Shardy. With multiple candidates, snapshots
are saved in separate candidate directories. Stdout prints the selected module;
costs, selection, and optional rewrite reports go to stderr. Full explicit-reshard insertion can also insert reduction collectives;
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
