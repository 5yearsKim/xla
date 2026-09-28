# StableHLO, egg-c, and Shardy prototype

The region optimizer architecture, policies and experiments are in
[REGION_OPTIMIZER.md](../misc/REGION_OPTIMIZER.md).

The [LayerNorm/four-arm report](results/four_arm/REPORT.md) compares fixed-contract
baseline, rewrite-only, sharding-only, and combined experiments. The original
LayerNorm's centered variance can be rewritten to raw moments with the explicit
`--numerical-policy=relaxed --allow-raw-moments` opt-in. This flag remains off in
every engine preset. The experiment driver is `tools:workload_ablation`, with
the [Python runner](../joint_shard_python/README.md#four-arm-experiment-and-aggressive-layernorm)
providing shape/layout sweeps, numerical checks, and reports.

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
  BUILD.bazel       # aggregate unit/integration-test suite
  bridge/           # TensorLang nodes and StableHLO ↔ egg-c import/export
  transforms/       # region discovery, saturation and candidate extraction
  sharding/         # boundary layouts, Shardy evaluation and costs
  export/           # selected Shardy path to XLA-ready StableHLO
  search/           # typed interfaces, region tables, pair composition and orchestration
  reporting/        # report formatting and artifact output
  tools/            # command-line executables and CLI parsing
  tests/            # cross-component integration tests and shared test support
  testdata/         # MLIR fixtures
  egg-c/            # e-graph engine, with its own package structure
```

Each component has its own `BUILD.bazel`. Libraries are visible only within
`joint_shard`; tools and integration tests depend on those component targets.
Component-specific unit tests live beside the implementation they test.
New source files belong in their component package, with explicit `srcs`/`hdrs`.

C++ APIs live in the `joint_shard` namespace. `summarizeRegions` returns
structured results and accepts synchronous `OptimizationObserver` callbacks;
search and Shardy evaluation do not write files. The optimizer tools connect an
`ArtifactWriter` when `--dump-dir` is requested and format results with
`formatReport`. CLI parsing belongs to `tools/`, while library callers populate
typed options directly.

Shardy retains only the final stage by default. `SnapshotCapture::AllStages`
enables intermediate snapshots, and `collect_statistics` separately enables
global work/payload diagnostics. Evaluations return final lowered IR and the
optimizer's time cost. Region summaries retain winners incrementally, then
assign stable plan IDs with `finalizePlans`.

This project uses C++20. Build with `--config=joint_shard`: the root `.bazelrc`
scopes C++20 source flags to joint_shard and selects the hermetic GCC 12 / glibc
2.35 sysroot, because XLA's default GCC 8 library lacks C++20 headers. Dependencies
retain C++17 source flags and use the same selected sysroot. This configuration
also keeps the `export/` bridge in C++17 to match Abseil's `SourceLocation` ABI.
It requires glibc 2.35 or newer to run; it does not change other XLA builds unless
selected. Engine CMake and standalone Bazel also require C++20. The engine checks
external node and analysis APIs with `Language` and `AnalysisFor` concepts.

Build and test from the XLA workspace root:

```sh
bazel build --config=joint_shard //research/joint_shard/tools:summarize_regions //research/joint_shard/tools:run_shardy //research/joint_shard/tools:parse_stablehlo
bazel test --config=joint_shard //research/joint_shard:tests
bazel-bin/research/joint_shard/tools/run_shardy \
  research/joint_shard/testdata/dot_general.mlir \
  --dump-dir=/tmp/joint_shard
```

The tools accept options before or after the input path and provide `--help`.
For example, `run_shardy --stop-after propagation input.mlir` is equivalent to
placing the input path first. `parse_stablehlo` also supports
`--rewrite-report`; rewrite controls shared with `summarize_regions` are listed
in each tool's generated help.

Inputs provide their own `sdy.mesh` definitions and sharding annotations. The
runner preserves them instead of generating boundary layouts. All non-maximal,
non-empty meshes must have the same device count; `mhlo.num_partitions`, when
present, must match. An explicit partitioning mesh is required.

An absent sharding is unconstrained. Closed dimensions, including empty `{}`
dimensions, specify fixed layouts. Open `?` dimensions permit further
propagation. Native Shardy attributes preserve mesh references, sub-axes,
replication, and priorities without a second sharding representation.

## Rewriting around fixed points

`parse_stablehlo` rewrites supported unannotated computations between preserved
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
Semantic parameters live in native typed attrs. Rules use typed C++ patterns,
operator/property guards, and optional checked replacement callbacks. Dot
dimensions bind as one value; replacement dot shapes are inferred. See
[TENSORLANG.md](TENSORLANG.md#c-pattern-definitions) for examples.
Shared standalone inference
validates proposed expressions before insertion/export. Derived `result_layout`
and `xla_shape` hints are recomputed downstream; unknown attrs remain boundaries.

The default pipeline compiles `tensor_rules.cc` and appends computed attribute
rules. Relaxed floating-point algebra is the default and can change numerical results.
Use `--numerical-policy=strict` to preserve floating evaluation, or configure
explicit numerical permissions. Every captured
operator occurrence is checked, and a complete RHS is validated before mutation.
Search limits and rejection reasons are exposed through `--rewrite-report`.
[WORKLOAD_REWRITES.md](WORKLOAD_REWRITES.md) records rule coverage, CPU numerical
checks on the Python originals, and remaining saturation/importer limits.
Contiguous supported islands share one graph and export cache across all outputs.

`summarize_regions` splits supported computations into bounded regions, saturates
once per region, enumerates input layouts, and evaluates each unique
original/compute/depth/memory candidate once per input assignment. Shardy infers
unconstrained output layouts, including combined mesh-axis layouts. Required
output annotations and function contracts remain constraints. The optimizer
retains the cheapest candidate per complete inferred boundary and prunes plans
served more cheaply by another implementation plus directed reshards.
`--max-input-states=256` caps input enumeration; outputs add no search factor.
Dominated implementations release their lowered IR and retain replacement
recipes for their boundary contracts. It prints
region frontiers and preserves the source module. Add `--compose-regions=0,1`
to compose an adjacent pair, prune its outer boundaries, and emit surviving modules; see the
[composition guide](../misc/REGION_OPTIMIZER.md#compose-two-regions). Candidate count
defaults to a cap of 32, with at most five distinct profile candidates in this milestone.

Add `--optimize-chain` to automatically select a complete implementation of a
supported function under one fixed external layout contract. Existing function
annotations are honored; open dimensions are closed without adding axes, and
unconstrained arguments/results are replicated. Function argument and return
order are preserved, including shared/unused arguments and repeated returns.
The first version requires one single-block `@main`, complete region coverage,
and one intermediate tensor between each pair of neighboring regions.

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/chain_3.mlir \
  --optimize-chain --output-dir=/tmp/joint_chain \
  2> /tmp/selection_report.txt
```

Chain mode writes `selected.mlir` (the chosen StableHLO + Shardy path) and
`xla_input.mlir` (XLA-ready StableHLO) to `--output-dir`; stdout lists their
paths and stderr contains the explanation. Use `--dump-dir` separately for
candidate snapshots and `selected_plan.txt`. Selection uses one search with
automatic dominance pruning of regions and composed prefixes. `--max-chain-transitions=65536` limits each DP layer
and reports truncation. See [REGION_OPTIMIZER.md](../misc/REGION_OPTIMIZER.md#optimize-a-complete-linear-chain)
and the recorded [chain experiment](../misc/CHAIN_EXPERIMENT.md).

Use `--optimize-dag` for residual connections, fan-out, joins, independent
branches, and multiple region outputs. It retains a layout for every live
produced tensor while processing regions in their original order. Each consumer
gets its own adapters; later consumers retain access to the producer's original
value and layout. Function arguments use the same fixed external contract.

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/residual_block.mlir \
  --optimize-dag --output-dir=/tmp/joint_dag \
  2> /tmp/selection_report.txt
```

This writes the same two final artifacts as chain mode.
DAG selection uses the same automatic dominance pruning across the full live cut. `--max-live-values=4` rejects wider live cuts.
`--max-dag-states=4096` and `--max-dag-transitions=65536` bound each search
layer and report truncation. See the [DAG search model](../misc/REGION_OPTIMIZER.md#optimize-residuals-and-branches)
and the recorded [DAG experiment](../misc/DAG_EXPERIMENT.md).

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/scaling_dot.mlir \
  --dump-dir=/tmp/joint_shard_regions
```

`run_shardy` runs the raw annotated pipeline. See
[REGION_OPTIMIZER.md](../misc/REGION_OPTIMIZER.md)
for boundary policies, cost parameters, budgets and output artifacts.

To inspect rewriting without running propagation:

```sh
bazel-bin/research/joint_shard/tools/parse_stablehlo \
  research/joint_shard/testdata/add_mul.mlir
```

Add `--print-egraph` to print every e-class and its alternative nodes to stderr
after rewriting, before extraction. Child references and roots use canonical
`eN` IDs. The header records whether rewriting saturated or stopped at a limit:

```sh
bazel-bin/research/joint_shard/tools/parse_stablehlo \
  research/joint_shard/testdata/megatron_layer/01.before_propagation.mlir \
  --print-egraph > /tmp/rewritten.mlir 2> /tmp/egraph.txt
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

`00_input` is the module entering Shardy. `run_shardy` prints the final module to stdout and snapshot cost diagnostics to
stderr. The optimizer stores snapshots separately for every candidate/boundary
evaluation. Full explicit-reshard insertion can also insert reduction collectives;
zero reshards does not imply zero communication.

The annotated matmul samples use `data=2, model=2`:

| File | LHS dimensions | RHS dimensions | Result dimensions | Expected communication |
|---|---|---|---|---|
| `dot_general.mlir` | data, unsharded | unsharded, model | data, model | none |
| `dot_gather.mlir` | data, unsharded | unsharded, model | data, unsharded | all-gather |
| `dot_contracting.mlir` | data, model | model, unsharded | data, unsharded | all-reduce |

For other input files with `bazel run`, pass an absolute path because Bazel
changes the working directory to the target's runfiles directory.

Selected output uses global tensor shapes and Shardy collectives
(`sdy.all_gather`, `sdy.all_reduce`, etc.). For chain/DAG selection, this
optimizer also exports XLA-ready `xla_input.mlir` from the exact same chosen
path. The separate [Python runner](../joint_shard_executor/README.md) consumes
that artifact using prebuilt XLA/PJRT, without rerunning propagation or search.
