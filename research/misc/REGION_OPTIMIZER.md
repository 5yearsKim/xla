# Joint rewrite and sharding optimizer

The optimizer searches rewritten StableHLO regions using input layout assignments.
Each candidate runs through Shardy once per assignment. Shardy determines
unconstrained output layouts. The resulting plans are composed with explicit,
costed adapters and pruned by their external boundary contracts.

```text
StableHLO → regions → e-graph saturation → unique rewrite candidates
  → input assignments → Shardy propagation/collectives → inferred boundary plans
  → best per boundary → reshard dominance → composition and prefix pruning
  → selected execution plan → verified StableHLO and XLA-ready export
```

## Run the prototype

From the XLA workspace root:

```sh
bazel build --config=joint_shard //research/joint_shard/tools:summarize_regions
bazel test --config=joint_shard //research/joint_shard:tests
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/scaling_dot.mlir \
  --dump-dir=/tmp/joint_regions
```

`--config=joint_shard` selects the project's C++20 configuration. The tools accept
options before or after the input path. `--help` lists current options.

## Component boundaries

| Component | Responsibility |
|---|---|
| `transforms/regionizer` | Typed SSA interfaces and bounded supported regions |
| `transforms/region_candidates` | One saturation and unique profile candidates per region |
| `sharding/boundary_state` | Mesh selection, input enumeration and required output constraints |
| `sharding/region_evaluator` | Propagation, inferred boundaries and costed collective artifacts |
| `search/region_summary` | Best candidate per boundary and shared dominance comparison |
| `search/resolved_plan` | Boundary replacement recipes using surviving region implementations |
| `search/pair_composer` | Adjacent composition and dominance on outer boundaries |
| `search/chain_optimizer` | Prefix search for linear chains |
| `search/dag_optimizer` | Prefix search across full live-value cuts |
| `search/execution_plan` | Region, adapter and nested composite reconstruction |
| `sharding/plan_materializer` | Artifact inlining and verification of layouts and costs |
| `export/selected_export` | XLA-ready export of the selected program |
| `reporting/` | Reports and optional artifact writing |

Search does not mutate the source module or write files. The tools connect
synchronous observers to artifact writers. See
[TensorLang](../joint_shard/TENSORLANG.md) for rewrite semantics and numerical
permissions.

## Regions and extraction

Supported operations are grouped into regions. Rule blockers such as exponential
and tanh become singleton regions, so their work remains costed. Unsupported and
annotated operations remain preserved boundaries. Whole-function selection
requires complete supported region coverage in a single-block `@main`.

`--max-region-ops=128` controls splitting; cuts prefer fewer crossing SSA values
and avoid breaking protected rewrite patterns. Protected regions may exceed the
operation budget and are reported as oversized.

Each region is saturated once. Original, compute, depth and memory extraction
profiles supply unique candidates, with `--max-candidates=32` as a cap. Candidate
extraction and saturation budgets are reported independently of sharding search.

## Input layouts and inferred outputs

An existing partitioning mesh is authoritative. Use `--mesh=NAME` if multiple
meshes are available. Mesh-free region evaluation uses `@joint_mesh` with
`data=2,model=2` in temporary wrappers. Existing `mhlo.num_partitions` must agree
with the selected mesh.

Canonical input choices are:

- R: replicated, with all tensor dimensions closed and unsharded.
- DP: the data axis on the configured data dimension.
- TP: the model axis on the configured model dimension.

Defaults are axes `data` and `model`, data dimension 0 and model dimension -1.
These are explicit experiment policies. Scalars only have R. Invalid dimensions,
missing axes and nondivisible shapes omit choices. Boundary shapes must be static.
Existing combined and sub-axis input annotations are retained outside this vocabulary.

```sh
bazel-bin/research/joint_shard/tools/summarize_regions INPUT.mlir \
  --data-axis=data --model-axis=model --data-dim=0 --model-dim=-1 \
  --input-policy=2:none:1 --max-input-states=256
```

Input policies index each region's deterministic interface. The example disables
DP for input port 2 and maps its TP choice to dimension 1.

`--max-input-states` bounds only the Cartesian product of input choices. A region
with no inputs has one empty assignment. Enumeration is deterministic and reports
truncation. With three layouts per tensor, an unconstrained region with `m` inputs
requires at most `3^m` assignments before the cap, independently of its output count.

Outputs are never enumerated. Each candidate/input assignment runs Shardy once,
and the evaluator records the actual finalized output layouts. Inferred outputs
can use combined axes or legal sub-axes. Missing post-lowering output attributes
represent replication and become explicit result contracts in stored artifacts.

Output value annotations, function result annotations and applicable dangling
constraints remain requirements. Open dimensions permit propagation; closed
layouts remain fixed. Used sharding constraints remain use-scoped preserved
boundaries. In whole-function selection, external annotations are closed without
adding axes and unannotated arguments/results use replicated contracts.

## Cost and pruning

Costs are illustrative microsecond estimates from lowered collective IR. Compute
and communication are summed sequentially. Unknown or infeasible conversions
cannot justify dominance.

For identical complete input/output boundaries, retain the cheapest candidate.
Tie-breaking is deterministic. Different boundaries are compared using directed
reshards. Plan Q replaces plan P when:

```text
cost(Q)
+ sum(reshard(P input → Q input))
+ sum(reshard(Q output → P output))
≤ cost(P)
```

All boundary ports participate. A replacement preserves the removed boundary's
callable interface by wrapping Q in input and output adapters. Every dominance
witness refers directly to a survivor; equal-cost plans cannot remove one another.
Dominated region implementations release their lowered IR, while small boundary
entries and replacement IDs remain available. Adapter lowering is cached by tensor
type and directed source/target layouts.

This rule also applies after composing two regions and after extending a function
prefix. Different boundary contracts still require adapter bookkeeping; pruning
reduces stored implementations while preserving their interfaces. It does not
assume that a direct adapter is cheaper than an adapter sequence through a recipe.

## Compose two regions

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/pair_scaling_dot.mlir \
  --compose-regions=0,1 --dump-dir=/tmp/joint_pair
```

A must have one output consumed by B, with no additional live-out use. Boundaries
use SSA identity, shared inputs are deduplicated, and shared input layouts must
agree. B may have additional inputs and multiple outputs.

Composition costs include both resolved child implementations and the directed
intermediate adapter. It keeps one winner per outer boundary and applies reshard
dominance to those winners. Surviving combined implementations are materialized;
removed contracts retain recipes that can be materialized on demand.
`--max-pair-evaluations=65536` bounds attempted child combinations.

## Optimize a complete linear chain

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/chain_3.mlir \
  --optimize-chain --output-dir=/tmp/joint_chain --dump-dir=/tmp/joint_chain_dump
```

The function must have one intermediate tensor between consecutive regions.
External arguments have a fixed contract. Each prefix cost includes previous
work, the incoming seam conversion and the next region implementation.

Equal outgoing layouts keep one cheapest prefix. Dominance compares different
outgoing layouts, recording output adapters around the surviving prefix. Recipes
preserve removed layouts for subsequent consumers. Prefix reconstruction emits
the selected region leaves, seam adapters and any prefix output adapters.
`--max-chain-transitions=65536` bounds attempts per layer.

## Optimize residuals and branches

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/residual_block.mlir \
  --optimize-dag --output-dir=/tmp/joint_dag --dump-dir=/tmp/joint_dag_dump
```

DAG selection processes regions in source order and records a layout for every
live produced tensor. Consumer input adapters are local to that consumer. Later
consumers retain access to the producer value. Outputs introduce new live values;
values leave the cut only when no later consumer or return needs them.

Same-state merging and dominance compare the entire live-layout tuple. Prefix
replacement adapters are charged and emitted for every differing live value.
Fixed final-result contracts remain pinned, including early and repeated returns.

`--max-live-values=4` rejects wider cuts. `--max-dag-states=4096` limits active
boundary states, and `--max-dag-transitions=65536` limits attempted transitions per
layer. Capped state selection retains reconstruction dependencies even when those
dependencies are no longer active. All truncation is reported.

## Artifacts and validation

Whole-function selection runs one search with automatic pruning. It writes
`selected.mlir` and `xla_input.mlir` to `--output-dir`. The former is global
StableHLO with Shardy collectives; the latter converts the selected implementation
to local code inside a manual wrapper suitable for XLA. `--dump-dir` additionally
writes prepared candidates, stage snapshots, surviving region/pair implementations,
summary reports and `selected_plan.txt`.

Reports include input assignments, Shardy evaluations, inferred boundary winners,
surviving implementations, replacement recipes, prefix dominance and search caps.

Materialization verifies types, exact layouts and recomputed compute/communication
costs. Tests check input enumeration, inferred combined/replicated outputs, fixed
constraints, directed dominance, executable recipes, chains, residuals, fan-out,
multiple/repeated returns and numerical fixture equivalence. Test-only small-plan
enumeration checks search costs independently. Numerical rewriting follows the
configured strict or explicitly relaxed policy.
