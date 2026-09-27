# Joint rewrite–sharding region prototype

`summarize_regions` implements:

```text
split → saturate once → bounded candidates → candidate × boundary Shardy runs
      → preserved exact table + directed reshard-dominance frontier
      → optional pair search → verified combined MLIR per external boundary
      → optional chain/DAG DP → one verified implementation under a fixed contract
```

The input module is never rewritten by this command. Each candidate is exported
as a standalone `@main` function and evaluated on fresh clones. The result is a
complete boundary table and frontier for each region. Optional two-region
composition selects and materializes one implementation per external boundary.
Chain mode selects a complete implementation of a supported linear function.
The emitted collective IR is not a device executable.

## Run the prototype

From the workspace root:

```sh
bazel build --config=joint_shard //research/joint_shard/tools:summarize_regions
bazel test --config=joint_shard //research/joint_shard:tests --test_output=errors
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/scaling_dot.mlir \
  --dump-dir=/tmp/joint_shard_regions
```

The scaling fixture has three operations, three inputs (one scalar), one output,
and 27 canonical boundary combinations. Relaxed algebra exposes moving a scalar
multiply from the larger dot input to the smaller output. The report lists the
original and unique extracted alternatives, evaluation counts, the best plans,
the surviving frontier, and the reshard witnesses for removed states.

`--numerical-policy=relaxed` is the default for the driver and rewrite tools.
It permits floating algebra under the existing TensorLang guards and can change
floating-point results. Use `--numerical-policy=strict` to preserve the existing
strict evaluation contract. Low-level property and rule APIs still accept an
explicit numerical policy.

## Component boundaries

| Component | Responsibility |
|---|---|
| `transforms/regionizer` | Immutable contiguous ranges, SSA interfaces, protected cuts |
| `transforms/rewrite_options` | Shared numerical policy, rules and search budgets |
| `transforms/region_candidates` | Rule compilation, saturation, original capture and profile extraction |
| `sharding/boundary_state` | Native exact layouts, mesh selection and bounded enumeration |
| `sharding/region_evaluator` | Candidate wrappers, isolated Shardy runs and cached reshard adapters |
| `sharding/cost_model` | One additive microsecond model for candidates and adapters |
| `search/region_interface` | Stable SSA value IDs and typed region/pair port mappings |
| `search/region_summary` | Preserved best-per-boundary table, frontier IDs and dominance witnesses |
| `search/pair_composer` | Joint pair search, dominance resolution and materialization orchestration |
| `search/resolved_plan` | Shared exact-boundary resolution through the retained frontier |
| `search/execution_plan` | Stable arena references to region, adapter and composite nodes |
| `search/chain_optimizer` | Prefix dynamic programming, greedy comparison and reconstruction |
| `search/dag_optimizer` | Live-value layout DP for residuals, fan-out and joins in source order |
| `sharding/plan_materializer` | Mesh-checked inlining of lowered artifacts and cost verification |
| `search/region_optimizer` | Orchestration, structured results and synchronous observers |
| `reporting/reports` | Human-readable report formatting |
| `reporting/artifact_writer` | Candidate, snapshot, region and pair artifact output |

`parse_stablehlo` remains a bridge/rewrite inspection tool. Its mutation path
uses the same saturation and shared-root export helpers as the optimizer.
`run_shardy` only inspects an already annotated input through propagation,
reshard insertion and collective conversion.

## Regions and extraction

Only single-block functions are supported. Import rejection preserves
unsupported operations, annotated operations, constraints and unknown metadata
as hard boundaries. They are reported and are not included in optimized region
costs. Original IR, including unused constraints and use-scoped constraints,
remains intact.

Supported `stablehlo.exponential` and `stablehlo.tanh` operations are singleton
regions by default. `--rule-blocker=none` disables those cuts;
`--rule-blocker=stablehlo.OP` appends another supported singleton blocker.

`--max-region-ops=128` controls recursive splitting. Each legal cut is scored by
distinct SSA values crossing it, then distance from the midpoint. SSA spans for
transpose→dot, reshape→reshape and broadcast→multiply→dot are protected,
including nonadjacent producers/consumers. If all cuts are protected, the region
is kept oversized and reported. No tensor-byte or sharding-combination score is
used for cuts.

Boundary inputs are ordered by first operand encounter. Outputs are ordered by
defining operation/result order and include values consumed by function returns.
Those positions are preserved throughout import, extraction, evaluation and
summary lookup. Every external input is bound before recursive import.

The original expression is recorded from the source SSA DAG before saturation.
Candidates share one saturated graph and may contain multiple ordered output
roots. Extraction tries compute tree, compute DAG, depth and memory profiles.
Exact structural duplicates are removed after canonicalizing reachable nodes
and shared computations. `--max-candidates=32` is a cap including the original;
the current profile milestone yields at most five distinct candidates. It does
not enumerate k-best expressions. `--extraction=depth|memory|compute` prioritizes
that profile when the candidate cap is small. Existing saturation and DAG
budgets are available through the shared rewrite CLI flags.

## Boundary layouts

An existing partitioning mesh is authoritative. Select an ambiguous input with
`--mesh=NAME`. A mesh-free input uses `@joint_mesh` with `data=2,model=2` in the
temporary evaluation modules. A present `mhlo.num_partitions` must match.
The source module is not given a synthetic mesh.

The canonical choices are:

- R: all tensor dimensions closed and unsharded.
- DP: shard the configured data dimension on the configured data axis.
- TP: shard the configured model dimension on the configured model axis.

The default mapping is data dimension 0 and model dimension -1 (the last
dimension), with axes named `data` and `model`. These are explicit experiment
policies, not inferred tensor roles. Scalars only have R. Missing mesh axes,
unavailable tensor dimensions and nondivisible shapes omit that canonical
choice. Dynamic boundary shapes are rejected in this version.

Override the common mapping:

```sh
bazel-bin/research/joint_shard/tools/summarize_regions INPUT.mlir \
  --data-axis=data --model-axis=model --data-dim=0 --model-dim=-1
```

Override individual ports, indexed within each region's deterministic interface:

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/scaling_dot.mlir \
  --input-policy=2:none:1 --output-policy=0:0:1
```

Here input 2 (the weight) has no DP choice and maps TP to dimension 1.
The C++ `LayoutPolicy` exposes the same per-input/per-output mappings.

Native Shardy attributes are wrapped, not translated to an R/DP/TP enum. Fixed
combined layouts and legal sub-axes outside the canonical vocabulary can be
retained. Existing input annotations, function return annotations and dangling
constraints restrict enumeration. Open annotations permit compatible exact
refinements; their existing axes/closed dimensions and replication restrictions
are respected. Used sharding constraints remain separate use-scoped boundaries.
Priorities do not distinguish exact execution layouts once dimensions are closed.

`--max-boundary-states=256` bounds the Cartesian product independently of the
candidate cap. Enumeration is deterministic, and truncation is reported. Each
state specifies a layout for every input and output. The evaluator validates
the types/layouts and checks the function boundary attributes after Shardy.

## Cost and pruning

All costs use estimated microseconds with a sequential sum and no overlap:

```text
compute = per-device work / compute_work_per_us
communication = sum(latency_us + estimated_transferred_bytes / bandwidth_bytes_per_us)
total = compute + communication
```

Illustrative defaults are 1,000,000 work units/µs, 50,000 bytes/µs and 5 µs
latency. Override with `--compute-work-per-us`, `--bandwidth-bytes-per-us` and
`--collective-latency-us`. Rates must be finite and positive; latency can be zero.

Compute estimates use static tensor shapes and propagated partitions. Dot work
accounts for sharding of both output and contracting dimensions; reductions use
the local input element count. Constants, reshapes, broadcasts and local slicing
are treated as free; transpose uses an element-movement work proxy. Scalar
reducer bodies are not charged separately. Internal values left without a
sharding by the completed propagation pipeline are replicated for costing.

Communication uses local payloads: all-gather and reduce-scatter use the payload
difference, all-reduce uses a ring-volume approximation, all-to-all uses its
axis-group volume, and collective-permute uses the input payload. This is still
a simple model without topology, overlap, scheduling, memory pressure or device
calibration. The optional global payload/work counters in `ModuleStatistics`
remain diagnostics, not optimizer selection criteria.

For each exact boundary, the cheapest known candidate wins. Selection happens
incrementally during evaluation, so losing lowered artifacts are released
immediately. `finalizePlans` orders the resulting table and assigns plan IDs.
Equal costs prefer the original (candidate 0), then the smaller candidate ID. A known estimate can
replace an unknown estimate; an unknown estimate cannot displace a known one.
Unknown plans remain inspectable but cannot justify dominance or win pair
selection. After best-per-boundary selection, plans receive stable `P` IDs.
Pruning retains the full exact table; the frontier references surviving IDs.
Every dominance witness stores both the removed and replacement IDs.

For dominance, input adapters convert the removed boundary to the replacement
boundary; output adapters convert the replacement back to the removed boundary.
The oracle builds an explicit one-tensor reshard, lowers it with the same Shardy
collective pass, and costs it with the same model. Its cache is scoped to the
mesh/model and keyed by type, source and target layouts. R→DP may only slice
locally; DP→R may gather, so adapter costs are directional.

Pruning compares each state against already retained states ordered by cost and
deterministic layout key. This conservative O(n²) procedure never mutually
deletes tied states, and every removal has a direct surviving replacement.
It can retain redundant equal-cost states whose replacement appears later.
Unsupported or unknown adapters never justify deletion. The reshard oracle
returns feasibility, cost, type, direction and the lowered adapter module from
the same lowering. Identity adapters have no operations and zero cost.

## Artifacts and scope

The library captures only final evaluation IR by default. The CLI connects an
`ArtifactWriter` through `OptimizationObserver` to capture all stages when
`--dump-dir=PATH` is supplied. Each region receives:

```text
region_0/
  candidate_0.mlir
  candidate_1.mlir
  boundary_0/candidate_0/00_input.mlir
  boundary_0/candidate_0/01_propagated.mlir
  boundary_0/candidate_0/02_explicit_reshards.mlir
  boundary_0/candidate_0/03_collectives.mlir
  ...
  summary.txt
  exact_P0.mlir
  ...
  frontier_0.mlir
  ...
```

Frontier artifact indices follow the table order in `summary.txt`. Use a fresh
dump directory for each experiment; the tool does not delete previous artifacts.

## Compose two regions

The default exact mode searches the complete child boundary tables:

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/pair_scaling_dot.mlir \
  --compose-regions=0,1 \
  --dump-dir=/tmp/joint_shard_pair
```

Region 0 contains the scaling and dot; region 1 contains tanh. Their 27 and 9
exact plans produce 243 evaluated pairs and 27 external boundary winners.
`pair_0_1/summary.txt` explains the external SSA values, input bindings, selected
child plans/rewrites, intermediate layouts and component costs. For each row:

```text
pair_0_1/
  summary.txt
  selected_0.mlir          # one mesh and combined @main
  selected_0_adapter.mlir  # intermediate adapter, or an identity comment
  ...
```

`selected_0` is the winner for its displayed external boundary, rather than a
winner across all outer layouts. `R` means replicated; `model:d1` means the
model axis shards tensor dimension 1. `P` identifies a child exact plan and
`R0`, `R1`, etc. identify rewrite candidates. Read `A + adapter + B = total`
with each child including any dominance-resolution wrappers. Compute and
communication are also shown separately for the combined plan.

A and B must be adjacent operation ranges in the same single-block function.
A must have exactly one output, consumed by B, and every use of that value must
be inside B. Fan-out and an intermediate returned from the original function
are rejected. Composition cannot cross a preserved unsupported/fixed operation.
B may have additional external inputs and multiple outputs. Shared external
inputs are deduplicated by SSA identity and must have identical requested
layouts in both children. Equal tensor types alone never establish identity.
Value IDs are deterministic for the current module; they are not identities
across source edits.

The external input order is A's inputs followed by B's remaining inputs in
first-use order. The intermediate disappears from the external interface.
Outputs retain B's order. Per-port layout policies are applied to each child
before composition; they are not reapplied to the new external port indices.
The selected mesh, numerical policy and cost model are shared by both children.

For every pair, the composer projects the outer boundary and costs the directed
A-output → B-input adapter. It retains the smallest known sum for each outer
boundary; ties use requested child plan IDs. `--max-pair-evaluations=65536`
bounds this Cartesian search. Caps, shared-layout rejections and unknown costs
are reported. A capped run searches a deterministic prefix and can miss better
plans and outer states; child candidate and boundary caps still apply.

Materialization inlines the selected child collective modules and the actual
adapter artifact using SSA mappings, into one `@main` with exact outer attrs.
It preserves closed internal layouts and treats absent internal post-propagation
sharding as replication. It does not rerun propagation. Every emitted module
must pass MLIR verification, layout/type checks and recomputed compute and
communication cost checks against the selected sum (relative tolerance 1e-9).
The source module is unchanged.

### Compose through the pruned frontier

Add `--compose-pruned` to resolve each requested exact child boundary to its
retained implementation. Input adapters run requested → retained; output
adapters run retained → requested. These executable wrappers preserve every
callable exact boundary, including the removed states. The report distinguishes
requested and implemented plan IDs, and charges all wrapper costs.

This mode still enumerates the original exact boundary space, so it is a
correctness milestone rather than a faster search algorithm. With the same
oracle and uncapped search, its cost for each outer boundary is no greater than
the exact-mode reference. It can be smaller because adapter-wrapped retained
plans add implementations to the original exact table. Unknown adapters cannot
justify a replacement. Every witness resolves directly to a surviving plan.

### A rewrite whose winner depends on layout

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/pair_boundary_rewrite.mlir \
  --input-policy=1:none:1 --input-policy=2:none:0 \
  --compose-regions=0,1 --dump-dir=/tmp/joint_shard_boundary_rewrite
```

The fixture computes `dot(a,b) + dot(a,c)` followed by tanh. `a` is 2×256;
`b` and `c` are 256×256. With fully replicated inputs/output, factoring to
`dot(a,b+c)` wins (candidate R1). With `a` replicated, `b` sharded on model
dimension 1, `c` on model dimension 0 and the dot output on model dimension 1,
the original two-dot form wins (R0): factoring must communicate a large RHS
rather than small dot outputs. The regression compares both real Shardy
lowerings and costs; this result is not a mocked scoring example.

The tests also include a synthetic example where greedy child selection costs
7 while joint selection costs 4, an independent exhaustive real-cost reference,
all nine canonical adapter directions, resolved-frontier cost preservation,
shared input identity/layout checks and multiple-output materialization.

This version does not mutate the source function, merge or re-saturate region
seams, cluster candidates, infer tensor roles, or do sharding-aware e-graph
extraction. Its results are relative to the sampled candidates/layouts, Shardy
propagation and the configured illustrative cost model. Device execution,
region scheduling and control flow remain future milestones.

## Optimize a complete linear chain

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/chain_5.mlir \
  --optimize-chain --numerical-policy=relaxed \
  --output-dir=/tmp/joint_chain --dump-dir=/tmp/joint_chain_debug \
  2> /tmp/selection_report.txt
```

This mode requires exactly one function, a nonempty single-block `@main`, whose
operations are completely covered by regions. Each non-final region has one
output consumed only by the immediate next region. Additional inputs must be
function arguments. The final region may have multiple outputs; return ordering
and repeated returns follow the source signature. Unsupported/fixed operations,
fan-out across regions, residual edges, early live-outs and partial function
coverage are rejected with an explanation. Multiple uses inside the next region
are allowed. Shared and unused function arguments retain their signature slots.

The external contract is chosen before candidate/boundary evaluation. Fixed
annotations are preserved as exact execution layouts. Open annotations are
closed while retaining their existing axes and replication restrictions; no new
axes are added. Unconstrained ports use replication. Repeated return values
must have consistent exact contracts. Native combined and sub-axis layouts
remain available through existing layout validation. Region port layout policies
continue to control internal boundary choices; they do not override this function
contract. External ports are fixed before Cartesian enumeration, so a boundary
cap cannot filter out a required contract merely because it appears late.

For each outgoing layout, DP retains the cheapest prefix under:

```text
prefix cost + directed seam adapter cost + next resolved region cost
```

All other region inputs must match the fixed function argument layouts. With one
live intermediate per cut, the outgoing layout is sufficient to describe the
future cost; earlier choices need not remain separate search states. Ties compare
the sequence of requested local plan IDs. Predecessors recover the complete
winning path. `--max-chain-transitions=N` bounds transitions **per layer**; a
truncated layer still supplies its retained prefixes to subsequent layers. If no
complete feasible prefix remains, the command fails rather than emitting a
partial implementation. A capped result may miss the global sampled-table
minimum; every layer cap, candidate profile skip, boundary cap, saturation limit
and DAG extraction limit is reported.

The command compares four modes with the same contract, mesh, cost model,
candidate/boundary evaluation set and per-layer transition cap:

| Mode | Implementations | Selection |
|---|---|---|
| Original + DP | Original source expression only | Cheapest complete chain |
| Joint greedy | Best rewrite per exact boundary | Cheapest next step including its incoming adapter |
| Joint exact DP | Best rewrite per exact boundary | Cheapest complete chain |
| Joint resolved DP | Retained implementations with exact-boundary wrappers | Cheapest complete chain |

The original-only table is retained during evaluation even when a rewrite wins
at the same boundary, so the baseline does not accidentally use rewritten plans.
`--chain-search=exact|resolved` chooses the emitted winner (default exact).
Resolved mode uses the same callable exact boundaries; it can improve their
implementations with input/output adapters. As in pair composition, its uncapped
cost is no greater than the uncapped exact reference, not necessarily equal.
It is not yet a reduction of the exact search space.

Execution plans form an arena of region references, executable adapters and
ordered composites that can reference earlier prefix composites. Each node has
stable value IDs, a typed interface, exact boundary and component cost. Region
references contain both the summary index and its local implementation ID.
Composite costs sum direct children; wrappers and seam adapters are each charged
once. Composites have local value bindings so adapting a shared function argument
for one region does not alter the argument supplied to another region.

Only the selected chain is materialized by the CLI. It inlines the chosen
collective artifacts and all nonidentity adapters into one `@main` without
rerunning propagation. MLIR verification, input/output layout checks and
recomputed compute/communication costs remain mandatory. The reconstructed plan
cost must also agree with the selected prefix cost. Chain mode writes
`selected.mlir` and `xla_input.mlir` to `--output-dir`, lists their paths on
stdout, and reports to stderr; `--dump-dir` writes `selected_plan.txt`,
`comparison.txt`, and region diagnostics.
Library search remains free of file writes. Original baseline execution nodes
refer to `OptimizationReport.original_regions`; joint modes refer to `regions`.

### Numerical validation policy

`chain_optimization_test` covers three, four and five regions, independent
Cartesian enumeration, real and synthetic greedy failures, shared argument and
return ordering, emitted adapter operation counts, component costs, capped
search, exact/resolved selection and strict rewriting. Numerical tests compare
both selected candidate expressions before collective lowering and the emitted
collective module with the source computation.

The test-only CPU simulator stores device shards in global coordinates and
executes gathers, slices, all-to-all, canonical permutations, sum all-reduce and
sum reduce-scatter between simulated devices. It supports the fixture's f32
scalar broadcasts, ordinary rank-2 dots, add/multiply and tanh; unsupported
operators/sub-axes fail explicitly. Tests also execute every canonical adapter
direction and a contracting dot requiring all-reduce. Removing that all-reduce
causes the simulation to fail. Collectives are never stripped or treated as
universal identity operations.

For the bounded f32 fixtures, the policy is seed 42, scalar scale 0.3, five
samples with other input elements uniformly drawn from [-0.5, 0.5], and an
alternating-sign cancellation sample. Comparisons require finite outputs and
`abs(actual-reference) <= 2e-5 + 2e-4 * abs(reference)`. Strict rewrite tests use
the same numerical tolerance: distributed reductions may still change the
accumulation order. These are regression checks for supported fixtures, not a
proof for arbitrary floating-point inputs. The CLI does not execute arbitrary
input modules or measure hardware runtime.

## Optimize residuals and branches

```sh
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/residual_block.mlir \
  --optimize-dag --dag-search=exact \
  --output-dir=/tmp/joint_dag --dump-dir=/tmp/joint_dag_debug \
  2> /tmp/selection_report.txt
```

DAG mode supports residual connections, fan-out, joins, independent branches,
multiple region outputs, early returns, repeated operands and repeated returns.
It requires one nonempty single-block `@main` with complete region coverage and
supported static tensor types. It preserves source region order; it does not
schedule branches, handle control flow, or keep several persistent converted
copies of a value. Pair composition, chain optimization and DAG optimization
are mutually exclusive.

### Live values and the fixed contract

Stable value IDs identify each function argument and region output. The
interface records its unique producer, distinct consuming regions, and last
use. Function return is treated as one final use, so early-returned values stay
live until the function ends. After each region, the live cut contains produced
values used by later regions or the return, sorted by value ID. Function
arguments are always available under their fixed external layouts and do not
add search dimensions. Duplicate uses in a region share one input port.

The external layout contract follows the same annotation rules as chain mode.
Function argument constraints fix requested region input layouts. Function
result constraints fix the produced output layout of the corresponding region.
An early-returned tensor can still be converted for another consumer: its
producer layout remains pinned, while that consumer's input layout is searched.
Repeated returns must agree on their layout; a returned function argument must
have matching argument/result contracts. Unused arguments keep their slots.

### Search and reconstruction

A prefix state assigns one persistent producer layout to every live value.
Each transition chooses a region implementation and adds:

```text
prefix cost
+ sum(adapter from each input's persistent layout to its requested layout)
+ resolved region implementation cost (including its wrappers)
```

The region's outputs introduce new persistent layouts. Input adapters create
consumer-local copies and never update the persistent layout map. Once a value
has no remaining use it leaves the state. The cheapest prefix per complete live
layout assignment survives, with requested plan-ID sequences breaking ties.
Predecessors recover all selected regions and adapters. Exact search is globally
optimal over the supplied tables under this fixed-order, additive-cost model
when search caps do not truncate it. Conversion sharing, cached copies, peak
memory costs, branch scheduling and cross-region rewrites are outside this
model.

`--max-live-values=4` rejects a wider cut before evaluating candidates. It never
silently drops a live value. `--max-dag-states=4096` caps retained states per
layer; excess states are deterministically discarded in cost/tie order.
`--max-dag-transitions=65536` caps attempted transitions per region. Both caps
mark the result truncated, report the affected layers, and continue searching
for a complete plan. If no complete feasible plan survives, the command fails
without emitting selected MLIR. A truncated result is the best complete plan
found by that search, without a guarantee of the sampled-table minimum.

Original-only DP, joint greedy, joint exact DP, and joint resolved DP share the
contract, region evaluation, mesh and configured budgets. Greedy retains only
its cheapest next prefix. `--dag-search=exact|resolved` chooses the emitted
winner. Resolved search uses retained implementations with executable wrappers
at the same requested boundaries; without truncation its optimum cannot exceed
the exact-table optimum. It does not reduce the boundary search space.

Reconstruction reuses the execution-plan arena and existing materializer. Each
consumer composite encloses its adapters and chosen region; it exposes only
produced outputs. Prefix composites expose the whole live cut and reference
earlier prefixes once. This retains a residual's original value while its
consumer receives an adapted copy, and emits every producer and adapter once.
The final composite restores function return order, including repeated outputs
and pass-through arguments. Selected MLIR must verify, match the exact external
layouts, and agree with both reconstructed and recomputed component costs.

Stdout contains selected MLIR and stderr contains the explanation. With
`--dump-dir`, `selected.mlir`, `selected_plan.txt`, and `comparison.txt` include
selected implementations, wrapper costs, each input adapter, live-value layouts,
search times, transitions, retained/discarded states, peak live cut width, and
all budget truncation indicators. The report distinguishes capped search from
an uncapped optimum. Library search does not write files.

### DAG regression policy

`dag_optimization_test` independently enumerates complete combinations for the
residual, fan-out/join, independent-branch, and multiple-output fixtures. It
checks chain/DAG agreement on the three-to-five-region fixtures, greedy failure,
all search caps, invalid coverage, unknown costs, strict/resolved selection,
annotated external contracts, and artifact emission. A forced fan-out case
requires a conversion for one consumer and the original layout for another;
its emitted collectives, numerical results, operation counts and costs are
checked. Leaf operation counts also detect duplicated producers or omitted
adapters.

The tests execute selected candidate expressions before lowering and simulate
emitted collective modules against the source. They use the chain fixture CPU
simulator and the same explicit f32 policy: seed 42, five samples bounded by
[-0.5, 0.5], one alternating-sign cancellation sample, scalar scale 0.3, and
`atol=2e-5, rtol=2e-4`. Fixture simulation establishes regression coverage for
these operations and inputs; the CLI itself does not run arbitrary modules or
measure hardware performance. See [DAG_EXPERIMENT.md](DAG_EXPERIMENT.md).
