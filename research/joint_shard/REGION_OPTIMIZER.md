# Joint rewrite–sharding region prototype

`summarize_regions` implements:

```text
split → saturate once → bounded candidates → candidate × boundary Shardy runs
      → best per exact boundary → directed reshard-dominance pruning
```

The input module is never rewritten by this command. Each candidate is exported
as a standalone `@main` function and evaluated on fresh clones. The result is a
frontier for each region, ready for a future composition phase. It is not a
selected whole-program implementation or device executable.

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
| `search/region_summary` | Best-per-boundary selection, dominance witnesses and reporting |
| `search/region_optimizer` | Orchestration and artifacts |

`parse_stablehlo` remains a bridge/rewrite inspection tool. Its mutation path
uses the same saturation and shared-root export helpers as the optimizer.
`run_shardy` only inspects an already annotated input through propagation,
reshard insertion and collective conversion. The previous whole-module
`--round-trip`/`--candidates` comparison and lexicographic winner selection were
removed.

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
calibration. The older global payload/work counters remain snapshot diagnostics,
not optimizer selection criteria.

For each exact boundary, the cheapest known candidate wins. Equal costs prefer
the original (candidate 0), then the smaller candidate ID. A known estimate can
replace an unknown estimate; an unknown estimate cannot displace a known one.
Unknown plans remain inspectable but cannot justify dominance.

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
Unsupported or unknown adapters never justify deletion.

## Artifacts and scope

With `--dump-dir=PATH`, each region receives:

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
  frontier_0.mlir
  ...
```

Frontier artifact indices follow the table order in `summary.txt`. Use a fresh
dump directory for each experiment; the tool does not delete previous artifacts.

This version does not compose summaries, apply a global winner, merge regions,
re-saturate seams, cluster candidates, infer tensor roles, or do sharding-aware
e-graph extraction. Its frontier is relative to the sampled candidate/layout
sets, Shardy propagation and the configured cost model. The next architectural
extension is composition over the shared typed boundary interfaces.
