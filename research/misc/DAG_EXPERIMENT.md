# Residual and branch experiment

Recorded on 2026-09-27 with relaxed rewriting, mesh `data=2, model=2`, and
fixed replicated function arguments/results. The same evaluated layouts,
rewrites and external contract are used for all four selection modes.

Costs are illustrative estimates in microseconds, with the default model:
compute work/us = 1,000,000; bandwidth bytes/us = 50,000; collective latency = 5 us.
Search times are single observed CPU runs after region evaluation and pruning
populate the adapter cache, including resolution and reconstruction. They are
not hardware runtimes or stable timing benchmarks. States count retained
prefixes across layers; transitions count attempted prefix/region combinations.

| Fixture / mode | Compute(us) | Communication(us) | Total(us) | Search(us) | States | Transitions | Truncated |
|---|---:|---:|---:|---:|---:|---:|---|
| **residual_block** | | | | | | | |
| original + DP | 0.001696 | 0 | 0.001696 | 274.436 | 13 | 111 | 0 |
| joint greedy | 0.0008 | 5.00128 | 5.00208 | 44.638 | 3 | 21 | 0 |
| joint exact DP | 0.0016 | 0 | 0.0016 | 259.478 | 13 | 111 | 0 |
| joint resolved DP | 0.0016 | 0 | 0.0016 | 273.575 | 13 | 111 | 0 |
| **fanout_join** | | | | | | | |
| original + DP | 0.000576 | 0 | 0.000576 | 459.982 | 22 | 192 | 0 |
| joint greedy | 0.000288 | 5.00128 | 5.001568 | 56.796 | 4 | 30 | 0 |
| joint exact DP | 0.000576 | 0 | 0.000576 | 438.425 | 22 | 192 | 0 |
| joint resolved DP | 0.000576 | 0 | 0.000576 | 469.387 | 22 | 192 | 0 |
| **independent_branches** | | | | | | | |
| original + DP | 0.000544 | 0 | 0.000544 | 249.864 | 13 | 93 | 0 |
| joint greedy | 0.000272 | 5.00128 | 5.001552 | 31.517 | 3 | 15 | 0 |
| joint exact DP | 0.000544 | 0 | 0.000544 | 225.633 | 13 | 93 | 0 |
| joint resolved DP | 0.000544 | 0 | 0.000544 | 226.819 | 13 | 93 | 0 |
| **dag_multi_output** | | | | | | | |
| original + DP | 0.000352 | 0 | 0.000352 | 195.119 | 7 | 57 | 0 |
| joint greedy | 0.000192 | 5.00128 | 5.001472 | 51.596 | 3 | 21 | 0 |
| joint exact DP | 0.000352 | 0 | 0.000352 | 431.066 | 7 | 57 | 0 |
| joint resolved DP | 0.000352 | 0 | 0.000352 | 335.4 | 7 | 57 | 0 |

The fixtures cover a three-region scaled matmul residual, a four-region
fan-out/join, two independent branches followed by a join, and a three-region
function with multiple produced outputs, an early return, repeated returns and
a pass-through argument. All have a peak live cut of two produced values.
All default runs complete without state/transition/boundary truncation,
candidate skips, saturation limits or extraction limits. Default limits are
four live values, 4,096 states and 65,536 transitions per region, 32 candidates
and 256 boundary states per region.

Greedy chooses inexpensive local compute but pays for communication later.
Whole-function DP chooses layouts that avoid those conversions at these cost
parameters. The residual also improves the original-only baseline by moving
scalar multiplication to the smaller matmul output. Exact and resolved DP agree
on these fixtures; retained implementations with wrappers can improve resolved
costs in other cases. The optimizer preserves source order and one persistent
producer layout per tensor. It does not share converted copies between consumers
or optimize region scheduling.

Reproduce any fixture (replace `residual_block` with another table entry):

```sh
bazel build --config=joint_shard //research/joint_shard/tools:summarize_regions
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/residual_block.mlir \
  --optimize-dag --numerical-policy=relaxed --dump-dir=/tmp/joint_dag \
  > /tmp/selected.mlir 2> /tmp/selection_report.txt
cat /tmp/joint_dag/comparison.txt
```

Every command emits one verified `@main`. Selected boundaries, reconstructed
cost, recomputed compute/communication cost and operation occurrence counts
are checked by the materializer and regression tests. CLI checks also confirm
resolved selection, complete but explicitly truncated plans with
`--max-dag-states=1` or `--max-dag-transitions=1`, rejection of the residual
with `--max-live-values=1`, mutually exclusive optimization modes, and continued
five-region chain support. Rejected commands emit no selected MLIR.

The tests compare exact DP against independent exhaustive Cartesian enumeration
on the four fixtures, and against chain DP on the three-to-five-region fixtures.
A forced fan-out case gathers a producer value for one consumer while another
consumer uses its original sharded copy. Both cost and numerical simulation
confirm that the producer is emitted once, the adapter is charged once, and the
persistent layout is preserved.

Numerical regression tests execute selected rewrites before lowering and the
emitted collective modules against the source. They use supported f32 fixtures,
seed 42, five bounded random samples and an alternating-sign cancellation
sample, with `atol=2e-5, rtol=2e-4`. A second model with compute work/us = 1 and
collective latency = 0.01 us exercises modules that select communication. Strict
rewriting, annotated inputs/results and artifact emission are also checked.
The CLI does not execute arbitrary modules or measure device runtime.

```sh
bazel test --config=joint_shard //research/joint_shard:tests --test_output=errors
```

All 19 test targets pass, including 10 DAG regression tests. Device execution,
calibrated costs, conversion sharing and scheduling remain follow-up work.
