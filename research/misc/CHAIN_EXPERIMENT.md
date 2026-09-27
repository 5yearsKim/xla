# Complete-chain experiment

Recorded on 2026-09-27 using the C++ optimizer, relaxed rewriting, mesh
`data=2, model=2`, and fixed replicated function arguments/results. The fixtures
compute scalar-scaled matmul, tanh, and additional matmul/tanh stages; the same
weight argument is reused in the five-region function. All modes use the same
external contract and evaluated internal layout choices.

Costs are **illustrative estimates in microseconds**, using the default model:
compute work/us = 1,000,000; bandwidth bytes/us = 50,000; collective latency = 5 us.
Search time is one observed CPU run, including plan resolution and winner
reconstruction, after region evaluation/pruning populated the adapter cache.
These figures are not hardware runtime measurements or stable timing benchmarks.
States count retained prefixes over all layers; transitions count attempted
prefix/region combinations.

| Mode | Compute(us) | Communication(us) | Total(us) | Search(us) | States | Transitions | Truncated |
|---|---:|---:|---:|---:|---:|---:|---|
| **3 regions** | | | | | | | |
| original + DP | 0.001664 | 0 | 0.001664 | 72.794 | 7 | 39 | 0 |
| joint greedy | 0.000784 | 5.00128 | 5.002064 | 21.921 | 3 | 15 | 0 |
| joint exact DP | 0.001568 | 0 | 0.001568 | 64.449 | 7 | 39 | 0 |
| joint resolved DP | 0.001568 | 0 | 0.001568 | 70.487 | 7 | 39 | 0 |
| **4 regions** | | | | | | | |
| original + DP | 0.00192 | 0 | 0.00192 | 131.912 | 10 | 66 | 0 |
| joint greedy | 0.000912 | 5.00128 | 5.002192 | 40.088 | 4 | 24 | 0 |
| joint exact DP | 0.001824 | 0 | 0.001824 | 114.343 | 10 | 66 | 0 |
| joint resolved DP | 0.001824 | 0 | 0.001824 | 126.623 | 10 | 66 | 0 |
| **5 regions** | | | | | | | |
| original + DP | 0.002176 | 0 | 0.002176 | 182.622 | 13 | 93 | 0 |
| joint greedy | 0.00104 | 5.00128 | 5.00232 | 53.209 | 5 | 33 | 0 |
| joint exact DP | 0.00208 | 0 | 0.00208 | 167.624 | 13 | 93 | 0 |
| joint resolved DP | 0.00208 | 0 | 0.00208 | 186.976 | 13 | 93 | 0 |

All runs completed without boundary truncation, skipped candidate profiles,
saturation limits, or DAG extraction limits. Caps were 32 candidates, 256
boundary states per region, and 65,536 transitions per chain layer.

Greedy selection saves local compute but introduces layout conversion
communication. Complete-chain DP chooses a compatible sequence with a smaller
sum of region and adapter costs. It also improves the original-only DP baseline
by moving scalar multiplication to the smaller first-dot output. Exact and
resolved DP agree on these fixtures at the default cost parameters; resolved
search may improve other cases by using retained implementations with wrappers.

Reproduce each row from the workspace root (replace 3 with 4 or 5):

```sh
bazel build --config=joint_shard //research/joint_shard/tools:summarize_regions
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/chain_3.mlir \
  --optimize-chain --numerical-policy=relaxed --dump-dir=/tmp/joint_chain \
  > /tmp/selected.mlir 2> /tmp/selection_report.txt
cat /tmp/joint_chain/comparison.txt
```

`selected.mlir` contains one verified `@main`; its exact boundary and recomputed
compute/communication cost are checked against the selected plan. The CLI also
successfully emits a complete five-region winner with `--chain-search=resolved`
and with `--max-chain-transitions=1`; the latter reports search truncation.

The regression tests independently enumerate three-to-five-region choices,
materialize original and greedy reference winners, and simulate selected rewrites
and emitted collectives. Numerical tests use seed 42, bounded f32 inputs and an
alternating-sign cancellation case, with `atol=2e-5, rtol=2e-4`. A second cost
setting (`--compute-work-per-us=1 --collective-latency-us=0.01`) exercises selected
modules containing communication. Canonical adapters and a contracting dot's
all-reduce are simulated explicitly; removing that reduction causes failure.

```sh
bazel test --config=joint_shard //research/joint_shard:tests --test_output=errors
```

Device execution and calibrated runtime comparisons remain the next experiment.
