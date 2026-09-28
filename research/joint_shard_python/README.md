# Original ML workloads

This package lowers original JAX computations for the joint rewrite/sharding
search engine. Workload implementations live in `ml_workloads/operations/`;
algebraic rewrites belong in `joint_shard`.

From `research/joint_shard_python/`, run:

```bash
uv run python main.py --operation kernel_linear_attention --execute
```

The CLI sets the CPU device count to four before importing JAX, unless
`XLA_FLAGS` already specifies a count. Each operation uses a one-dimensional
`model` mesh over available JAX devices. Sharded dimensions must be divisible
by its size. Arrays are float32, with int32 cross-entropy labels and expert IDs.
New workloads initialize deterministically for a given shape and seed,
independent of layout.

`--operation NAME --help` shows shape, layout, seed, and numerical options.
Shape flags use hyphens, such as `--feature-rank` and `--value-dim`.
The existing default `megatron_layer` and its flags remain available:

```bash
uv run python main.py --batch 2 --sequence 4 --hidden 8 --ffn 16 --execute
```

## Workloads

Every new workload has a configuration dataclass, input dataclass with ordered
`arrays()`, and operation class implementing `BaseOperation`. Classes are exported
from `ml_workloads` and `ml_workloads.operations`. `OPERATIONS` maps CLI names to
`(operation class, configuration class)`.

| Operation | Original computation and output | Input layouts (first is default) |
|---|---|---|
| `output_projection_ce` | Complete `XW`, stable log-softmax and mean target cross-entropy; scalar | `vocabulary`, `tokens`, `replicated` |
| `context_attention` | `softmax(QKᵀ / sqrt(head_dim)) V`; `[batch, queries, value_dim]` | `sequence`, `queries`, `replicated` |
| `layernorm_linear` | Affine LayerNorm, then projection; `[tokens, output]` | `hidden`, `tokens`, `replicated` |
| `reduce_dot` | Sum over sequence, then project; `[batch, output]` | `sequence`, `hidden`, `replicated` |
| `lora` | `X(W + AB)` with dense effective weight; `[tokens, output]` | `output`, `hidden`, `tokens`, `replicated` |
| `qkv_projection` | Three separate projections; tuple `(Q, K, V)`, each `[tokens, projection]` | `output`, `hidden`, `tokens`, `replicated` |
| `gated_mlp` | Separate SwiGLU gate/value projections, product, then down projection; `[tokens, output]` | `ffn`, `hidden`, `tokens`, `replicated` |
| `kernel_linear_attention` | Form `phi(Q)phi(K)ᵀ`, multiply by V, divide by kernel row sums; `[batch, queries, value_dim]` | `sequence`, `queries`, `replicated` |
| `graph_convolution` | Dense `max((AX)W, 0)`; `[graphs, nodes, output]` | `nodes`, `independent_graphs`, `replicated` |
| `barlow_twins` | Normalize embeddings, form feature correlation, penalize diagonal/off-diagonal terms; scalar | `features`, `samples`, `replicated` |
| `moe_router` | Complete expert projection, global stable top-k, routing weights; tuple `(expert_ids, weights)`, each `[tokens, top_k]` | `experts`, `tokens`, `replicated` |
| `megatron_layer` | Existing single-head attention, residuals, and GELU MLP; `[batch, sequence, hidden]` | Existing tensor-parallel weight layouts |

Layouts shard the named input dimension and replicate other inputs unless their
matching contracting dimension must also be partitioned. Hidden sharding, for
example, partitions X's hidden dimension and matching rows of W. Sequence attention
partitions K/V and replicates Q. Independent graphs partition the leading graph
dimension of A and X and replicate W.

New operations leave intermediate and output layouts unconstrained for propagation.
Pass `out_sharding` to `lower()` or `compile()` for a fixed output contract. The
existing Megatron CLI output stays replicated.

## Numerical semantics

- Context attention is single-head, unmasked, and includes every key/value row.
  It explicitly forms the full score matrix.
- LayerNorm uses centered population variance, affine scale and bias, and
  `--epsilon` (default `1e-5`). It avoids second-moment cancellation.
- LoRA has unit adapter scaling and no dropout. A is `[hidden, rank]`, B is
  `[rank, output]`. `W + AB` is constructed before applying X.
- Gated MLP applies SiLU to the gate branch, without biases or residuals.
- Kernel attention fixes `phi(x) = max(x, 0) + 1` for Q/K of width `feature_rank`.
  For finite inputs and positive dimensions, its denominator is positive. This
  defines a kernel-attention workload, not a rewrite of softmax attention.
- Graph inputs are fixed row-normalized directed dense adjacency matrices with
  self edges. `--edge-probability` controls generated edges. This prototype does
  not model sparse neighborhood exchange. Activation follows the complete chain.
- Barlow Twins centers each feature over samples and divides by
  `sqrt(population_variance + epsilon)`. It computes
  `C = A_normalizedᵀ B_normalized / samples` and returns
  `sum((diag(C) - 1)^2) + lambda * sum(off_diagonal(C)^2)`.
  `--off-diagonal-weight` defaults to `0.005`. The off-diagonal norm is the full
  correlation squared norm minus its diagonal squared norm; no sample Gram
  matrices are formed.
- MoE ties favor smaller global expert IDs. `--top-k` defaults to 2 and cannot
  exceed `experts`. `--weight-normalization selected` (default) takes softmax over
  selected logits. With `full`, weights are selected entries of full-expert
  softmax and generally sum to less than one. Token dispatch, capacity limits,
  and expert computation are outside this router workload.

Numerical parameters (`epsilon`, `off_diagonal_weight`, `top_k`, and
`weight_normalization`) are operation constructor options in Python. Shape, layout,
and seed belong to the configuration dataclass:

```python
from ml_workloads import MoERouter, MoERouterConfig

operation = MoERouter(top_k=4, weight_normalization="full")
inputs = operation.create_inputs(MoERouterConfig(experts=64, layout="experts"))
output = operation.execute(inputs)  # (global expert IDs, routing weights)
```

## Search input and replication experiments

Each CLI run writes `lowered/NAME/original.mlir` before backend compilation.
Use it as search-engine input: it retains original matrix chains, input mesh and
layouts, and unconstrained outputs for new workloads. Arithmetic is emitted
directly instead of hidden behind JAX convenience-function calls.
Backend-optimized HLO may already fuse or reassociate computations.

Other dumps include `jax_ir*_jit__operation_compile.mlir` and
`module_*jit__operation*.txt`. Use `--output-dir PATH` for another parent directory.
`--execute` runs and synchronizes outputs before printing shapes/shardings.

To reproduce full-tensor boundaries from the report, `--full-tensor-boundary`
explicitly replicates logits in `output_projection_ce`, K/V in
`context_attention`, and X before normalization in `layernorm_linear`.
This is opt-in: the engine currently treats `sdy.sharding_constraint` as a fixed
rewrite boundary, which can prevent transformations that must cross it.
Compare unconstrained originals too.

All workloads lower and execute in JAX. Current TensorLang coverage is narrower:
CE/Barlow Twins use gathers; MoE uses `chlo.top_k` and, for full-softmax weights,
a gather. These operations remain preserved boundaries in the current importer.
The separate Python executor accepts only float32 ports today, so CE's integer
labels and MoE's integer outputs also need executor support for end-to-end runs.
Adding workloads does not add rewrite rules or accelerator performance evidence.

## Dimension and layout comparisons

These examples specify proposed experiments, not performance claims. Use smaller
shapes for quick CPU checks.

```bash
# Query count versus feature rank; queries layout is the control.
uv run python main.py --operation kernel_linear_attention \
  --queries 256 --feature-rank 32 --layout sequence
uv run python main.py --operation kernel_linear_attention \
  --queries 32 --feature-rank 256 --layout sequence

# Aggregation widths; independent_graphs layout is the control.
uv run python main.py --operation graph_convolution \
  --hidden 1024 --output 128 --layout nodes
uv run python main.py --operation graph_convolution \
  --hidden 128 --output 1024 --layout nodes

# Feature correlation versus sample Gram sizes.
uv run python main.py --operation barlow_twins \
  --samples 256 --features 8192 --layout features
uv run python main.py --operation barlow_twins \
  --samples 1024 --features 128 --layout samples

# Expert-sharding case and token-sharding control.
uv run python main.py --operation moe_router --experts 256 --top-k 2 --layout experts
uv run python main.py --operation moe_router --experts 256 --top-k 2 --layout tokens
```

## Checks

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

Tests compare every new workload under every supported layout against independent
NumPy references, check gradients against NumPy directional finite differences,
verify original dot ordering/intermediate shapes and optional constraints, and
exercise every workload through the CLI. Edge cases include large logits,
large-offset LayerNorm inputs, constant Barlow features, and global MoE ties and
both weight-normalization modes. These are four-device CPU correctness checks,
not accelerator latency or collective-traffic benchmarks.


The C++ engine integration tests additionally lower the originals, rewrite them,
verify exported MLIR, and compile/execute the replacements. Build the tool from
the XLA root first:

```bash
bazel build --config=joint_shard //research/joint_shard/tools:parse_stablehlo
```

Then, from this directory:

```bash
JOINT_SHARD_REWRITE_REPORT_DIR=/tmp/joint_shard_workload_validation \
  uv run python -m unittest discover -s tests -p test_rewrite_saturation.py -v
```

The tests cover all layouts and both numerical policies, reversed dimension
controls, and backward programs. They skip if the default binary is absent;
`JOINT_SHARD_REWRITE_BINARY` selects another build. The optional report directory
retains originals, replacements, rewrite diagnostics, and `summary.json`.
Numerical execution removes placement annotations and uses full inputs on one
CPU; the engine verifies the original four-device module. These checks establish
correctness and rule availability, not communication savings or accelerator
speedups. See [the workload report](../joint_shard/WORKLOAD_REWRITES.md) for
selected transformations and bounded-search results. The engine supports moving
row-wise division after the attention dot and the standalone feature-Gram norm
identity. Tests verify shape tradeoffs, numerical permissions, and Gram gradients.
Distributed CE/attention summaries, the full Barlow loss,
hierarchical routing, and projection fusion remain omitted.

## Four-arm experiment and aggressive LayerNorm

The engine implements centered-square reductions to raw moments behind
`--numerical-policy=relaxed --allow-raw-moments`. This permission is disabled in
every engine preset. It accepts cancellation; the original Python LayerNorm
continues to use centered variance.

Build from the XLA root:

```bash
bazel build --config=joint_shard //research/joint_shard/tools:workload_ablation \
  //research/joint_shard/tools:parse_stablehlo
```

Then run from this directory:

```bash
uv run python -m experiments.four_arm --output-dir=/tmp/joint_four_arm
# Smaller regression shapes:
uv run python -m experiments.four_arm --quick --output-dir=/tmp/joint_four_arm_quick
```

The suite covers reduce-dot, dense graph convolution, kernel linear attention,
and LayerNorm: favorable/reversed dimensions and all three layouts for each.
It retains baseline, rewrite-only, sharding-only, and combined arms with fixed
external input/output contracts. Input adapters and output-contract collectives
are included. Layout search enumerates bounded input assignments and lets Shardy
propagate intermediate layouts. Profile candidates plus checked direct target
rule applications keep communication-saving alternatives available even when
they increase computation.

The runner opts LayerNorm into raw moments and automatically repeats its six
cases with the flag off. `--no-raw-moments` disables the opt-in;
`--no-layernorm-controls` skips the repeat. It checks all selected arithmetic on
CPU against independent NumPy references, checks rewritten original backward
programs against JAX gradients, and includes a variance cancellation example.
Cost parameters are configurable and illustrative. Outputs include selected
collective IR, XLA inputs, numerical inputs/references, detailed JSON, and
`REPORT.md`. See [the recorded report](../joint_shard/results/four_arm/REPORT.md).

To measure the exact exported four-device programs, use the separate executor:

```bash
cd ../joint_shard_executor
CUDA_VISIBLE_DEVICES=0,2,3,4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  uv run --with 'jax[cuda12]==0.11.2' python benchmark_four_arm.py \
  /tmp/joint_four_arm --platform=gpu --warmup=5 --repeats=30
```

This compiles the selected artifacts, validates distributed outputs, and records
`accelerator.json`. Timing includes host dispatch and synchronized device
completion; compilation and input/output transfers are excluded. It rotates arm
order between rounds and records every sample. Use four available GPUs on your
host; these example device IDs are specific to the recorded run.
The recorded GPU run disables Triton GEMM and TF32 through the shown runtime
settings to satisfy the same numerical tolerance as CPU. Default GPU matmul
settings exceeded that tolerance on the original reduce-dot baseline.
