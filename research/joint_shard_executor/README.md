# Run an optimizer-selected program with prebuilt XLA

The optimizer writes two artifacts: `selected.mlir` records the chosen global
StableHLO + Shardy program, and `xla_input.mlir` contains its exported,
XLA-ready StableHLO. The Python runner here consumes only `xla_input.mlir`. It
does not import the optimizer, invoke its compiler passes, or reselect a path.

From the workspace root, build the optimizer and produce both files:

```sh
bazel build --config=joint_shard //research/joint_shard/tools:summarize_regions
bazel-bin/research/joint_shard/tools/summarize_regions \
  research/joint_shard/testdata/residual_block.mlir \
  --optimize-dag --output-dir=/tmp/joint_plan
```

`--output-dir` is required for `--optimize-chain` or `--optimize-dag`. It contains
`selected.mlir` and `xla_input.mlir`. `--dump-dir` remains optional and writes
candidate/search snapshots separately. The optimizer uses upstream Shardy's
global-to-local pass and XLA's StableHLO exporter. Its C++ frontend build needs
MLIR/StableHLO/Shardy, but no locally built XLA CPU/GPU backend.

From the workspace root, set up the executor's uv environment:

```sh
cd research/joint_shard_executor
uv sync --locked
```

Python 3.12 is selected by `.python-version`. Dependencies live in
`pyproject.toml` and `uv.lock`; uv creates this directory's `.venv`.
`no-build = true` requires prebuilt wheels instead of source builds.

From this executor directory, compile or execute the exported artifact:

```sh
uv run --locked python run.py \
  /tmp/joint_plan/xla_input.mlir --compile-only
uv run --locked python run.py \
  /tmp/joint_plan/xla_input.mlir --platform=cpu --dump-dir=/tmp/joint_execution
```

The runner creates enough logical CPU devices for the exported partition count.
`--platform=gpu` uses an installed matching JAX GPU plugin and needs enough
physical GPUs on one host. The four-arm experiment adds GPU numerical and
synchronized latency checks; see [the recorded report](../joint_shard/results/four_arm/REPORT.md).

Execution uses seeded global f32 inputs by default (`--seed=42`). For real
inputs, pass an NPZ with keys `arg0`, `arg1`, etc., using `--inputs=...`.
`--outputs=...` writes global results keyed `result0`, `result1`, etc. With
`--dump-dir`, output defaults to `outputs.npz` there, alongside XLA's `xla/`
compiler dumps. `--repeats` controls execution count.

The runner is pinned to JAX/jaxlib 0.11.2 because it uses their low-level PJRT
and MLIR bindings. It expects a single-host `@main` with static ranked f32
ports, `mhlo.sharding` on each port, and explicit partition/replica metadata.
XLA imports the StableHLO and compiles it for the selected hardware. The
optimizer's manual body protects the selected device-local computation from
repartitioning.

To validate after building the optimizer, run the C++ test from the workspace root:

```sh
bazel test --config=joint_shard //research/joint_shard:tests --test_output=errors
```

Then run the integration test from this executor directory:

```sh
uv run --locked python test_run.py -v
```

The C++ export test checks source preservation, collective counts, and invalid
input contracts. The Python test checks both optimizer artifacts, compilation,
numerical chain/residual results, NPZ inputs/outputs, and XLA dumps on CPU.

## Benchmark selected four-arm programs

First generate inputs and all selected artifacts with
`joint_shard_python/experiments/four_arm.py`, as described in
[its README](../joint_shard_python/README.md#four-arm-experiment-and-aggressive-layernorm).
From this directory:

```sh
CUDA_VISIBLE_DEVICES=0,2,3,4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  uv run --with 'jax[cuda12]==0.11.2' python benchmark_four_arm.py \
  /tmp/joint_four_arm --platform=gpu --warmup=5 --repeats=30
```

Choose four available GPU IDs on your host. The optional CUDA dependency supplies
the matching prebuilt plugin and runtime libraries without changing the CPU
environment. `--platform=cpu` checks distributed CPU execution instead.
`--case=layernorm_linear.hidden.favorable` selects one case.
The recorded run uses the shown TF32/Triton settings because default GPU matmul
settings failed the strict reference tolerance on the original baseline. The
selected arithmetic and shardings are preserved; backend code generation changes.

The benchmark compiles each exact exported arm, uploads inputs once, warms up,
rotates arm order, and waits for every output shard after every timed call.
Compilation and host/device transfers are excluded; Python/PJRT dispatch is
included. Global outputs must match saved independent numerical references.
`accelerator.json` records all samples, numerical errors, failures, and a device
snapshot; `ACCELERATOR.md` summarizes medians. Shared device load and single-run
noise mean a lower median alone is not established performance evidence.
