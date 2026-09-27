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
physical GPUs on one host. GPU execution has not yet been validated here.

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
