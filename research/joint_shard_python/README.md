# Megatron layer workload

Lower a single-head attention and GELU MLP layer with Megatron-style tensor
parallel weight shardings. The program sets the CPU device count to four before
JAX loads, then builds a concrete `model` mesh from those devices. The default
shape is `B=8`, `S=128`, `H=1024`, `FFN=4096`.

From `python-apps/`, run:

```bash
uv run python main.py
```

This compiles the layer and writes XLA dumps to `lowered/megatron_layer/`.
Files include `jax_ir*_jit__operation_compile.mlir` (StableHLO) and
`module_*jit__operation*.txt` (HLO before, during, and after optimization).
JAX also dumps its input initialization computations there. Add `--execute`
to run the compiled layer. Use `--batch`, `--sequence`, `--hidden`, and `--ffn`
to change the shape. The hidden and FFN dimensions must each be divisible by
the device count.

For a small four-device CPU run, without setting an environment variable:

```bash
uv run python main.py --batch 2 --sequence 4 --hidden 8 --ffn 16 --execute
```

Use `--output-dir PATH` to choose a different parent directory for the dumps.
