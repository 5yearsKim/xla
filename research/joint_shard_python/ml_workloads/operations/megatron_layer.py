"""Megatron-style tensor parallel attention and MLP workload."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .base import BaseOperation, BaseOperationInput


@dataclass(frozen=True)
class MegatronLayerConfig:
    """Shape and random seed for one Megatron layer workload."""

    batch: int = 8
    sequence: int = 128
    hidden: int = 1024
    ffn: int = 4096
    seed: int = 0


@dataclass(frozen=True)
class MegatronLayerInput(BaseOperationInput):
    """Activation and projection weights for a single transformer layer."""

    x: jax.Array
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array

    def arrays(self) -> tuple[jax.Array, ...]:
        return (self.x, self.wq, self.wk, self.wv, self.wo, self.w1, self.w2)


class MegatronLayer(BaseOperation[MegatronLayerInput, MegatronLayerConfig]):
    """Single-head attention followed by a GELU MLP, with tensor parallel weights."""

    def __init__(self) -> None:
        super().__init__("megatron_layer")
        self.mesh = Mesh(np.asarray(jax.devices()), axis_names=("model",))
        self.output_sharding = NamedSharding(self.mesh, P(None, None, None))

    def create_inputs(self, config: MegatronLayerConfig) -> MegatronLayerInput:
        """Initialize arrays and place each weight on its intended mesh partition."""
        batch, sequence, hidden, ffn = (
            config.batch,
            config.sequence,
            config.hidden,
            config.ffn,
        )
        if min(batch, sequence, hidden, ffn) <= 0:
            raise ValueError("All dimensions must be positive")
        model_size = self.mesh.shape["model"]
        if hidden % model_size or ffn % model_size:
            raise ValueError("Hidden and FFN dimensions must be divisible by mesh size")

        key = jax.random.key(config.seed)
        keys = jax.random.split(key, 7)
        column = NamedSharding(self.mesh, P(None, "model"))
        row = NamedSharding(self.mesh, P("model", None))

        def placed(
            key: jax.Array, shape: tuple[int, ...], sharding: NamedSharding
        ) -> jax.Array:
            return jax.device_put(jax.random.normal(key, shape), sharding)

        return MegatronLayerInput(
            x=placed(keys[0], (batch, sequence, hidden), self.output_sharding),
            wq=placed(keys[1], (hidden, hidden), column),
            wk=placed(keys[2], (hidden, hidden), column),
            wv=placed(keys[3], (hidden, hidden), column),
            wo=placed(keys[4], (hidden, hidden), row),
            w1=placed(keys[5], (hidden, ffn), column),
            w2=placed(keys[6], (ffn, hidden), row),
        )

    def _operation(
        self,
        x: jax.Array,
        wq: jax.Array,
        wk: jax.Array,
        wv: jax.Array,
        wo: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
    ) -> jax.Array:
        """Apply attention and MLP, each followed by a residual connection."""
        q = x @ wq
        k = x @ wk
        v = x @ wv

        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(
            jnp.asarray(x.shape[-1], dtype=q.dtype)
        )
        probs = jax.nn.softmax(scores, axis=-1)
        attn = probs @ v
        x = x + attn @ wo

        h = jax.nn.gelu(x @ w1)
        return x + h @ w2

    def execute(self, inputs: MegatronLayerInput) -> jax.Array:
        """Compile and execute the layer with replicated output."""
        return jax.jit(self._operation, out_shardings=self.output_sharding)(
            *inputs.arrays()
        )
