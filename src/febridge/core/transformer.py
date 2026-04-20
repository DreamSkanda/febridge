"""Transformer network for febridge."""

import dataclasses
from typing import Tuple, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from .layers import layer_norm


@dataclasses.dataclass
class Transformer(hk.Module):
    """A transformer stack."""

    num_heads: int
    num_layers: int
    key_size: int
    widening_factor: int = 1
    name: Optional[str] = None

    def __call__(
        self,
        embeddings: jnp.ndarray,  # [T, D]
        t: float,
    ) -> jnp.ndarray:  # [T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        seq_len, dim = embeddings.shape
        model_size = dim + 1

        embeddings = jnp.concatenate(
            [
                embeddings,
                jnp.repeat(jnp.array(t).reshape(1, 1), seq_len, axis=0),
            ],
            axis=1,
        )
        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)

        h = embeddings
        for _ in range(self.num_layers):
            # First the attention block.
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=model_size,
                w_init=initializer,
            )
            h_norm = layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm)
            h = h + h_attn

            # Then the dense block.
            dense_block = hk.Sequential(
                [
                    hk.Linear(self.widening_factor * model_size, w_init=initializer),
                    jax.nn.gelu,
                    hk.Linear(model_size, w_init=initializer),
                ]
            )
            h_norm = layer_norm(h)
            h_dense = dense_block(h_norm)
            h = h + h_dense

        h = hk.Linear(dim, w_init=initializer)(h)
        return layer_norm(h)


def make_transformer(
    key: jax.random.PRNGKey,
    n: int,
    dim: int,
    num_heads: int,
    num_layers: int,
    key_sizes: int,
) -> Tuple[hk.Params, Callable]:
    """Creates a transformer network for vector field prediction.

    Args:
        key: PRNG key for initialization.
        n: Sequence length.
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        key_sizes: Key size for attention.

    Returns:
        A tuple of (params, apply_fn) where apply_fn has signature
        (params, x, t) -> output.
    """
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x: jnp.ndarray, t: float) -> jnp.ndarray:
        net = Transformer(num_heads, num_layers, key_sizes)
        return net(x.reshape(n, dim), t).reshape(n * dim)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply
