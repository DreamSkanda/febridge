import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

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
      t: float
  ) -> jnp.ndarray:  # [T, D]
    """Transforms input embedding sequences to output embedding sequences."""
    
    seq_len, dim = embeddings.shape
    model_size = dim + 1

    embeddings = jnp.concatenate([embeddings, 
                         jnp.repeat(jnp.array(t).reshape(1, 1), seq_len, axis=0)], 
                         axis=1)
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
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = layer_norm(h)
      h_dense = dense_block(h_norm)
      h = h + h_dense
        
    h = hk.Linear(dim, w_init=initializer)(h)
    return layer_norm(h)

def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  """Applies a unique LayerNorm to x with default settings."""
  return x

def make_transformer(key, n, dim, num_heads, num_layers, key_sizes):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Transformer(num_heads, num_layers, key_sizes)
        return net(x.reshape(n, dim), t).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply 

if __name__ == '__main__':
    from jax.config import config
    config.update("jax_enable_x64", True)

    n = 6
    spatial_dim = 2

    params, vec_field_net = make_vec_field_net(jax.random.PRNGKey(42), n, spatial_dim, symmetry=False)

    x = jax.random.normal(jax.random.PRNGKey(41), (n*spatial_dim,))
    t = jax.random.normal(jax.random.PRNGKey(40), (1,))

    import time
    start = time.time()
    v = vec_field_net(params,  x, t)
    end = time.time()
    print(end - start)
    print(v)
    print(vec_field_net(jax.tree_util.tree_map(lambda x: x -0.01, params),  x, t))
