"""Vector field matching loss for febridge."""

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from typing import Callable

import haiku as hk


def make_loss(
    vec_field_net: Callable[[hk.Params, jnp.ndarray, float], jnp.ndarray],
) -> Callable[[hk.Params, jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
    """Creates a vector field matching loss function.

    The loss computes mean squared error between the true displacement
    (x1 - x0) and the predicted vector field.

    Args:
        vec_field_net: Vector field network with signature (params, x, t) -> vector.
            Expected to take x of shape (n*dim,) and return (n*dim,).

    Returns:
        Loss function with signature (params, x0, x1, t) -> loss float.
    """

    @partial(vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params: hk.Params, x0: jnp.ndarray, x1: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # Flatten inputs for vec_field_net
        x0_flat = x0.reshape(-1)
        x1_flat = x1.reshape(-1)
        x_flat = t * x1_flat + (1 - t) * x0_flat
        pred_flat = vec_field_net(params, x_flat, t)
        diff_flat = x1_flat - x0_flat
        return jnp.sum((diff_flat - pred_flat) ** 2)

    def loss(params: hk.Params, x0: jnp.ndarray, x1: jnp.ndarray, t: jnp.ndarray) -> float:
        m = _matching(params, x0, x1, t)
        return jnp.mean(m)

    return loss
