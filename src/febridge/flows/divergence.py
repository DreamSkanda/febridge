"""Divergence computation for flow models."""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Callable


def divergence_fwd(
    f: Callable[[hk.Params, jnp.ndarray, float], jnp.ndarray],
) -> Callable:
    """Creates a function that computes the divergence of f.

    Args:
        f: Vector field function with signature (params, x, t) -> vector.

    Returns:
        A function (params, x, t) -> divergence (float).
    """
    def _div_f(params: hk.Params, x: jnp.ndarray, t: float) -> float:
        jac = jax.jacfwd(lambda x: f(params, x, t))
        return jnp.trace(jac(x))

    return _div_f
