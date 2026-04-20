"""Layer implementations for febridge."""

import jax.numpy as jnp
from typing import Optional


def layer_norm(
    x: jnp.ndarray,
    offset: Optional[jnp.ndarray] = None,
    scale: Optional[jnp.ndarray] = None,
    epsilon: float = 1e-5,
) -> jnp.ndarray:
    """Applies layer normalization to x.

    Args:
        x: Input array of shape [..., dim].
        offset: Optional bias term.
        scale: Optional scale term.
        epsilon: Small constant for numerical stability.

    Returns:
        Normalized array of the same shape as x.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(variance + epsilon)

    if scale is not None:
        x_norm = x_norm * scale
    if offset is not None:
        x_norm = x_norm + offset

    return x_norm
