"""Free energy bound calculation for febridge."""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple

import haiku as hk


def make_free_energy(
    batched_sampler: Callable,
    logp_fun_0: Callable,
    logp_fun_1: Callable,
    n: int,
    dim: int,
) -> Callable[[jax.random.PRNGKey, hk.Params, int, int], Tuple[float, float, jnp.ndarray]]:
    """Creates a free energy bound calculator.

    Args:
        batched_sampler: Sampler with signature (rng, params, batchsize, sign) ->
            (x0, x1, logp).
        logp_fun_0: Log probability function for reference distribution.
        logp_fun_1: Log probability function for target distribution.
        n: Sequence length.
        dim: Dimension per particle.

    Returns:
        free_energy_bound function with signature (rng, params, batchsize, sign) ->
        (mean, std, samples).
    """

    def free_energy_bound(
        rng: jax.random.PRNGKey,
        params: hk.Params,
        batchsize: int,
        sign: int,
    ) -> Tuple[float, float, jnp.ndarray]:
        """Computes free energy bound.

        Args:
            rng: PRNG key.
            params: Model parameters.
            batchsize: Number of samples.
            sign: 1 for upper bound, -1 for lower bound.

        Returns:
            Tuple of (mean, standard_error, all_f_values).
        """
        x0, x1, logp = batched_sampler(rng, params, batchsize, sign)

        # logp_fun returns scalar per sample after vmap over batch
        e0 = jax.vmap(logp_fun_0, (0, None, None))(x0.reshape(batchsize, n, dim), n, dim)
        e1 = jax.vmap(logp_fun_1, (0, None, None))(x1.reshape(batchsize, n, dim), n, dim)

        f = e1 - e0 + logp

        return f.mean(), f.std() / jnp.sqrt(batchsize), f

    return free_energy_bound
