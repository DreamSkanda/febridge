"""ODE-based flow sampling with logp computation."""

import jax
import jax.numpy as jnp
from jax.experimental import ode
from functools import partial
from typing import Callable, Tuple

import haiku as hk

from .divergence import divergence_fwd


def make_flow(
    vec_field_net: Callable,
    X0: jnp.ndarray,
    X1: jnp.ndarray,
    mxstep: int = 1000,
) -> Callable:
    """Creates an ODE-based flow sampler with logp computation.

    Args:
        vec_field_net: Vector field network with signature (params, x, t) -> vector.
        X0: Reference samples (shape [num_samples, n*dim]).
        X1: Target samples (shape [num_samples, n*dim]).
        mxstep: Maximum number of ODE integration steps.

    Returns:
        sample_and_logp_fn with signature (key, params, batchsize, sign) ->
        (x0, x1, logp).
    """

    def sample_x0(key: jax.random.PRNGKey, batchsize: int) -> jnp.ndarray:
        idx = jax.random.choice(key, jnp.arange(len(X0)), (batchsize,), replace=False)
        return X0[idx]

    def sample_x1(key: jax.random.PRNGKey, batchsize: int) -> jnp.ndarray:
        idx = jax.random.choice(key, jnp.arange(len(X1)), (batchsize,), replace=False)
        return X1[idx]

    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=(0, 0, 0))
    def integrate_with_logp(
        params: hk.Params, x0: jnp.ndarray, sign: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        def _ode(state: Tuple[jnp.ndarray, float], t: float) -> Tuple[jnp.ndarray, float]:
            x = state[0]
            time = t if sign > 0 else 1 - t
            return (
                sign * vec_field_net(params, x, time),
                -sign * divergence_fwd(vec_field_net)(params, x, t),
            )

        logp0 = 0.0
        xt, logpt = ode.odeint(
            _ode,
            [x0, logp0],
            jnp.array([0.0, 1.0]),
            rtol=1e-10,
            atol=1e-10,
            mxstep=mxstep,
        )
        if sign > 0:
            x0_out, x1_out = xt[0], xt[1]
        else:
            x0_out, x1_out = xt[1], xt[0]
        return x0_out, x1_out, sign * logpt[-1]

    @partial(jax.jit, static_argnums=(2, 3))
    def sample_and_logp_fn(
        key: jax.random.PRNGKey,
        params: hk.Params,
        batchsize: int,
        sign: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        key1, key2 = jax.random.split(key)
        if sign > 0:
            x = sample_x0(key1, batchsize)
        else:
            x = sample_x1(key1, batchsize)
        return integrate_with_logp(params, x, sign)

    return sample_and_logp_fn
