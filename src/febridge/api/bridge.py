"""High-level API for febridge."""

import os
import sys
from typing import Union, Callable, List, Tuple

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from ..core import make_transformer
from ..flows import make_flow
from ..losses import make_loss
from ..energy import make_free_energy
from ..training import train_and_evaluate
from ..utils import SampleSizeError, SizeMismatchError


ArrayLike = Union[jnp.ndarray, np.ndarray]


def model_train(
    X0: ArrayLike,
    X1: ArrayLike,
    rng: jax.random.PRNGKey,
    *,
    path: str = os.getcwd(),
    nheads: int = 2,
    nlayers: int = 2,
    keysize: int = 2,
    epochs: int = 1000,
    batchsize: int = 2048,
    lr: float = 0.01,
) -> hk.Params:
    """Train the vector field network for free energy calculation.

    Args:
        X0: Reference distribution samples of shape [num_samples, n*dim].
        X1: Target distribution samples of shape [num_samples, n*dim].
        rng: PRNG key for initialization.
        path: Directory for checkpoints and logs.
        nheads: Number of attention heads.
        nlayers: Number of transformer layers.
        keysize: Key size for attention.
        epochs: Number of training epochs.
        batchsize: Training batch size.
        lr: Learning rate.

    Returns:
        Trained model parameters.

    Raises:
        SampleSizeError: If sample set is smaller than batch size.
        SizeMismatchError: If X0 and X1 have different sizes.
    """
    init_rng, rng = jax.random.split(rng)

    X0 = jnp.array(X0)
    X1 = jnp.array(X1)

    print("\n=================== Processing samples ===================")
    if X0.shape[0] < batchsize or X1.shape[0] < batchsize:
        raise SampleSizeError(
            "Sample set size is smaller than the batch size. "
            f"Got {X0.shape[0]} samples and batchsize {batchsize}."
        )
    elif X0.shape[0] != X1.shape[0]:
        raise SizeMismatchError(
            f"X0 and X1 must have equal sizes. Got X0: {X0.shape[0]}, X1: {X1.shape[0]}."
        )
    else:
        print("\nThe sizes of X0 and X1 are equal.")
        datasize = X0.shape[0]
        target_train_size = int(datasize * 0.8)
        iterations = target_train_size // batchsize
        train_size = iterations * batchsize
        remaining_size = datasize - train_size
        val_size = (remaining_size // batchsize) * batchsize
        print(f"Train size: {train_size}, Validation size: {val_size}.")

    print("\n============ Constructing transformer network ============")
    n = X0.shape[1]
    dim = X0.shape[2]
    params, vec_field_net = make_transformer(init_rng, n, dim, nheads, nlayers, keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (nlayers, nheads, keysize)
    print(f"\nModel Name: {modelname}")

    loss = make_loss(vec_field_net)
    value_and_grad = jax.value_and_grad(loss)

    print("\n======================== Training ========================")
    hyperparams = (epochs, iterations, batchsize)

    training_data = (X0[0:train_size], X1[0:train_size])
    validation_data = (X0[train_size : train_size + val_size], X1[train_size : train_size + val_size])

    params = train_and_evaluate(
        rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, lr, path
    )

    return params


def model_inference(
    X0: ArrayLike,
    X1: ArrayLike,
    logp_fun_0: Callable,
    logp_fun_1: Callable,
    params: hk.Params,
    rng: jax.random.PRNGKey,
    *,
    sign: int = 1,
    nheads: int = 2,
    nlayers: int = 2,
    keysize: int = 2,
    batchsize: int = 2048,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Run inference to compute free energy.

    Args:
        X0: Reference distribution samples of shape [num_samples, n*dim].
        X1: Target distribution samples of shape [num_samples, n*dim].
        logp_fun_0: Log probability function for reference distribution.
        logp_fun_1: Log probability function for target distribution.
        params: Trained model parameters.
        rng: PRNG key.
        sign: 1 for upper bound, -1 for lower bound.
        nheads: Number of attention heads (must match training).
        nlayers: Number of transformer layers (must match training).
        keysize: Key size for attention (must match training).
        batchsize: Inference batch size.

    Returns:
        Tuple of (x0, x1, logp, free_energy).

    Raises:
        SampleSizeError: If sample set is smaller than batch size.
        SizeMismatchError: If X0 and X1 have different sizes.
    """
    init_rng, rng = jax.random.split(rng)

    X0 = jnp.array(X0)
    X1 = jnp.array(X1)

    print("\n=================== Processing samples ===================")
    if X0.shape[0] < batchsize or X1.shape[0] < batchsize:
        raise SampleSizeError(
            "Sample set size is smaller than the batch size. "
            f"Got {X0.shape[0]} samples and batchsize {batchsize}."
        )
    elif X0.shape[0] != X1.shape[0]:
        raise SizeMismatchError(
            f"X0 and X1 must have equal sizes. Got X0: {X0.shape[0]}, X1: {X1.shape[0]}."
        )
    else:
        print(f"\nThe sizes of X0 and X1 are equal, data size: {X0.shape[0]}.")

    print("\n============ Constructing transformer network ============")
    n = X0.shape[1]
    dim = X0.shape[2]
    _, vec_field_net = make_transformer(init_rng, n, dim, nheads, nlayers, keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (nlayers, nheads, keysize)
    print(f"\nModel Name: {modelname}")

    # Flatten samples for make_flow (expects n*dim)
    X0_flat = X0.reshape(-1, n * dim)
    X1_flat = X1.reshape(-1, n * dim)
    sample_and_logp_fn = make_flow(vec_field_net, X0_flat, X1_flat)
    free_energy_fn = make_free_energy(sample_and_logp_fn, logp_fun_0, logp_fun_1, n, dim)

    print("\n====================== Inference =======================")

    key = jax.random.PRNGKey(42)
    x0, x1, logp = sample_and_logp_fn(key, params, X0.shape[0], sign)

    fe, fe_err, _ = free_energy_fn(rng, params, batchsize, sign)

    print(f"\nFree Energy: {fe}")

    return x0, x1, logp, fe
