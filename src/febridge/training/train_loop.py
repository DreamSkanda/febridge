"""Core training loop for febridge."""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import NamedTuple, Tuple, Callable
import itertools

from .checkpoint import save_data


class TrainingState(NamedTuple):
    """Container for training state."""
    params: hk.Params
    opt_state: optax.OptState


def train(
    rng: jax.random.PRNGKey,
    value_and_grad: Callable,
    hyperparams: Tuple[int, int, int],
    params: hk.Params,
    data: jnp.ndarray,
    lr: float,
    path: str,
) -> hk.Params:
    """Basic training loop without validation.

    Args:
        rng: PRNG key.
        value_and_grad: Function with signature (params, x0, x1, t) -> (value, grad).
        hyperparams: Tuple of (num_epochs, num_iterations, batchsize).
        params: Initial model parameters.
        data: Training data (x1 samples).
        lr: Learning rate.
        path: Directory to save checkpoints.

    Returns:
        Trained model parameters.
    """
    num_epochs, num_iterations, batchsize = hyperparams
    assert len(data) // batchsize == num_iterations and len(data) % batchsize == 0

    @jax.jit
    def step(rng: jax.random.PRNGKey, i: int, state: TrainingState, x1: jnp.ndarray) -> Tuple[TrainingState, float]:
        sample_rng, rng = jax.random.split(rng)
        x0 = jax.random.normal(sample_rng, x1.shape)
        t = jax.random.uniform(rng, (batchsize,))

        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value

    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "loss.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()

    for epoch in range(1, num_epochs + 1):
        permute_rng, rng = jax.random.split(rng)
        data = jax.random.permutation(permute_rng, data)

        total_loss = 0.0
        counter = 0
        for batch_index in range(0, num_iterations * batchsize, batchsize):
            x1_batch = data[batch_index : batch_index + batchsize]

            step_rng, rng = jax.random.split(rng)
            state, d_mean = step(step_rng, next(itercount), state, x1_batch)
            total_loss += d_mean
            counter += 1

        f.write(("%6d" + "  %.6f" + "\n") % (epoch, total_loss / counter))

        if epoch % 100 == 0:
            ckpt = {"params": state.params}
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" % epoch)
            save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params


import os
