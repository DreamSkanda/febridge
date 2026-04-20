"""Training loop with evaluation for febridge."""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import NamedTuple, Tuple, Callable
import itertools
import os

from .checkpoint import save_data


class TrainingState(NamedTuple):
    """Container for training state."""
    params: hk.Params
    opt_state: optax.OptState


def train_and_evaluate(
    rng: jax.random.PRNGKey,
    loss: Callable,
    value_and_grad: Callable,
    hyperparams: Tuple[int, int, int],
    params: hk.Params,
    training_data: Tuple[jnp.ndarray, jnp.ndarray],
    validation_data: Tuple[jnp.ndarray, jnp.ndarray],
    lr: float,
    path: str,
) -> hk.Params:
    """Training loop with periodic validation.

    Args:
        rng: PRNG key.
        loss: Loss function with signature (params, x0, x1, t) -> loss.
        value_and_grad: Function with signature (params, x0, x1, t) -> (value, grad).
        hyperparams: Tuple of (num_epochs, num_iterations, batchsize).
        params: Initial model parameters.
        training_data: Tuple of (X0_train, X1_train).
        validation_data: Tuple of (X0_val, X1_val).
        lr: Learning rate.
        path: Directory to save checkpoints.

    Returns:
        Trained model parameters.
    """
    num_epochs, num_iterations, batchsize = hyperparams

    training_X0, training_X1 = training_data
    validation_X0, validation_X1 = validation_data

    # Use same batchsize for validation if divisible, otherwise adjust
    validation_size = len(validation_X1)
    if validation_size % batchsize == 0:
        validation_batchsize = batchsize
    else:
        validation_batchsize = validation_size // num_iterations

    num_validation_iterations = validation_size // validation_batchsize

    assert len(training_X1) // batchsize == num_iterations
    assert len(training_X1) % batchsize == 0
    assert validation_size % validation_batchsize == 0

    @jax.jit
    def train_step(
        rng: jax.random.PRNGKey,
        i: int,
        state: TrainingState,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
    ) -> Tuple[TrainingState, float]:
        t = jax.random.uniform(rng, (x0.shape[0],))
        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value

    @jax.jit
    def test_step(
        rng: jax.random.PRNGKey,
        i: int,
        state: TrainingState,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
    ) -> float:
        t = jax.random.uniform(rng, (x0.shape[0],))
        value = loss(state.params, x0, x1, t)
        return value

    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "loss.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()

    for epoch in range(1, num_epochs + 1):
        permute_rng, rng = jax.random.split(rng)
        training_X0 = jax.random.permutation(permute_rng, training_X0)
        training_X1 = jax.random.permutation(permute_rng, training_X1)

        train_loss = 0.0
        validation_loss = 0.0
        counter = 0

        # Train
        for batch_index in range(0, num_iterations * batchsize, batchsize):
            x0 = training_X0[batch_index : batch_index + batchsize]
            x1 = training_X1[batch_index : batch_index + batchsize]

            step_rng, rng = jax.random.split(rng)
            state, d_mean = train_step(step_rng, next(itercount), state, x0, x1)
            train_loss += d_mean
            counter += 1

        # Validation
        for batch_index in range(0, num_validation_iterations * validation_batchsize, validation_batchsize):
            x0 = validation_X0[batch_index : batch_index + validation_batchsize]
            x1 = validation_X1[batch_index : batch_index + validation_batchsize]

            step_rng, rng = jax.random.split(rng)
            d_mean = test_step(step_rng, next(itercount), state, x0, x1)
            validation_loss += d_mean

        f.write(("%6d" + "  %.6f" + "  %.6f" + "\n") % (epoch, train_loss / counter, validation_loss / counter))

        if epoch % 100 == 0:
            ckpt = {"params": state.params}
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" % epoch)
            save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params
