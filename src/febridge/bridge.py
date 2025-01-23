import jax
import os
import sys
import haiku as hk

from .net import make_transformer
from .flow import make_flow
from .loss import make_loss
from .energy import make_free_energy
from .train import train_and_evaluate

from typing import Union
from types import FunctionType
from typing import List
import numpy as np
import jax.numpy as jnp

def model_train(
        X0: Union[jnp.ndarray, np.ndarray],
        X1: Union[jnp.ndarray, np.ndarray],
        rng,
        path=os.getcwd(),
        nheads=2, nlayers=2, keysize=2,
        epochs=1000,batchsize=2048,lr=0.01
    ) -> hk.Params:
    
    init_rng, rng = jax.random.split(rng)

    print("\n=================== Processing samples ===================")
    if X0.shape[0] < batchsize or X1.shape[0] < batchsize:
        print("\nYour sample set size is smaller than the batch size you have set.")
        sys.exit(1)
    elif X0.shape[0] != X1.shape[0]:
        print("\nThe sizes of X0 and X1 are not equal.")
        sys.exit(1)
    else:
        print("\nThe sizes of X0 and X1 are equal.")
        datasize = X0.shape[0]
        # Calculate the target train size
        target_train_size = int(datasize * 0.8)
        # Adjust the train size so that it can be divisible by the batch size
        iterations = target_train_size // batchsize
        train_size = iterations * batchsize
        # Adjust the validation size so that it can be divisible by the batch size
        remaining_size = datasize - train_size
        val_size = (remaining_size // batchsize) * batchsize
        print(f"Train size: {train_size}, Validation size: {val_size}.")
    
    print("\n============ Constructing transformer network ============")
    n = X0.shape[1]
    dim = X0.shape[2]
    params, vec_field_net = make_transformer(init_rng, n, dim, nheads, nlayers, keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (nlayers, nheads, keysize)
    print(f"\nModel Name: {modelname}")
    
    """initializing the loss function"""
    loss = make_loss(vec_field_net)
    value_and_grad = jax.value_and_grad(loss)

    ####################################################################################

    print("\n======================== Training ========================")
    hyperparams = (epochs, iterations, batchsize)
    
    training_data = (X0[0:train_size], X1[0:train_size])
    validation_data = (X0[train_size:train_size+val_size], X1[train_size:train_size+val_size])

    params = train_and_evaluate(rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, lr, path)

    return params

def model_inference(
        X0: Union[jnp.ndarray, np.ndarray],
        X1: Union[jnp.ndarray, np.ndarray],
        logp_fun_0: FunctionType,
        logp_fun_1: FunctionType,
        rng,
        model: hk.Params,
        sign=1,
        nheads=2, nlayers=2, keysize=2,
        batchsize=2048
    ) -> List:
    
    init_rng, rng = jax.random.split(rng)

    print("\n=================== Processing samples ===================")
    if X0.shape[0] < batchsize or X1.shape[0] < batchsize:
        print("\nYour sample set size is smaller than the batch size you have set.")
        sys.exit(1)
    elif X0.shape[0] != X1.shape[0]:
        print("\nThe sizes of X0 and X1 are not equal.")
        sys.exit(1)
    else:
        print("\nThe sizes of X0 and X1 are equal, data size: %d." % X0.shape[0])
    
    print("\n============ Constructing transformer network ============")
    n = X0.shape[1]
    dim = X0.shape[2]
    _, vec_field_net = make_transformer(init_rng, n, dim, nheads, nlayers, keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (nlayers, nheads, keysize)
    print(f"\nModel Name: {modelname}")
    
    """initializing the sampler and logp calculator"""
    sample_and_logp_fn = make_flow(vec_field_net, X0, X1)
    free_energy = make_free_energy(sample_and_logp_fn, logp_fun_0, logp_fun_1, n, dim)

    ####################################################################################

    print("\n====================== Inference ======================")
    
    # Run inference
    key = jax.random.PRNGKey(42)
    x0, x1, logp = sample_and_logp_fn(key, model, X0.shape[0], sign)
    
    # Calculate free energy
    fe = free_energy(rng, model, batchsize, sign)
    
    print(f"\nFree Energy: {fe}")
    
    return [x0, x1, logp, fe]