import jax
import time
import os
import sys

from net import make_transformer
from flow import make_flow
from loss import make_loss
from energy import make_free_energy
from train import train_and_evaluate

def fediff(rng, X0, X1, logp_fun_0, logp_fun_1, n, dim, nheads=2, nlayers=2, keysize=2, epochs=1000, batchsize=2048, lr=0.01, sign=1):

    """
    rng: random key
    X0: a array of samples of logp_fun_0
    X1: a array of samples of logp_fun_1
    logp_fun_0: a jax function
    logp_fun_1: a jax funciton
    """
    
    init_rng, rng = jax.random.split(rng)

    print("\n========== Processing samples ==========")
    if X0.shape[0] < batchsize or X1.shape[0] < batchsize:
        print("\nYour sample set size is smaller than the batch size you have set.")
        sys.exit(1)
    elif X0.shape[0] != X1.shape[0]:
        print("\nThe sizes of X0 and X1 are not equal.")
        sys.exit(1)
        # datasize = min(X0.shape[0], X1.shape[0])
        # iterations = int(0.8 * datasize // batchsize)
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
        print(f"Train size: {train_size}, Validation size: {val_size}")
    
    print("\n========== Constructing transformer network ==========")
    params, vec_field_net = make_transformer(init_rng, n, dim, nheads, nlayers, keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (nlayers, nheads, keysize)
    
    """initializing the sampler and logp calculator"""
    sample_fn, sample_and_logp_fn = make_flow(vec_field_net, X0, X1)
    free_energy = make_free_energy(sample_and_logp_fn, logp_fun_0, logp_fun_1, n, dim)
    
    """initializing the loss function"""
    loss = make_loss(vec_field_net)
    value_and_grad = jax.value_and_grad(loss)

    ####################################################################################

    print("\n========== Training ==========")
    hyperparams = (epochs, iterations, batchsize)
    
    training_data = (X0[0:train_size], X1[0:train_size])
    validation_data = (X0[train_size:train_size+val_size], X1[train_size:train_size+val_size])

    start = time.time()
    params = train_and_evaluate(rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, lr, path)
    end = time.time()

    running_time = end - start
    print("training time: %.5f sec" %running_time)

    ####################################################################################

    print("\n========== Sampling and calculating free energy ==========")

    fe_rng, rng = jax.random.split(rng)

    start = time.time()
    fe, fe_err, f = free_energy(fe_rng, params, batchsize, sign)
    end = time.time()

    running_time = end - start
    print("free energy: %f ± %f" %(fe, fe_err))
    print("importance sampling time: %.5f sec" %running_time)

    return fe, fe_err