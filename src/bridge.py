import jax
import os
import sys

from net import make_transformer
from flow import make_flow
from loss import make_loss
from energy import make_free_energy
from train import train_and_evaluate

def fediff(rng, X0, X1, logp_fun_0, logp_fun_1, sign=1, path=os.getcwd(), nheads=2, nlayers=2, keysize=2, epochs=1000, batchsize=2048, lr=0.01):

    """
    INPUT:
        rng: random key
        
        X0: An array composed of samples drawn from a probability density p0(x),
            formatted as (datasize, n, dim).
        
        X1: An array composed of samples drawn from a probability density p1(x),
            formatted as (datasize, n, dim).
        
        logp_fun_0: A function that calculates the log probability corresponding to a sample. 
                    
                    INPUT:  x, n, dim
                            
                            where x is a sample array with the format (n, dim),
                            and n and dim represent the number of particles and 
                            the spatial dimensions of a sample x, respectively. 
                    
                    OUTPUT: the value of log probability of x
                            
                            which can be a scalar or a scalar array.
        
        logp_fun_1: Same as above.

        sign: Select the upper and lower bounds for estimating free energy difference.
            1 for estimating the upper bound, -1 for estimating the lower bound.

        path: Path to save training parameters.

    OUTPUT:
        fediff: Estimated bounds of the free energy difference.
        
        fediff_err: Corresponding statistical error.
    """

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
        print(f"Train size: {train_size}, Validation size: {val_size}")
    
    print("\n============ Constructing transformer network ============")
    n = X0.shape[1]
    dim = X0.shape[2]
    params, vec_field_net = make_transformer(init_rng, n, dim, nheads, nlayers, keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (nlayers, nheads, keysize)
    print(f"\nModel Name: {modelname}")
    
    """initializing the sampler and logp calculator"""
    sample_and_logp_fn = make_flow(vec_field_net, X0, X1)
    free_energy = make_free_energy(sample_and_logp_fn, logp_fun_0, logp_fun_1, n, dim)
    
    """initializing the loss function"""
    loss = make_loss(vec_field_net)
    value_and_grad = jax.value_and_grad(loss)

    ####################################################################################

    print("\n======================== Training ========================")
    hyperparams = (epochs, iterations, batchsize)
    
    training_data = (X0[0:train_size], X1[0:train_size])
    validation_data = (X0[train_size:train_size+val_size], X1[train_size:train_size+val_size])

    params = train_and_evaluate(rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, lr, path)

    ####################################################################################

    print("\n========== Sampling and calculating free energy ==========")

    fe_rng, rng = jax.random.split(rng)

    fediff, fediff_err, f = free_energy(fe_rng, params, batchsize, sign)

    return fediff, fediff_err