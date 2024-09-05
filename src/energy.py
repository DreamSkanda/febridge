import jax.numpy as jnp
from jax import vmap
from functools import partial

def make_free_energy(batched_sampler, energy_fun_0, energy_fun_1, n, dim):

    def free_energy_bound(rng, params, batchsize, sign):
        '''
        upper bound sign = 1
        lower bound sign = -1
        '''
        x0, x1, logp = batched_sampler(rng, params, batchsize, sign)
        e0 = jax.vmap(energy_fun_0, (0, None, None))(x0.reshape(batchsize, n, dim), n, dim)
        e1 = jax.vmap(energy_fun_1, (0, None, None))(x1.reshape(batchsize, n, dim), n, dim)
        
        f = e1 - e0 + logp/beta

        return f.mean(), f.std()/jnp.sqrt(batchsize), f
    
    return free_energy
