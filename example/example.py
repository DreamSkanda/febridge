import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from mcmc import mcmc_fun

rng = random.PRNGKey(42)
n = 6
dim = 2
data_size = 102400

def logp_fun_0(x, n, dim):
    return norm.logpdf(x).sum()

def logp_fun_1(x, n, dim):
    i, j = jnp.triu_indices(n, k=1)
    r_ee = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(1/r_ee)
    return jnp.sum(x**2) + v_ee
mcmc_logp_fun = jax.vmap(logp_fun_1, in_axes=(0, None, None))

sample_rng, rng = random.split(rng)
def sampler(rng, n, dim, mc_epoch=20, mc_steps=100, mc_width=0.05):
    X0_rng, rng = random.split(rng)
    X = random.normal(X0_rng, (data_size, n*dim))
    X0 = X

    for _ in range(mc_epoch):
        mcmc_rng, rng = random.split(rng)
        X, acc = mcmc_fun(mcmc_rng, lambda X: -mcmc_logp_fun(X, n, dim), X, mc_steps, mc_width)
    X1 = X
    
    return X0, X1

X0, X1 = sampler(sample_rng, n, dim)

fe, fe_err = model_train(rng, X0, X1, logp_fun_0, logp_fun_1, )
print(fe, fe_err)

    