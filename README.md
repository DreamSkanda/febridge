# Febridge

Free energy difference calculation using neural network flows.

## Installation

```bash
pip install -e .
```

## Theory

Febridge computes free energy differences between reference (X₀) and target (X₁) distributions using continuous normalizing flows. The model learns a vector field that transforms samples from X₀ to X₁, then computes the free energy bound via:

```
F = E[log p₁(x₁) - log p₀(x₀) + log_det_J]
```

For Gaussian distributions, the analytical solution is `F = 0.5 * ||μ||² * n * dim`.

## API Reference

### High-Level API

```python
from febridge import model_train, model_inference
```

#### `model_train(X0, X1, rng, *, path, nheads, nlayers, keysize, epochs, batchsize, lr)`

Train the vector field network for free energy calculation.

**Parameters:**
- `X0` (ArrayLike): Reference distribution samples, shape `[num_samples, n, dim]`
- `X1` (ArrayLike): Target distribution samples, shape `[num_samples, n, dim]`
- `rng` (jax.random.PRNGKey): PRNG key for initialization
- `path` (str, optional): Directory for checkpoints and logs. Default: current working directory
- `nheads` (int, optional): Number of attention heads. Default: 2
- `nlayers` (int, optional): Number of transformer layers. Default: 2
- `keysize` (int, optional): Key size for attention. Default: 2
- `epochs` (int, optional): Number of training epochs. Default: 1000
- `batchsize` (int, optional): Training batch size. Default: 2048
- `lr` (float, optional): Learning rate. Default: 0.01

**Returns:** `hk.Params` — Trained model parameters

**Raises:**
- `SampleSizeError`: If sample set is smaller than batch size
- `SizeMismatchError`: If X0 and X1 have different sizes

---

#### `model_inference(X0, X1, logp_fun_0, logp_fun_1, params, rng, *, sign, nheads, nlayers, keysize, batchsize)`

Run inference to compute free energy.

**Parameters:**
- `X0` (ArrayLike): Reference distribution samples, shape `[num_samples, n, dim]`
- `X1` (ArrayLike): Target distribution samples, shape `[num_samples, n, dim]`
- `logp_fun_0` (Callable): Log probability function for reference distribution with signature `(x, n, dim) -> scalar`
- `logp_fun_1` (Callable): Log probability function for target distribution with signature `(x, n, dim) -> scalar`
- `params` (hk.Params): Trained model parameters
- `rng` (jax.random.PRNGKey): PRNG key
- `sign` (int, optional): 1 for upper bound, -1 for lower bound. Default: 1
- `nheads` (int, optional): Number of attention heads (must match training). Default: 2
- `nlayers` (int, optional): Number of transformer layers (must match training). Default: 2
- `keysize` (int, optional): Key size for attention (must match training). Default: 2
- `batchsize` (int, optional): Inference batch size. Default: 2048

**Returns:** `Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]` — (x0, x1, logp, free_energy)

**Raises:**
- `SampleSizeError`: If sample set is smaller than batch size
- `SizeMismatchError`: If X0 and X1 have different sizes

---

### Low-Level API

```python
from febridge import make_transformer, make_flow, make_loss, make_free_energy, train_and_evaluate
```

#### `make_transformer(key, n, dim, num_heads, num_layers, key_sizes)`

Creates a transformer network for vector field prediction.

**Parameters:**
- `key` (jax.random.PRNGKey): PRNG key for initialization
- `n` (int): Sequence length (number of particles)
- `dim` (int): Embedding dimension
- `num_heads` (int): Number of attention heads
- `num_layers` (int): Number of transformer layers
- `key_sizes` (int): Key size for attention

**Returns:** `Tuple[hk.Params, Callable]` — (params, apply_fn) where apply_fn has signature `(params, x, t) -> output`

---

#### `make_flow(vec_field_net, X0, X1, mxstep)`

Creates an ODE-based flow sampler with logp computation.

**Parameters:**
- `vec_field_net` (Callable): Vector field network with signature `(params, x, t) -> vector`
- `X0` (jnp.ndarray): Reference samples, shape `[num_samples, n*dim]`
- `X1` (jnp.ndarray): Target samples, shape `[num_samples, n*dim]`
- `mxstep` (int, optional): Maximum ODE integration steps. Default: 1000

**Returns:** `sample_and_logp_fn` with signature `(key, params, batchsize, sign) -> (x0, x1, logp)`

---

#### `make_loss(vec_field_net)`

Creates a vector field matching loss function.

The loss computes mean squared error between the true displacement (x1 - x0) and the predicted vector field.

**Parameters:**
- `vec_field_net` (Callable): Vector field network with signature `(params, x, t) -> vector`. Expected to take x of shape `(n*dim,)` and return `(n*dim,)`.

**Returns:** Loss function with signature `(params, x0, x1, t) -> loss float`

---

#### `make_free_energy(batched_sampler, logp_fun_0, logp_fun_1, n, dim)`

Creates a free energy bound calculator.

**Parameters:**
- `batched_sampler` (Callable): Sampler with signature `(rng, params, batchsize, sign) -> (x0, x1, logp)`
- `logp_fun_0` (Callable): Log probability function for reference distribution
- `logp_fun_1` (Callable): Log probability function for target distribution
- `n` (int): Sequence length
- `dim` (int): Dimension per particle

**Returns:** `free_energy_bound` function with signature `(rng, params, batchsize, sign) -> (mean, std, samples)`

---

#### `train_and_evaluate(rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, lr, path)`

Training loop with periodic validation.

**Parameters:**
- `rng` (jax.random.PRNGKey): PRNG key
- `loss` (Callable): Loss function with signature `(params, x0, x1, t) -> loss`
- `value_and_grad` (Callable): Function with signature `(params, x0, x1, t) -> (value, grad)`
- `hyperparams` (Tuple[int, int, int]): Tuple of (num_epochs, num_iterations, batchsize)
- `params` (hk.Params): Initial model parameters
- `training_data` (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of (X0_train, X1_train)
- `validation_data` (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of (X0_val, X1_val)
- `lr` (float): Learning rate
- `path` (str): Directory to save checkpoints

**Returns:** Trained model parameters `hk.Params`

---

### Exceptions

```python
from febridge import FebridgeError, SampleSizeError, SizeMismatchError
```

- `FebridgeError`: Base exception class
- `SampleSizeError`: Raised when sample set is smaller than batch size
- `SizeMismatchError`: Raised when X0 and X1 have different sizes

---

## Example

See `example/example.py` for a complete usage example with Gaussian distributions:

```bash
python example/example.py
```

The example computes the free energy difference between X₀ ~ N(0, I) and X₁ ~ N(0.5, I), comparing the numerical result against the analytical solution F = 0.5.