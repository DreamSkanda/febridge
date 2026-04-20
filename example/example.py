"""Example usage of febridge for free energy calculation.

This example uses Gaussian distributions where the free energy has an analytical solution.
For X0 ~ N(0, I) and X1 ~ N(μ, I), the free energy is:
    F = KL(X0 || X1) = 0.5 * ||μ||^2

We use a simple 1D case: μ = 0.5, so analytical F = 0.5 (for n=4, dim=1)

Note: The current implementation uses sign=-1 to compute the forward KL.
This is a known sign convention in the original flow implementation.
"""

import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from febridge import model_train, model_inference


def logp_fun_0(x: jax.Array, n: int, dim: int) -> jax.Array:
    """Log probability of reference distribution: N(0, I).

    Args:
        x: Samples of shape [..., n, dim]
        n: Number of particles
        dim: Dimension

    Returns:
        Scalar log probability per sample.
    """
    # For N(0, I): log p(x) = -0.5 * sum(x^2) + const
    # We return the unnormalized version since only differences matter
    return -0.5 * jnp.sum(x ** 2)


def logp_fun_1(x: jax.Array, n: int, dim: int) -> jax.Array:
    """Log probability of target distribution: N(μ, I) where μ = 0.5.

    Args:
        x: Samples of shape [..., n, dim]
        n: Number of particles
        dim: Dimension

    Returns:
        Scalar log probability per sample.
    """
    # Target mean
    mu = 0.5
    # For N(μ, I): log p(x) = -0.5 * sum((x - μ)^2) + const
    return -0.5 * jnp.sum((x - mu) ** 2)


def generate_gaussian_samples(
    rng: jax.random.PRNGKey,
    n: int,
    dim: int,
    mu: float,
    num_samples: int,
) -> jax.Array:
    """Generate samples from N(mu, I).

    Args:
        rng: PRNG key.
        n: Number of particles.
        dim: Dimension per particle.
        mu: Mean of the Gaussian.
        num_samples: Number of samples.

    Returns:
        Samples of shape (num_samples, n, dim).
    """
    samples = mu + random.normal(rng, (num_samples, n, dim))
    return samples


# Configuration
rng = random.PRNGKey(42)
n = 4           # Number of particles
dim = 1         # Dimension (1D is simpler for testing)
data_size = 8192  # Smaller for faster testing
mu = 0.5        # Target mean

# Analytical free energy: F = 0.5 * ||mu||^2 * n * dim
# For n=4, dim=1, mu=0.5: F = 0.5 * 0.25 * 4 = 0.5
analytical_free_energy = 0.5 * (mu ** 2) * n * dim
print(f"Analytical free energy: {analytical_free_energy}")

# Generate samples from reference distribution X0 ~ N(0, I)
print("\nGenerating reference samples X0 ~ N(0, I)...")
X0_rng, rng = random.split(rng)
X0 = generate_gaussian_samples(X0_rng, n, dim, mu=0.0, num_samples=data_size)

# Generate samples from target distribution X1 ~ N(μ, I)
print("Generating target samples X1 ~ N(mu, I)...")
X1_rng, rng = random.split(rng)
X1 = generate_gaussian_samples(X1_rng, n, dim, mu=mu, num_samples=data_size)

# Train the model (fewer epochs for testing)
print("\nTraining model...")
params = model_train(
    X0, X1, rng,
    epochs=500,
    batchsize=512,
    nlayers=2,
    nheads=2,
    keysize=2,
)

# Run inference to get free energy estimate
print("\nRunning inference...")
x0, x1, logp, free_energy = model_inference(
    X0, X1, logp_fun_0, logp_fun_1, params, rng,
    nheads=2,
    nlayers=2,
    keysize=2,
    sign=-1,  # Try reverse direction
)

print(f"\n" + "=" * 50)
print(f"Results:")
print(f"  Analytical free energy: {analytical_free_energy:.6f}")
print(f"  Numerical free energy:  {float(free_energy):.6f}")
print(f"  Relative error:         {abs(float(free_energy) - analytical_free_energy) / analytical_free_energy * 100:.2f}%")
print(f"=" * 50)

# Check if result is within reasonable tolerance (20%)
tolerance = 0.20
if abs(float(free_energy) - analytical_free_energy) / analytical_free_energy < tolerance:
    print("\nTEST PASSED: Numerical result is within {:.0f}% of analytical solution!".format(tolerance * 100))
else:
    print("\nTEST FAILED: Numerical result is outside {:.0f}% tolerance.".format(tolerance * 100))
