import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src.main import EM

if __name__ == "__main__":

    n_groups = 3
    n = 500

    pi_true = jnp.ones(n_groups) / n_groups
    mean_true = jnp.array([1, 0, 2])
    sigma_true = jnp.array([1, 1, 1])/3

    rng = jax.random.PRNGKey(42)
    rng_Z, rng_X = jax.random.split(rng)
    Z = jax.random.choice(rng_Z, jnp.arange(n_groups), p=pi_true, shape=(n,))
    X_all = mean_true + jax.random.normal(rng_X, shape=(n, n_groups)) * sigma_true
    X = jnp.take_along_axis(X_all, indices=Z[:, jnp.newaxis], axis=1).squeeze()

    def init_fn(n_groups, rng):
        # return jax.random.uniform(rng, minval=0, maxval=1, shape=(n_groups, 2))
        rng_1, rng_2 = jax.random.split(rng)
        return jnp.c_[
            jax.random.uniform(rng_1, minval=0, maxval=5, shape=(n_groups, 1)),
            jax.random.uniform(rng_2, minval=0.1, maxval=1, shape=(n_groups, 1))
        ]


    def loglike_fn(theta, data):
        X = data[0]
        loglikes = jax.scipy.stats.norm.logpdf(X[:, jnp.newaxis], jnp.take(theta, 0, axis=1), jnp.take(theta, 1, axis=1))
        return loglikes


    rng = jax.random.PRNGKey(42)
    data = tuple([X])

    # theta = init_fn(n_groups, rng)
    # loglikes = loglike_fn(theta, data)
    # post = jnp.exp(loglikes - jax.scipy.special.logsumexp(loglikes, axis=1)[:, jnp.newaxis])

    em = EM(init_fn, loglike_fn, max_iters=10000000, tol=1e-3)
    # em.fit(rng, n_groups, data, init_fn, loglike_fn)
    runner, _ = em.fit(rng, n_groups, data)
    pj = np.asarray(runner[0])
    theta = np.asarray(runner[1])


    fig = plt.figure()
    colors = ["b", "r", "g"]
    x_grid = np.linspace(-4, 4, 1_000)
    y_sum = np.zeros_like(x_grid)
    for i in range(n_groups):
        y = stats.norm.pdf(x_grid, mean_true[i], sigma_true[i])
        y_sum += y * pi_true[i]
        plt.plot(x_grid, y, c=colors[i])
    plt.plot(x_grid, y_sum, c="k")

    for i in range(n_groups):
        y = stats.norm.pdf(x_grid, theta[i, 0], theta[i, 1])
        plt.plot(x_grid, y, c=colors[i], linestyle="--")

