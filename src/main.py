import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm, loop_tqdm
from flax.training.train_state import TrainState
from dataclasses import dataclass
from functools import partial
import numpy as np


class TrainState(TrainState):
    obj_keeper: jnp.float32
    grads_keeper: jnp.array
    converged: jnp.bool_
    convergence_epoch: jnp.int32

@dataclass
class FitResults:
    theta: np.array
    converged: bool
    convergence_epoch: int
    objective_value: float
    grads: np.array


class EM:

    def __init__(self, init_fn, loglike_fn, max_iters=int(1e5), tol=1e-6):
        self.init_fn = init_fn
        self.loglike_fn = loglike_fn
        self.max_iters = max_iters
        self.tol = tol

    def _init_params(self, n_groups, rng):
        return self.init_fn(n_groups, rng)

    @partial(jax.jit, static_argnums=(0, ))
    def E_step(self, pj, theta, data):
        loglikes = self.loglike_fn(theta, data)
        log_post = loglikes + jnp.log(pj + 1e-16)
        log_post -= jax.scipy.special.logsumexp(log_post, axis=1)[:, jnp.newaxis]
        post = jnp.exp(log_post)
        return post

    @partial(jax.jit, static_argnums=(0, ))
    def M_step(self, data, post):
        X = data[0]
        nj = post.sum(axis=0)
        mus = X.dot(post) / nj
        sse = (X[:, jnp.newaxis] - mus) ** 2
        sigmas = jnp.sqrt(jnp.sum(post*sse, axis=0) / nj)
        theta = jnp.c_[mus.T, sigmas.T]
        return theta

    @jax.block_until_ready
    @partial(jax.jit, static_argnums=(0, ))
    def _run(self, runner, epoch):
        pj, theta, data, loglike_old, _ = runner
        post = self.E_step(pj, theta, data)
        loglike = jnp.sum(post * pj)
        converged = jnp.less(jnp.abs(loglike - loglike_old), self.tol)
        theta = self.M_step(data, post)
        pj = post.mean(axis=0)
        runner = (pj, theta, data, loglike, converged)
        return runner, {}

    def fit(self, rng, n_groups, data):

        rng_pj, rng_theta = jax.random.split(rng)
        pj = jax.random.dirichlet(rng_pj, alpha=jnp.ones(n_groups))
        theta = self._init_params(n_groups, rng_theta)

        step_runner = (pj, theta, data, jnp.inf, jnp.array(False, dtype=jnp.bool_))
        step_runner, metrics = lax.scan(
            scan_tqdm(self.max_iters)(self._run),
            step_runner,
            jnp.arange(self.max_iters),
            self.max_iters
        )

        # for i in range(max_iters):
        #     step_runner, _ = self._run(step_runner, i)

        metrics = {}
        return step_runner, metrics


