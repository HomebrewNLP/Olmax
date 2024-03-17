import random

import jax
from jax import numpy as jnp

from src.constants import ParallelAxes

trials = 1
sample_sizes = [2 ** 6]


def randn_fn():
    rng = random.Random(0)

    def _fn(*shape: int):
        seed = rng.randint(0, 2 ** 30)

        def _gen(x):
            return jax.random.normal(jax.random.PRNGKey(x + seed), shape, jnp.float32).astype(jnp.float64) / div

        div = (shape[-1] * jax.device_count()) ** 0.25
        devices = jax.local_device_count()
        return jax.pmap(_gen)(jnp.arange(devices * jax.process_index(), devices * (1 + jax.process_index())))

    return _fn


def grad_fn(dy: jax.Array, *args):
    def _fn(fn):
        return jax.pmap(jax.grad(lambda x, y: (fn(x) * y).sum()), ParallelAxes.model)(args, dy)

    return _fn
