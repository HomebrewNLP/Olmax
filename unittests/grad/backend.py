import random
import typing

import jax
from jax import numpy as jnp

from src.constants import ParallelAxes

trials = 8
sample_sizes = [2 ** 10, 2 ** 14]


def randn_fn():
    rng = random.Random(0)

    def _fn(*shape: int):
        seed = rng.randint(0, 2 ** 30)
        div = (shape[-1] * jax.device_count()) ** 0.25
        fn = jax.pmap(
                lambda x: jax.random.normal(jax.random.PRNGKey(x + seed), shape, jnp.float32).astype(jnp.float_) / div)
        local_devices = jax.local_device_count()
        seeds = jnp.arange(local_devices * jax.process_index(), local_devices * (1 + jax.process_index()))
        return fn(seeds)

    return _fn


def grad_fn(out_shape: typing.Iterable[int], *args):
    dy = randn_fn()(*out_shape)  # constant for given shape. calling grad_fn twice gives same thing

    def _fn(fn):
        return jax.pmap(jax.grad(lambda *x: (fn(*x) * dy).sum()), ParallelAxes.model)(args)

    return _fn
