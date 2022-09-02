import jax
from jax import numpy as jnp


def activate_forward(inp: jnp.ndarray) -> jnp.ndarray:
    return inp * activate_grad(inp)


def activate_grad(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(inp < 0, 0.01, 1)


def activate(inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(x: jnp.ndarray):
        return activate_forward(x), lambda dy: dy * activate_grad(x)

    return _fn(inp)
