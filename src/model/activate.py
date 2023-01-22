import jax
from jax import lax


# [Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992) showed that
# "complicated" activation functions such as SiLU, GeLU and Mish help increase the rank of the embedding matrix.
# To ensure low-redundancy features, we use a function from that family.
def activate_forward(inp: jax.Array) -> jax.Array:
    return inp * lax.tanh(lax.max(inp, 0) + lax.log1p(lax.exp(-lax.abs(inp))))  # jax.nn.softplus without nan check


def activate_grad(inp: jax.Array) -> jax.Array:
    tanh_sp = lax.tanh(lax.max(inp, 0) + lax.log1p(lax.exp(-lax.abs(inp))))
    return tanh_sp + inp * jax.nn.sigmoid() * (1 - lax.square(tanh_sp))


def activate(inp: jax.Array) -> jax.Array:
    @jax.custom_gradient
    def _fn(x: jax.Array):
        return activate_forward(x), lambda dy: dy * activate_grad(x)

    return _fn(inp)
