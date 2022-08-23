import jax
from jax import lax, numpy as jnp

from src.backend import matmul
from src.constants import ParallelAxes
from src.context import Context
from src.model.loss import cross_entropy_loss


def main():
    ctx = Context()
    key = jax.random.PRNGKey(0)
    k0, k1, k2 = jax.random.split(key, 3)
    tgt = jax.random.randint(k0, (ctx.dims.batch, ctx.dims.sequence), 0, ctx.dims.vocab)
    tgt = jax.device_put_replicated(tgt, jax.local_devices())
    src = jax.random.normal(k1, (ctx.dims.heads, ctx.dims.batch, ctx.dims.sequence, ctx.dims.features))
    wgt = jax.random.normal(k2, (ctx.dims.heads, ctx.dims.vocab, ctx.dims.features)) / ctx.dims.features
    grad0 = jax.pmap(lambda x, y: jax.grad(lambda x, y: cross_entropy_loss(ctx, x, y), has_aux=True)(x, y)[0],
                     ParallelAxes.model, in_axes=((0, 0), 0))((src, wgt), tgt)

    def _loss(x, y):
        tmp = lax.psum(matmul(x[0], x[1]), ParallelAxes.model)
        pos = jax.nn.logsumexp(tmp, -1).mean()
        neg = jnp.take_along_axis(tmp.reshape(-1, ctx.dims.vocab), y.reshape(-1, 1), -1).mean()
        return pos - neg

    grad1 = jax.pmap(jax.grad(_loss), ParallelAxes.model, in_axes=((0, 0), 0))((src, wgt), tgt)
    for gr0, gr1 in zip(grad0, grad1):  # iterate over src and weight
        print("NEW TENSOR")
        if not jnp.allclose(gr0, gr1):
            for i, (g0, g1) in enumerate(zip(gr0.ravel(), gr0.ravel())):
                if not jnp.isclose(g0, g1):
                    print(i, g0, g1)


if __name__ == '__main__':
    main()
