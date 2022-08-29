import math

import jax
import numpy as np
from jax import lax, numpy as jnp

from src.backend import promote_to, with_context
from src.context import Context
from src.model.conv import conv
from src.model.norm import prenorm, scale_norm_act


def qrnn(forget: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dtype = forget.dtype
    for offset in 2 ** np.arange(int(math.log2(x.shape[1]))):
        x = x.at[:, offset:].add(x[:, :-offset] * forget[:, offset:])
        forget = forget.at[:, offset:].mul(forget[:, :-offset])
    return x.astype(dtype)


def qrnn_grad(ctx: Context, forget: jnp.ndarray, src: jnp.ndarray) -> jnp.ndarray:
    if ctx.is_initializing:
        return src

    @jax.custom_gradient
    def _fn(fgt: jnp.ndarray, inp: jnp.ndarray):
        dtype = inp.dtype
        out = qrnn(jax.nn.hard_sigmoid(promote_to(fgt, jnp.float32)), promote_to(inp, jnp.float32))
        out = out.astype(dtype)

        def _grad(dy: jnp.ndarray):
            x = promote_to(inp, jnp.float32)
            f = jax.nn.hard_sigmoid(promote_to(fgt, jnp.float32))
            f = lax.rev(f, (1,))
            f = jnp.concatenate([jnp.ones((x.shape[0], 1, x.shape[2])), f[:, :-1]], 1)
            dy_rev = lax.rev(dy, (1,))
            dx = lax.rev(qrnn(f, dy_rev), (1,))
            df = dx * promote_to(out, jnp.float32)
            df = jnp.where(jnp.logical_or(fgt > 3, fgt < -3), 0, df / 6)
            df = df.astype(dtype)
            dx = dx.astype(dtype)
            return df, dx

        return out, _grad

    return _fn(forget, src)


@prenorm
@with_context()
def qrnn_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    # 500ms at 256 features (forward pass, backward takes slightly longer)
    # While conv 256->256 with kernel_size=5 takes ~11.3ms
    mid = conv(ctx, inp, ctx.dims.pointwise_kernel, ctx.optimizer.qrnn_scale, ctx.dims.features,
               ctx.dims.inner_bottleneck_features * 2)
    mid, forget = jnp.split(mid, 2, -1)
    out = qrnn_grad(ctx, forget, mid)
    out = scale_norm_act(ctx, out, ctx.dims.inner_bottleneck_features)
    return conv(ctx, out, ctx.dims.pointwise_kernel, ctx.optimizer.qrnn_scale, ctx.dims.inner_bottleneck_features,
                ctx.dims.features)
