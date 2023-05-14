import math
from typing import Tuple

import jax
from jax import lax, numpy as jnp

from src.backend import device_id, matmul, promote_to, with_context, get_param, SIX_ARRAYS, dot
from src.constants import ParallelAxes
from src.context import Context
from src.model.norm import norm_forward, norm_backward


@with_context()
def loss_fn(ctx: Context, src: SIX_ARRAYS, tgt: jax.Array) -> Tuple[SIX_ARRAYS, jax.Array]:
    # TODO: This forces ctx.dims.batch to be divisible by jax.device_count(). Dependence has to be removed.
    # Forward: logsumexp(x) - x[target]
    # Backward: (logsumexp(x) - x[target] + logsumexp(x)^2 * z_loss).grad
    # -> softmax(x) - one_hot(target) + softmax(x) * logsumexp(x) * z_loss
    devices = jax.device_count()
    total_items = ctx.dims.batch
    steps = 1
    step_batch = total_items
    local_batch = step_batch // devices

    param = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1, scale=1 / jax.device_count())

    if ctx.is_initializing:
        return src, jnp.zeros((2,))

    def _input(x: Tuple[jax.Array, jax.Array], wgt: jax.Array):
        inp_slice, tgt_slice = x
        tmp = matmul(inp_slice, wgt)
        tmp = promote_to(tmp, jnp.float32)
        tmp = lax.psum_scatter(tmp, ParallelAxes.model).reshape(local_batch, ctx.dims.vocab)
        lse = jax.nn.logsumexp(promote_to(tmp, jnp.float64), 1, keepdims=True)
        return tmp, lse

    def _xent_slice_loss(carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array], x: Tuple[jax.Array, jax.Array],
                         wgt: jax.Array):
        loss, acc = carry
        inp_slice, tgt_slice = x
        tmp, lse = _input(x, wgt)

        loss = loss + (lse / total_items).sum()
        loss = loss - (jnp.take_along_axis(tmp, tgt_slice.reshape(*tgt_slice.shape, 1), -1) / total_items).sum()
        acc = acc + lax.eq(lax.argmax(tmp, 1, jnp.int32), tgt_slice).sum() / total_items
        return (loss.astype(jnp.float64), acc.astype(jnp.float64)), None

    def _xent_slice_derivative(d_wgt: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                               x: Tuple[jax.Array, jax.Array], wgt: jax.Array):
        inp_slice, tgt_slice = x
        tmp, lse = _input(x, wgt)

        dy = lax.exp(tmp - (lse + math.log(total_items)))  # [LocalBatch, Vocab]
        zloss = dy * lse * ctx.training.z_loss * 2
        dy = dy.at[jnp.arange(local_batch).reshape(-1, 1), tgt_slice.reshape(-1, 1)].add(-1 / total_items)
        dy = dy + zloss
        dy = dy * jax.device_count()
        dy = dy.astype(ctx.model.computation_dtype)
        dy = lax.all_gather(dy, ParallelAxes.model)
        dy = dy.reshape(step_batch, ctx.dims.vocab)
        dx = dot(dy, wgt, 1, 1)  # [StepBatch, Vocab (Reduced)] x [Features, Vocab (Reduced)]  -> [StepBatch, Features]

        inp_slice = inp_slice.reshape(step_batch, ctx.dims.features)
        d_wgt = d_wgt + dot(inp_slice, dy, 0, 0)  # [StepBatch, Features] x [StepBatch, Vocab] -> [Features, Vocab]
        return d_wgt, dx

    @jax.custom_gradient
    def _fn(inp: SIX_ARRAYS, tgt: jax.Array, wgt: jax.Array):
        inp = inp[0] + inp[2]
        tgt = tgt.reshape(steps, step_batch)  # [Steps, StepBatch]
        tgt = lax.dynamic_slice_in_dim(tgt, device_id() * local_batch, local_batch, 1)  # [Steps, LocalBatch]

        def _slice_fn_loss(carry, x):
            x, tgt_slice = x
            out, _, _, _, _ = norm_forward(ctx, x, 1, False, x.ndim - 1, False)
            return _xent_slice_loss(carry, (out, tgt_slice), wgt)

        def _slice_fn_grad(carry, x):
            x, tgt_slice = x
            out, norm_out, multiplied, src_fp64, std = norm_forward(ctx, x, 1, False, x.ndim - 1, False)
            out = out.reshape(devices, local_batch, ctx.dims.features)
            dwgt, dx = _xent_slice_derivative(carry, (out, tgt_slice), wgt)
            dx, _ = norm_backward(x.reshape(dx.shape), 1, std, dx, False, x.ndim - 1, False, (), src_fp64.dtype)
            return dwgt, dx

        (loss, acc), _ = lax.scan(_slice_fn_loss, (jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64)),
                                  (inp.reshape(steps, devices, local_batch, ctx.dims.features), tgt))

        def _grad(dy: Tuple[jax.Array, jax.Array, jax.Array]):
            # dy == 1 since this is the last function before the output
            (pdx0, x0, pdx1, x1, dsp, sp), x, d_loss, d_acc = dy
            dwgt, dx = lax.scan(_slice_fn_grad, jnp.zeros_like(wgt),
                                (x.reshape(steps, step_batch, ctx.dims.features), tgt))
            dx = (dx.reshape(ctx.dims.batch, ctx.dims.features) * d_loss).astype(inp.dtype)
            dwgt = (dwgt * d_loss).astype(wgt.dtype)
            return (dx + pdx0, x0, dx + pdx1, x1, dsp, sp), None, dwgt

        return (src, loss, acc), _grad

    src, loss, acc = _fn(src, tgt, param)
    return src, jnp.stack([loss, acc])
