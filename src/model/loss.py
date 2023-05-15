import math
from typing import Tuple

import jax
from jax import lax, numpy as jnp

from src.backend import device_id, matmul, promote_to, with_context, get_param, SIX_ARRAYS, dot
from src.constants import ParallelAxes
from src.context import Context
from src.model.norm import norm_forward, norm_backward


@with_context()
def loss_fn(ctx: Context, src: jax.Array, tgt: jax.Array) -> jax.Array:
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

    def _xent_slice(carry: Tuple[jax.Array, jax.Array, jax.Array], x: Tuple[jax.Array, jax.Array],
                    wgt: jax.Array):
        d_wgt, loss, acc = carry
        inp_slice, tgt_slice = x
        tmp, lse = _input(x, wgt)

        loss = loss + (lse / total_items).sum()
        loss = loss - (jnp.take_along_axis(tmp, tgt_slice.reshape(*tgt_slice.shape, 1), -1) / total_items).sum()
        acc = acc + lax.eq(lax.argmax(tmp, 1, jnp.int32), tgt_slice).sum() / total_items

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
        return (d_wgt, loss.astype(jnp.float64), acc.astype(jnp.float64)), dx

    tgt = tgt.reshape(steps, step_batch)  # [Steps, StepBatch]
    tgt = lax.dynamic_slice_in_dim(tgt, device_id() * local_batch, local_batch, 1)  # [Steps, LocalBatch]

    @jax.custom_gradient
    def _fn(inp: jax.Array, wgt: jax.Array):
        def _slice_fn(carry, x):
            x, tgt_slice = x
            out, norm_out, multiplied, src_fp64, std = norm_forward(ctx, x, 1, False, x.ndim - 1, False)
            out = out.reshape(devices, local_batch, ctx.dims.features)
            carry, dx = _xent_slice(carry, (out, tgt_slice), wgt)
            dx, _ = norm_backward(x, 1, std, dx.reshape(x.shape), False, x.ndim - 1, False, (), src_fp64.dtype)
            return carry, dx

        carry = jnp.zeros_like(wgt), jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64)
        scan_inp = inp.reshape(steps, devices, local_batch, ctx.dims.features), tgt
        (dwgt, loss, acc), dx = lax.scan(_slice_fn, carry, scan_inp)
        dx = dx.reshape(*inp.shape).astype(inp.dtype)
        dwgt = dwgt.astype(wgt.dtype)

        def _grad(dy: Tuple[SIX_ARRAYS, jax.Array, jax.Array]) -> Tuple[SIX_ARRAYS, jax.Array]:
            # dy == 1 since this is the last function before the output
            dy, _ = dy
            return dx * dy.astype(inp.dtype), dwgt * dy.astype(wgt.dtype)

        return jnp.stack([loss, acc]), _grad

    return _fn(src, param)
