import math
from typing import Tuple, Callable

import jax
from jax import lax, numpy as jnp

from src.backend import device_id, matmul, promote_to
from src.constants import ParallelAxes
from src.context import Context


def cross_entropy_loss(ctx: Context) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    # TODO: This forces ctx.dims.batch to be divisible by jax.device_count(). Dependence has to be removed.
    # Forward: logsumexp(x) - x[target]
    # Backward: (logsumexp(x) - x[target] + logsumexp(x)^2 * z_loss).grad
    # -> softmax(x) - one_hot(target) + softmax(x) * logsumexp(x) * z_loss
    devices = jax.device_count()
    total_items = ctx.dims.batch
    steps = 1
    step_batch = total_items
    local_batch = step_batch // devices

    def _xent_slice(carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                    x: Tuple[jax.Array, jax.Array], wgt: jax.Array):
        d_wgt, loss, acc = carry
        inp_slice, tgt_slice = x
        tmp = matmul(inp_slice, wgt)
        tmp = promote_to(tmp, jnp.float32)
        tmp = lax.psum_scatter(tmp, ParallelAxes.model).reshape(local_batch, ctx.dims.vocab)
        lse = jax.nn.logsumexp(promote_to(tmp, jnp.float64), 1, keepdims=True)

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
        dy = dy.reshape(step_batch, ctx.dims.vocab).transpose(1, 0)
        dx = matmul(wgt, dy)  # [Features, Vocab] @ [Vocab, StepBatch] -> [Features, StepBatch]

        inp_slice = inp_slice.reshape(step_batch, ctx.dims.features)
        d_wgt = d_wgt + matmul(dy, inp_slice)  # [Vocab, StepBatch] @ [StepBatch, Features] -> [Vocab, Features]
        return (d_wgt, loss.astype(jnp.float64), acc.astype(jnp.float64)), dx

    @jax.custom_gradient
    def _fn(inp: jax.Array, tgt: jax.Array, wgt: jax.Array):
        inp = inp.reshape(steps, devices, local_batch, ctx.dims.features)
        tgt = tgt.reshape(steps, step_batch)  # [Steps, StepBatch]
        tgt = lax.dynamic_slice_in_dim(tgt, device_id() * local_batch, local_batch, 1)  # [Steps, LocalBatch]

        def _slice_fn(carry, x):
            return _xent_slice(carry, x, wgt)

        init = (jnp.zeros(wgt.shape[::-1]), jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64))
        (d_wgt, loss, acc), dx = lax.scan(_slice_fn, init, (inp, tgt))

        dx = dx.reshape(ctx.dims.batch, ctx.dims.features)
        d_wgt = d_wgt.transpose(1, 0)  # [Vocab, Features]  ->  [Features, Vocab]

        def _grad(dy: Tuple[jax.Array, None]) -> Tuple[jax.Array, None, jax.Array]:
            # dy == 1 since this is the last function before the output
            dy, _ = dy
            return (dx * dy).astype(inp.dtype), None, (d_wgt * dy).astype(wgt.dtype)

        return (loss, acc), _grad

    return _fn  # _fn(src, outer_tgt, param)
