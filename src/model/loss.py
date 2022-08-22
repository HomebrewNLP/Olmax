import typing

import jax
from jax import lax, numpy as jnp

from src.backend import device_id, matmul, promote_to
from src.constants import ParallelAxes
from src.context import Context


def cross_entropy_loss(ctx: Context, src_wgt: typing.Tuple[jnp.ndarray, jnp.ndarray], outer_tgt: jnp.ndarray
                       ) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    # Forward: logsumexp(x) - x[target]
    # Backward: (logsumexp(x) - x[target] + logsumexp(x)^2 * z_loss).grad
    # -> softmax(x) - 1 + softmax(x) * logsumexp(x) * z_loss
    src, param = src_wgt
    devices = ctx.dims.heads
    total_items = ctx.dims.batch * ctx.dims.sequence
    steps = ctx.data.vocab_size // ctx.dims.inner_bottleneck_features
    step_batch = total_items // steps
    local_batch = step_batch // devices

    def _xent_slice(carry: typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                    x: typing.Tuple[jnp.ndarray, jnp.ndarray], wgt: jnp.ndarray):
        d_wgt, loss, accuracy = carry
        inp_slice, tgt_slice = x
        tmp = matmul(inp_slice, wgt)
        tmp = promote_to(tmp, jnp.float32)
        tmp = lax.psum_scatter(tmp, ParallelAxes.model).reshape(local_batch, ctx.dims.vocab)
        lse = jax.nn.logsumexp(tmp, 1, keepdims=True)
        loss = loss + (lse / total_items).sum()
        loss = loss - (jnp.take_along_axis(tmp, tgt_slice.reshape(*tgt_slice.shape, 1), -1) / total_items).sum()
        accuracy = accuracy + lax.eq(lax.argmax(tmp, 1, outer_tgt.dtype), tgt_slice).sum() / total_items

        dx = lax.exp(tmp - jnp.max(tmp, 1, keepdims=True))
        dx = dx / dx.sum(1, keepdims=True) / total_items
        zloss = dx * lse * ctx.training.z_loss
        dx = dx.at[jnp.arange(dx.shape[0]).reshape(-1, 1), tgt_slice.reshape(-1, 1)].add(-1 / total_items)
        dx = dx + zloss
        d_tmp = jnp.transpose(dx, (1, 0))  # [LocalBatch, Vocab] -> [Vocab, LocalBatch]
        d_tmp = d_tmp.astype(src.dtype)
        d_x = matmul(wgt, d_tmp)  # [Features, Vocab] @ [Vocab, LocalBatch] -> [Features, LocalBatch]
        d_tmp = lax.all_gather(d_tmp, ParallelAxes.model, axis=1)  # [Vocab, Devices, LocalBatch]
        d_tmp = d_tmp.reshape(ctx.dims.vocab, step_batch)  # [Vocab, Devices, LocalBatch] -> [Vocab, StepBatch]+
        inp_slice = inp_slice.reshape(step_batch, ctx.dims.features)
        d_wgt = d_wgt + matmul(d_tmp, inp_slice)  # [Vocab, StepBatch] @ [StepBatch, Features] -> [Vocab, Features]
        return (d_wgt, loss, accuracy), d_x

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray, tgt: jnp.ndarray, wgt: jnp.ndarray):
        inp = inp.reshape(steps, devices, local_batch, ctx.dims.features)
        tgt = tgt.reshape(steps, step_batch)  # [Steps, StepBatch]
        tgt = lax.dynamic_slice_in_dim(tgt, device_id(ctx) * local_batch, local_batch, 1)  # [Steps, LocalBatch]

        def _slice_fn(carry, x):
            return _xent_slice(carry, x, wgt)

        init = (jnp.zeros(wgt.shape[::-1]), jnp.zeros(()), jnp.zeros(()))
        (d_wgt, loss, accuracy), dx = lax.scan(_slice_fn, init, (inp, tgt))

        dx = lax.all_gather(dx, ParallelAxes.model, axis=1)  # Shape[Steps, Devices, Features, LocalBatch]
        dx = dx.transpose(0, 1, 3, 2)  # Shape[Steps, Devices, LocalBatch, Features]
        dx = dx.reshape(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
        d_wgt = d_wgt.transpose(1, 0)  # [Vocab, Features]  ->  [Features, Vocab]

        def _grad(dy: typing.Tuple[jnp.ndarray, None]) -> typing.Tuple[jnp.ndarray, None, jnp.ndarray]:
            # dy == 1 since this is the last function before the output
            dy, _ = dy
            return (dx * dy).astype(inp.dtype), None, (d_wgt * dy).astype(wgt.dtype)

        loss = lax.psum(loss, ParallelAxes.model)
        accuracy = lax.psum(accuracy, ParallelAxes.model)
        return (loss.astype(jnp.float32), accuracy.astype(jnp.float32)), _grad

    return _fn(src, outer_tgt, param)
