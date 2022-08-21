import typing

import jax
from jax import lax, numpy as jnp

from src.backend import device_id, matmul, promote_to
from src.constants import ParallelAxes
from src.context import Context


def cross_entropy_loss(ctx: Context, src_wgt: typing.Tuple[jnp.ndarray, jnp.ndarray], tgt: jnp.ndarray) -> typing.Tuple[
    jnp.ndarray, jnp.ndarray]:
    # Forward: logsumexp(x) - x[target]
    # Backward: (logsumexp(x) - x[target] + logsumexp(x)^2 * z_loss).grad
    # -> softmax(x) - 1 + softmax(x) * logsumexp(x) * z_loss
    src, param = src_wgt
    devices = ctx.dims.heads

    def _xent_slice(inp: typing.Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], carry):
        inp, i, wgt, inner_tgt, d_wgt, loss, accuracy = inp
        index = device_id(ctx)
        inp_slice = inp[i]
        tmp = matmul(inp_slice, wgt).reshape(devices, -1, ctx.dims.vocab)
        tmp = promote_to(tmp, jnp.float32)
        tmp = lax.psum_scatter(tmp, ParallelAxes.model).reshape(-1, ctx.dims.vocab)
        tgt_slice = lax.dynamic_slice_in_dim(inner_tgt[i], index * tmp.shape[0], tmp.shape[0])
        lse = jax.nn.logsumexp(tmp, 1, keepdims=True)
        loss = loss + (lse - jnp.take_along_axis(tmp, tgt_slice.reshape(*tgt_slice.shape, 1), -1)).sum() / tgt.size
        accuracy = accuracy + lax.eq(lax.argmax(tmp, 1, tgt.dtype), tgt_slice).sum() / tgt.size

        dx = lax.exp(tmp - lse)
        zloss = dx * lse * ctx.training.z_loss
        dx -= ctx.training.label_smoothing / ctx.dims.vocab
        label = ctx.training.label_smoothing * (ctx.dims.vocab - 1) / ctx.dims.vocab - 1  # div so it sums to `LS - 1`
        dx = dx.at[jnp.arange(dx.shape[0]).reshape(-1, 1), tgt_slice.reshape(-1, 1)].add(label)
        dx = dx + zloss
        d_tmp = jnp.transpose(dx, (1, 0)) / tgt.size
        d_tmp = d_tmp.astype(inp_slice.dtype)
        d_x = matmul(wgt, d_tmp)  # [Features, Vocab] @ [Vocab, Batch] -> [Features, Batch]
        d_tmp = lax.all_gather(d_tmp, ParallelAxes.model, axis=1).reshape(ctx.dims.vocab, -1)
        d_wgt = d_wgt + matmul(d_tmp, inp_slice)  # [Vocab, Batch] @ [Batch, Features] -> [Vocab, Features]
        return (inp, i + 1, wgt, inner_tgt, d_wgt, loss, accuracy), d_x

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray, inner_tgt: jnp.ndarray, wgt: jnp.ndarray):
        original_shape = inp.shape
        inp = inp.reshape(ctx.data.vocab_size // ctx.dims.inner_bottleneck_features, -1, ctx.dims.features)
        inner_tgt = inner_tgt.reshape(ctx.data.vocab_size // ctx.dims.inner_bottleneck_features, -1)
        (_, _, _, _, d_wgt, loss, accuracy), dx = lax.scan(_xent_slice, (
                inp, jnp.zeros((), dtype=jnp.int32), wgt, inner_tgt,
                jnp.zeros(wgt.shape[::-1], dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.float32)), None, inp.shape[0])
        dx = dx.transpose(1, 0, 2) / tgt.size  # Shape[Features, inp.shape[0] // step, step // devices]
        dx = lax.all_gather(dx, ParallelAxes.model, axis=2).reshape(ctx.dims.features, -1).transpose(1, 0)
        dx = dx.reshape(original_shape)
        d_wgt = d_wgt.transpose(1, 0)
        d_wgt = d_wgt.reshape(param.shape)

        def _grad(dy: typing.Tuple[jnp.ndarray, None]) -> typing.Tuple[jnp.ndarray, None, jnp.ndarray]:
            # dy == 1 since this is the last function before the output
            return dx, None, d_wgt

        loss = lax.psum(loss, ParallelAxes.model)
        accuracy = lax.psum(accuracy, ParallelAxes.model)
        return (loss, accuracy), _grad

    return _fn(src, tgt, param)
