from typing import Tuple

import jax
from jax import numpy as jnp, lax

from src.backend import dot, get_param, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.norm import scale_norm_act, scale_norm_act_linear


@with_context()
def linear(ctx: Context, inp: jax.Array, in_features: int, out_features: int):
    weight = get_param(ctx, "conv_weight", [out_features, in_features])
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (out_features,), dtype=inp.dtype)

    return dot(inp, weight, -1, -1)


def all2all(inp):
    out = inp.reshape(*inp.shape[:-1], jax.device_count(), 1, -1)
    out = lax.all_to_all(out, ParallelAxes.model, out.ndim - 3, out.ndim - 2, tiled=False)
    return out.reshape(*inp.shape)


@with_context()
def input_embed(ctx: Context, inp: jax.Array, dim: int) -> jax.Array:
    param = get_param(ctx, "inp_embd", [dim, ctx.dims.pointwise_features], std=1 / ctx.dims.pointwise_features)
    return jnp.take(param, inp, 0)


@with_context()
def pos_and_scale(ctx: Context, gates: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # from https://github.com/HomebrewNLP/HomebrewNLP-MTF/blob/v1.16.0/src/model/basic.py#L93
    gate_sqrt = int(ctx.dims.memory_slots ** 0.5)
    assert gate_sqrt ** 2 == ctx.dims.memory_slots

    gates = gates.reshape(ctx.dims.batch, ctx.dims.memory_heads, 2, gate_sqrt)
    gates = scale_norm_act(ctx, gates, gate_sqrt, act=False)
    gates -= lax.stop_gradient(gates.max(-1, keepdims=True).sum(-2, keepdims=True))
    gates = lax.exp(gates)
    denominator = lax.reciprocal(gates.sum(-1, keepdims=True))
    values, idx = lax.top_k(gates, ctx.dims.memory_slots_per_head)  # along last axis
    values = values * denominator

    if ctx.dims.memory_slots_per_head == 1:
        idx = idx[:, :, 0] + idx[:, :, 1] * gate_sqrt
        values = values.prod(-2, keepdims=True)
    else:
        idx = idx[:, :, 0, :, None] + idx[:, :, 1, None, :] * gate_sqrt
        values = values[:, :, 0, :, None] + values[:, :, 1, None, :]
        idx = idx.reshape(ctx.dims.batch, ctx.dims.memory_heads, ctx.dims.memory_slots_per_head ** 2)
        values = values.reshape(ctx.dims.batch, ctx.dims.memory_heads, ctx.dims.memory_slots_per_head ** 2)
        values, new_idx = lax.top_k(values, ctx.dims.memory_slots_per_head)
        idx = jnp.take_along_axis(idx, new_idx, 2)

    # [Batch Slots MemoryFeatures] [Batch Heads TopK] -> [Batch, Heads * TopK, MemoryFeatures]
    return idx.reshape(ctx.dims.batch, -1), values.reshape(ctx.dims.batch, -1, 1)


@with_context()
def input_fn(ctx: Context, token: jax.Array, position: jax.Array, dense: jax.Array, *out: int) -> Tuple[jax.Array, ...]:
    token_embedding = input_embed(ctx, token, ctx.dims.vocab)
    position_embedding = input_embed(ctx, position, ctx.dims.sequence)
    dense = scale_norm_act_linear(ctx, dense, ctx.dims.features, ctx.dims.pointwise_features)
    return scale_norm_act_linear(ctx, token_embedding + position_embedding + dense, ctx.dims.pointwise_features,
                                 list(out), transform_fns=[all2all])


@with_context()
def _output(ctx: Context, inp: jax.Array, offset: jax.Array) -> Tuple[jax.Array, jax.Array]:
    out, scale = scale_norm_act_linear(ctx, inp, ctx.dims.pointwise_features, [ctx.dims.features, ctx.dims.features])
    out = scale_norm_act(ctx, out + offset, ctx.dims.features)
    return out, jax.nn.sigmoid(scale)


@with_context()
def read(ctx: Context, dense0: jax.Array, sparse: jax.Array, token: jax.Array, position: jax.Array
         ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    total_read = ctx.dims.memory_features * ctx.dims.memory_heads * ctx.dims.memory_slots_per_head
    gate_sqrt = int(ctx.dims.memory_slots ** 0.5)

    offset1, offset0, gates = input_fn(ctx, token, position, dense0, ctx.dims.features, ctx.dims.pointwise_features,
                                       gate_sqrt * 2 * ctx.dims.memory_heads)
    idx, val = pos_and_scale(ctx, gates)
    inp = (jnp.take_along_axis(sparse, idx.reshape(*idx.shape, 1), 1) * val).reshape(ctx.dims.batch, total_read)

    inp = linear(ctx, inp, total_read, ctx.dims.pointwise_features)
    out, scale = _output(ctx, inp + offset0, offset1)
    return out, scale, idx


@with_context()
def write(ctx: Context, dense1: jax.Array, token: jax.Array, position: jax.Array
          ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    total_read = ctx.dims.memory_features * ctx.dims.memory_heads * ctx.dims.memory_slots_per_head
    gate_sqrt = int(ctx.dims.memory_slots ** 0.5)

    offset0, offset1, offset2 = input_fn(ctx, token, position, dense1, ctx.dims.pointwise_features,
                                         ctx.dims.pointwise_features, ctx.dims.features)

    out = scale_norm_act_linear(ctx, offset0 + offset1, ctx.dims.pointwise_features,
                                [ctx.dims.pointwise_features, total_read, gate_sqrt * 2 * ctx.dims.memory_heads])
    dense0, scatter_values, gates = out
    idx, val = pos_and_scale(ctx, gates)
    scatter_values = scatter_values.reshape(ctx.dims.batch, -1, ctx.dims.memory_features) * val
    out, scale = _output(ctx, dense0, offset2)
    return out, scale, scatter_values, idx
