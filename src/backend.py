import typing

import jax._src.util as util
import numpy as np
from jax import lax, numpy as jnp, random

from .context import Context

INT_OR_TUPLE = typing.Union[int, typing.Sequence[int]]


def pos_dim(inp: jnp.ndarray, dims: typing.Sequence[int]) -> typing.Sequence[int]:
    return tuple([d % inp.ndim for d in dims])


def tuple_int(obj: INT_OR_TUPLE) -> typing.Sequence[int]:
    if isinstance(obj, (tuple, list)):
        return tuple(obj)
    if isinstance(obj, int):
        return obj,
    raise ValueError


def sum_pool(inputs: jnp.ndarray, window_shape: typing.List[int],
             padding: typing.List[typing.Tuple[int, int]]) -> jnp.ndarray:
    # TODO: Validate what's happening in the backend
    strides = (1,) * (len(window_shape) + 2)
    dims = (1,) + tuple(window_shape) + (1,)
    padding = ((0, 0),) + tuple(padding) + ((0, 0),)
    return lax.reduce_window(inputs, 0, lax.add, dims, strides, padding)


def conv(inp: jnp.ndarray, weight: jnp.ndarray, padding: typing.List[typing.Tuple[int, int]], groups: int):
    ndim = weight.ndim
    dimension_numbers = (0, ndim - 1) + tuple(range(1, ndim - 1))
    dimension_numbers = lax.ConvDimensionNumbers(dimension_numbers, tuple(range(ndim)), dimension_numbers)
    return lax.conv_general_dilated(inp, weight, (1,) * (ndim - 2), padding=padding, feature_group_count=groups,
                                    dimension_numbers=dimension_numbers, precision='fastest')


def dot(left: jnp.ndarray, right: jnp.ndarray, left_contract_dims: INT_OR_TUPLE, right_contract_dims: INT_OR_TUPLE,
        left_batch_dims: INT_OR_TUPLE = tuple(), right_batch_dims: INT_OR_TUPLE = tuple()) -> jnp.ndarray:
    dims = ((pos_dim(left, tuple_int(left_contract_dims)), pos_dim(right, tuple_int(right_contract_dims))),
            (pos_dim(left, tuple_int(left_batch_dims)), pos_dim(right, tuple_int(right_batch_dims))))
    return lax.dot_general(left, right, dims, "fastest")


def matmul(left: jnp.ndarray, right: jnp.ndarray, reduced_dims=1):
    return dot(left, right, tuple(range(-reduced_dims, 0)), tuple(range(reduced_dims)))


def dims_to_shape(ctx: Context, dims: typing.List[str]) -> typing.List[int]:
    return [ctx.dims.sizes[d] for d in dims]


def prefixed_name(ctx: Context, name: str):
    return ctx.add_to_prefix(name, count=False).global_prefix


def assign(ctx: Context, name: str, inp: jnp.ndarray):
    name = prefixed_name(ctx, name)
    ctx.parameters[name] = inp
    ctx.parameter_variance[name] = inp.var()


def orthogonal_init(ctx: Context, shape: typing.List[int], column_axes=(-1,)) -> jnp.ndarray:
    axes = tuple([shape[c] for c in column_axes])
    n_rows, n_cols = util.prod(shape) // util.prod(axes), util.prod(axes)
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, ctx.model.dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim)
    if n_rows < n_cols:
        out = out.T
    return jnp.reshape(out, tuple(np.delete(shape, column_axes)) + axes)


def get_param(ctx: Context, name: str, str_shape: typing.Optional[typing.List[str]] = None,
              std: typing.Optional[float] = None, mean: typing.Optional[float] = None,
              column_axes: int = 1, scale: float = 1.) -> jnp.ndarray:
    prefix_name = prefixed_name(ctx, name)
    shape = dims_to_shape(ctx, str_shape)
    if prefix_name not in ctx.parameters:
        ctx.parameter_dims[prefix_name] = str_shape
        if std is None and mean is None:
            if ctx.dims.depth in str_shape:
                del shape[str_shape.index(ctx.dims.depth)]
                param = jnp.stack([orthogonal_init(ctx, shape, range(len(shape) - column_axes, len(shape)))
                                   for _ in range(ctx.dims.sizes.depth)], str_shape.index(ctx.dims.depth))
            else:
                param = orthogonal_init(ctx, shape, range(len(shape) - column_axes, len(shape)))
            param *= scale
        else:
            param = random.normal(ctx.prng_key, shape, ctx.model.dtype)
            if std is not None:
                param *= std
            if mean is not None:
                param += mean
        param = param.astype(ctx.model.dtype)
        assign(ctx, name, param)
    param = ctx.parameters[prefix_name]
    return param


def zero_param(ctx: Context, name: str, shape: typing.List[str]) -> jnp.ndarray:
    return get_param(ctx, name, shape, 0, 0)
