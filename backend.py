import typing

import jax._src.util as util
import numpy as np
from jax import lax, numpy as jnp, random
from jax.experimental import PartitionSpec
from jax.experimental import pjit

from context import Context


def dims_to_shape(ctx: Context, dims: typing.List[str]) -> typing.List[int]:
    return [ctx.dims.sizes[d] for d in dims]


def is_intermediate(ctx, inp: jnp.ndarray) -> bool:
    return inp.shape[-1] != ctx.dims.sizes.features_per_head


def get_feature_dim(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    return ctx.dims.intermediate_feed_forward if is_intermediate(ctx, inp) else ctx.dims.features_per_head


def shard(tensor: jnp.ndarray, head: typing.Optional[int] = -2, batch: typing.Optional[int] = 0):
    spec: typing.List[typing.Optional[str]] = [None] * tensor.ndim
    if isinstance(batch, int):
        spec[batch] = "data_parallel"
    if isinstance(head, int):
        spec[head] = "model_parallel"
    try:
        return pjit.with_sharding_constraint(tensor, PartitionSpec(*spec))
    except ValueError as e:
        e_str = str(e)
        if ("One of with_sharding_constraint arguments was given the resource assignment of PartitionSpec(" in e_str and
                ", but resource axis " in e_str and "is undefined. Did you forget to declare the mesh?" in e_str):
            return tensor
        raise e


def orthogonal_init(ctx: Context, shape: typing.List[int], column_axes=(-1,)) -> jnp.ndarray:
    axes = tuple([shape[c] for c in column_axes])
    n_rows, n_cols = util.prod(shape) // util.prod(axes), util.prod(axes)
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, ctx.model.dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim)
    if n_rows < n_cols:
        out = out.T
    return jnp.reshape(out, tuple(np.delete(shape, column_axes)) + axes)


def default(value: typing.Any, default_value: typing.Any) -> typing.Any:
    return default_value if value is None else value


def get_param(ctx: Context, name: str, shape: typing.Optional[typing.List[str]] = None,
              std: typing.Optional[float] = None, mean: typing.Optional[float] = None,
              column_axes: typing.Sequence[int] = tuple(), scale: float = 1.) -> jnp.ndarray:
    name = ctx.add_to_prefix(name, count=False).global_prefix
    if name not in ctx.parameters:
        ctx.parameter_dims[name] = shape
        shape = dims_to_shape(ctx, shape)
        if std is None and mean is None:
            ctx.parameters[name] = orthogonal_init(ctx, shape, column_axes if column_axes else (-1,)) * scale
        else:
            ctx.parameters[name] = random.normal(ctx.prng_key, shape, ctx.model.dtype)
            if std is not None:
                ctx.parameters[name] *= std
            if mean is not None:
                ctx.parameters[name] += mean
    return ctx.parameters[name]


def zero_param(ctx: Context, name: str, shape: typing.List[str]) -> jnp.ndarray:
    return get_param(ctx, name, shape, 0, 0)


def one_shape(ndim: int, dim_name: str, dim_idx: int) -> typing.List[str]:
    base = ["one"] * ndim
    base[dim_idx] = dim_name
    return base


def base_spec(inp: jnp.ndarray) -> str:
    return ''.join(chr(ord('a') + i) for i in range(inp.ndim))
