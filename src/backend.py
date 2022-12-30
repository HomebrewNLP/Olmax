from typing import Sequence
from typing import Tuple, List, Any, Optional, TypeVar, Union, Callable

import jax
import jax._src.util as util
import numpy as np
from jax import lax, numpy as jnp, random

from src.constants import ParallelAxes
from src.context import Context

INT_OR_TUPLE = Union[int, Sequence[int]]

Output = TypeVar("Output")
CtxFn = TypeVar("CtxFn")

PRECISION = "highest"
jax.config.update("jax_default_matmul_precision", PRECISION)


def square_grad(fn: Callable[[jax.Array, jax.Array], jax.Array], src: jax.Array, weight: jax.Array,
                weight_sq: jax.Array):
    @jax.custom_gradient
    def _fn(x: jax.Array, wgt: jax.Array, _wgt_dummy: jax.Array):
        def _grad(dy: jax.Array):
            d_x, d_wgt = jax.vjp(fn, x, wgt)[1](dy)
            _, d_wgt_sq = jax.vjp(fn, lax.square(x), wgt)[1](lax.square(dy))
            return d_x, d_wgt, d_wgt_sq * x.shape[0]

        return fn(x, wgt), _grad

    return _fn(src, weight, weight_sq)


def add_sq(name: str) -> str:
    if name.endswith('_stacked'):
        return name[:-len('_stacked')] + '_sq_stacked'
    return name + '_sq'


def promote_to(inp: jax.Array, dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(inp, jnp.promote_types(dtype, jnp.result_type(inp)))


def with_context(count: Optional[bool] = None):
    def _inner(fn: CtxFn):
        prefix_kwargs = {"appended": fn.__name__}
        if count is not None:
            prefix_kwargs["count"] = count

        def _fn(ctx: Context, *args, add_to_prefix: bool = True, **kwargs):
            if add_to_prefix:
                ctx = ctx.add_to_prefix(**prefix_kwargs)
            return fn(ctx, *args, **kwargs)

        return _fn

    return _inner


def is_main():
    return jax.process_index() == 0


def stable_rsqrt(inp: jax.Array, eps: float) -> jax.Array:
    return jnp.reciprocal(jnp.maximum(jnp.sqrt(inp), eps))


def pos_dim(inp: jax.Array, dims: Sequence[int]) -> Sequence[int]:
    return tuple(d % inp.ndim for d in dims)


def tuple_int(obj: INT_OR_TUPLE) -> Sequence[int]:
    if isinstance(obj, (tuple, list)):
        return tuple(obj)
    if isinstance(obj, int):
        return obj,  # skipcq: PYL-R1707
    raise ValueError


def is_model(param_name: str):
    return "/stem:" in param_name and '/optimizer' not in param_name


def is_stacked(param_name: str):
    return param_name.endswith('_stacked') and is_model(param_name)


def conv(inp: jax.Array, weight: jax.Array, padding: List[Tuple[int, int]], groups: int):
    ndim = weight.ndim
    lhs = (0, ndim - 1) + tuple(range(1, ndim - 1))
    dimension_numbers = lax.ConvDimensionNumbers(lhs, (0, ndim - 1,) + tuple(range(1, ndim - 1)), lhs)
    return lax.conv_general_dilated(inp, weight, (1,) * (ndim - 2), padding=padding, feature_group_count=groups,
                                    dimension_numbers=dimension_numbers, precision=PRECISION)


def device_id():
    return (lax.psum_scatter(jnp.arange(jax.device_count()), ParallelAxes.model) / jax.device_count()).astype(jnp.int32)


def dot(left: jax.Array, right: jax.Array, left_contract_dims: INT_OR_TUPLE, right_contract_dims: INT_OR_TUPLE,
        left_batch_dims: INT_OR_TUPLE = (), right_batch_dims: INT_OR_TUPLE = ()) -> jax.Array:
    dims = ((pos_dim(left, tuple_int(left_contract_dims)), pos_dim(right, tuple_int(right_contract_dims))),
            (pos_dim(left, tuple_int(left_batch_dims)), pos_dim(right, tuple_int(right_batch_dims))))
    return lax.dot_general(left, right, dims, PRECISION)


def matmul(left: jax.Array, right: jax.Array, reduced_dims=1):
    return dot(left, right, tuple(range(-reduced_dims, 0)), tuple(range(reduced_dims)))


def prefixed_name(ctx: Context, name: str):
    return ctx.add_to_prefix(name, count=False).global_prefix


def assign(ctx: Context, name: str, inp: jax.Array):
    name = prefixed_name(ctx, name)
    ctx.parameters[name] = inp


def normal(ctx: Context, shape: Sequence[int]):
    ctx.prng_key, key = random.split(ctx.prng_key)
    return random.normal(key, shape, ctx.model.storage_dtype)


def deep_replace(d, value):
    if isinstance(d, dict):
        return {k: deep_replace(v, value) for k, v in d.items()}
    return value


def orthogonal_init(ctx: Context, shape: List[int], column_axes=(-1,)) -> jax.Array:
    column_axes = tuple(column_axes)
    axes = tuple(shape[c] for c in column_axes)
    n_rows, n_cols = util.prod(shape) // util.prod(axes), util.prod(axes)
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(normal(ctx, matrix_shape))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim)
    if n_rows < n_cols:
        out = out.T
    return jnp.reshape(out, tuple(np.delete(shape, column_axes)) + axes).astype(ctx.model.storage_dtype)


def get_param(ctx: Context, name: str, shape: Optional[List[int]] = None,
              std: Optional[float] = None, mean: Optional[float] = None, column_axes: int = 1,
              scale: float = 1., post_variance_scale: float = 1,
              lr_scale: float = 1, dtype: Optional[jnp.dtype] = None,
              init_val: Optional[jax.Array] = None,
              tied: bool = False,
              return_sq: bool = False,
              add_parameter_usages: bool = True) -> Union[Tuple[jax.Array, Optional[jax.Array]], jax.Array]:
    if return_sq:
        args = [shape, std, mean, column_axes, scale, post_variance_scale, lr_scale, dtype, init_val, tied, False]
        out0 = get_param(ctx, name, *args)
        if ctx.is_initializing:
            return out0, None
        return out0, get_param(ctx, add_sq(name), *args, add_parameter_usages=False)
    if not tied:
        name = name + '_stacked'
    add_depth = ctx.add_depth and not tied

    prefix_name = prefixed_name(ctx, name)

    if dtype is None:
        computation_dtype = ctx.model.computation_dtype
        storage_dtype = ctx.model.storage_dtype
    else:
        computation_dtype = dtype
        storage_dtype = dtype

    if add_parameter_usages:  # can't inline, because += 0 would still cause a new item (with val=0) to be created
        ctx.parameter_usages[prefix_name] += 1
    if prefix_name in ctx.parameters:
        return ctx.parameters[prefix_name].astype(computation_dtype)

    if not ctx.is_initializing and ctx.fail_on_missing_parameter:
        raise ValueError(f"Couldn't find parameter {prefix_name}. {ctx.name_cache=}")

    if init_val is not None:
        param = init_val * scale * post_variance_scale
    elif std is None and mean is None:
        param = orthogonal_init(ctx, shape, range(len(shape) - column_axes, len(shape)))
        if add_depth:
            param = normal(ctx, [ctx.dims.depth] * add_depth + list(shape)) * param.std() + param.mean()
        param *= scale * post_variance_scale
    else:
        param = normal(ctx, [ctx.dims.depth] * add_depth + list(shape)) * scale
        if std is not None:
            param *= std
        if mean is not None:
            param += mean
    ctx.parameter_variance[prefix_name] = lr_scale * scale
    assign(ctx, name, param.astype(storage_dtype))
    return param.astype(computation_dtype)


def default(option_1, option_2):
    if option_1 is None:
        return option_2
    return option_1


def zero_param(ctx: Context, name: str, shape: List[int], dtype: Optional[jnp.dtype]) -> jax.Array:
    return get_param(ctx, name, shape, 0, 0, dtype=dtype)


def loop(fn: Callable, fn_input: Any, steps: int, unroll: int = 1):
    return lax.scan(lambda *x: (fn(*x[:-1]), None), fn_input, None, steps, unroll=unroll)[0]


typevar = TypeVar("typevar")


def pattern_match(gen_fn: Callable[[int], Callable[[typevar], jax.Array]], cases: int,
                  predicate: jax.Array, base: typevar):
    return lax.switch(predicate.astype(jnp.int32) % cases, [gen_fn(i) for i in range(cases)], base)
