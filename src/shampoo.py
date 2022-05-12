# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An implementation of distributed Shampoo optimizer from:
#
#  Scalable Second Order Optimization for Deep Learning
#  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
#  Preprint Paper: https://arxiv.org/abs/2002.09018
#
# This implementation moves computation of inverse pth root back to the
# accelerator (if higher precision is available).
#
# Authors: Rohan Anil (rohananil at google dot com)
#    &     Vineet Gupta (vineet at google dot com)
#
"""Distributed Shampoo Implementation."""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .context import Context

# Dtype for inverse-pth root routine
# Switch to f64 if you have hardware that supports it. Enable the jax flag
# jax_enable_x64 for this to work, otherwise it will default to float32.
_MAT_INV_PTH_ROOT_DTYPE = jnp.float64
INVERSE_FAILURE_THRESHOLD = 0.1


@jax.jit
def materialize_matrix_from_concat(block_rows_concat):
    """Returns a materialized symmetric matrix from concatenated slices.
    Args:
      block_rows_concat: The matrix represented as the concatenated
        lower-triangular blocks.
      num_blocks: The number of block-rows used to represent the symmetric matrix.
        If not specified, it is inferred from the shape of block_rows_concat.
    """
    num_blocks = find_num_blocks(block_rows_concat)

    block_size = block_rows_concat.shape[-2]

    block_rows = [block_rows_concat[Ellipsis, (k * (k + 1)) // 2 * block_size:
                                              (((k + 1) * (k + 2)) // 2 + 1) * block_size]
                  for k in range(num_blocks)]
    block_size = block_rows[0].shape[-2]
    num_blocks = len(block_rows)

    # Slice the lower-triangular and diagonal blocks into blocks.
    blocks = [[block_row[Ellipsis, i * block_size:(i + 1) * block_size] for i in range(k + 1)]
              for k, block_row in enumerate(block_rows)]

    # Generate the (off-diagonal) upper-triangular blocks.
    off_diags = [[] for _ in range(num_blocks - 1)]
    for k, block_row in enumerate(block_rows[1:]):
        for i in range(k + 1):
            off_diags[i].append(jnp.swapaxes(a=block_row[Ellipsis, i * block_size:(i + 1) * block_size], axis1=-1,
                                             axis2=-2))
    return jnp.block([row + row_t for row, row_t in zip(blocks[:-1], off_diags)] + [blocks[-1]])


def find_num_blocks(block_rows_concat):
    """Returns the number of (row) blocks representing the concatenated matrix.
    For example, an input with dimensions [256, 2560] represents 10 square blocks,
    which matches 4 lower-triangular block rows (1+2+3+4). So this function will
    return 4.
    Use ordinary numpy functions here so that the returned value is static.
    Args:
      block_rows_concat: The concatenated block array.
    Raises:
      ValueError: When the dimensions of the matrix do not correspond to a lower
      triangular block representation.
    """
    # Compute the number of square blocks used to represent the matrix.
    total_blocks = block_rows_concat.shape[-1] / block_rows_concat.shape[-2]
    # Determine the number of block rows by inverting y = x*(x+1)/2.
    num_blocks = np.round((np.sqrt(8 * total_blocks + 1) - 1) / 2).astype(np.int32)
    if (num_blocks * (num_blocks + 1)) / 2 != total_blocks:
        raise ValueError(f"total_blocks={total_blocks} does not correspond to "
                         f"a symmetric matrix. It must have the form total_blocks = x*(x+1)/2.")
    return num_blocks


def power_iteration(matrix, num_iters=100, error_tolerance=1e-6, ):
    r"""Power iteration algorithm.

    The power iteration algorithm takes a symmetric PSD matrix `A`, and produces
    a scalar `\lambda` , which is the greatest (in absolute value) eigenvalue
    of `A`, and a vector v, which is the corresponding eigenvector of `A`.

    References:
      [Wikipedia, 2021](https://en.wikipedia.org/wiki/Power_iteration)

    Args:
      matrix: the symmetric PSD matrix.
      num_iters: Number of iterations.
      error_tolerance: Iterative exit condition.

    Returns:
      eigen vector, eigen value
    """
    matrix_size = matrix.shape[-1]

    def _iter_condition(state):
        i, unused_v, unused_s, unused_s_v, run_step = state
        return jnp.logical_and(i < num_iters, run_step)

    def _iter_body(state):
        """One step of power iteration."""
        i, new_v, s, s_v, unused_run_step = state
        new_v = new_v / jnp.linalg.norm(new_v)

        s_v = jnp.einsum("ij,j->i", matrix, new_v, precision=lax.Precision.HIGHEST)
        s_new = jnp.einsum("i,i->", new_v, s_v, precision=lax.Precision.HIGHEST)
        return (i + 1, s_v, s_new, s_v,
                jnp.greater(jnp.abs(s_new - s), error_tolerance))

    # Figure out how to use step as seed for random.
    v_0 = np.random.RandomState(1729).uniform(-1.0, 1.0,
                                              matrix_size).astype(matrix.dtype)

    init_state = tuple([0, v_0, jnp.zeros([], dtype=matrix.dtype), v_0, True])
    _, v_out, s_out, _, _ = lax.while_loop(_iter_condition, _iter_body, init_state)
    v_out = v_out / jnp.linalg.norm(v_out)
    return v_out, s_out


def mat_power(mat_m, p):
    """A simple matrix power method. M^p where p can be TracedValue."""
    power = jnp.eye(mat_m.shape[0], dtype=_MAT_INV_PTH_ROOT_DTYPE)

    def _iter_condition(state):
        i, _, _ = state
        return i > 0

    def _iter_body(state):
        i, power, mat = state

        power = jax.lax.cond(i % 2 == 1, lambda: jnp.matmul(mat, power, precision=lax.Precision.HIGHEST), lambda: power)
        i //= 2
        mat = jnp.matmul(mat, mat, precision=lax.Precision.HIGHEST)
        return i, power, mat

    _, result, _ = lax.while_loop(_iter_condition, _iter_body, (p, power, mat_m))
    return result


def matrix_inverse_pth_root(matrix, p, num_iters=100, ridge_epsilon=1e-6, error_tolerance=1e-6):
    """Computes `matrix^(-1/p)`, where `p` is a positive integer.

    This function uses the Coupled newton iterations algorithm for
    the computation of a matrix's inverse pth root.


    References:
      [Functions of Matrices, Theory and Computation,
       Nicholas J Higham, Pg 184, Eq 7.18](
       https://epubs.siam.org/doi/book/10.1137/1.9780898717778)

    Args:
      matrix: the symmetric PSD matrix whose power it to be computed
      p: exponent, for p a positive integer.
      num_iters: Maximum number of iterations.
      ridge_epsilon: Ridge epsilon added to make the matrix positive definite.
      error_tolerance: Error indicator, useful for early termination.

    Returns:
      matrix^(-1/p)
    """

    # If the input is not square, materialize it from the concatenated form.
    if matrix.shape[0] != matrix.shape[1]:
        matrix = materialize_matrix_from_concat(matrix)

    assert matrix.shape[0] == matrix.shape[1]

    # We use _MAT_INV_PTH_ROOT_DTYPE for the matrix inverse pth root.
    # Switch to f64 if you have hardware that supports it. Enable the jax flag
    # jax_enable_x64 for this to work.
    matrix_size = matrix.shape[0]
    orig_dtype = matrix.dtype
    matrix = matrix.astype(_MAT_INV_PTH_ROOT_DTYPE)
    alpha = jnp.asarray(-1.0 / p, _MAT_INV_PTH_ROOT_DTYPE)
    identity = jnp.eye(matrix_size, dtype=_MAT_INV_PTH_ROOT_DTYPE)
    _, max_ev = power_iteration(matrix=matrix, num_iters=100, error_tolerance=1e-6)
    ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-6)

    def _iter_condition(state):
        (i, unused_mat_m, unused_mat_h, unused_old_mat_h, error, run_step) = state
        error_above_threshold = jnp.logical_and(error > error_tolerance, run_step)
        return jnp.logical_and(i < num_iters, error_above_threshold)

    def _iter_body(state):
        (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
        mat_m_i = (1 - alpha) * identity + alpha * mat_m
        new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=lax.Precision.HIGHEST)
        new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=lax.Precision.HIGHEST)
        new_error = jnp.max(jnp.abs(new_mat_m - identity))
        # sometimes error increases after an iteration before decreasing and
        # converging. 1.2 factor is used to bound the maximal allowed increase.
        return (i + 1, new_mat_m, new_mat_h, mat_h, new_error,
                new_error < error * 1.2)

    if matrix_size == 1:
        resultant_mat_h = (matrix + ridge_epsilon) ** alpha
        error = jnp.array(0, jnp.float32)
    else:
        damped_matrix = matrix + ridge_epsilon * identity

        z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
        new_mat_m_0 = damped_matrix * z
        new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
        new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
        init_state = tuple([0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
        _, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(_iter_condition, _iter_body, init_state)
        error = jnp.max(jnp.abs(mat_m - identity))
        is_converged = jnp.asarray(convergence, old_mat_h.dtype)
        resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    return resultant_mat_h.astype(orig_dtype), error.astype(orig_dtype)


def merge_small_dims(shape_to_merge, max_dim):
    """Merge small dimensions.

    If there are some small dimensions, we collapse them:
    e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
         [1, 2, 768, 1, 2048] --> [2, 768, 2048]

    Args:
      shape_to_merge: Shape to merge small dimensions.
      max_dim: Maximal dimension of output shape used in merging.

    Returns:
      Merged shape.
    """
    if shape_to_merge and np.all(np.array(shape_to_merge) == 1):
        return [1]

    resulting_shape = []
    product = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d
    if product > 1:
        resulting_shape.append(product)
    return resulting_shape


def pad_square_matrix(mat, max_size):
    """Pad a square matrix up to max_size.

    Args:
      mat: a matrix to pad.
      max_size: matrix size requested.

    Returns:
      Given M returns [[M, 0], [0, I]]
    """
    rows, cols = mat.shape
    if rows != cols:
        raise ValueError(f"Must have rows == cols, instead got rows={rows}, cols={cols}")
    if cols > max_size:
        raise ValueError(f"Must have cols <= max_size. Instead got cols={cols}, max_size={max_size}.")
    if rows == max_size:
        return mat
    pad_size = max_size - rows

    zs1 = jnp.zeros([rows, pad_size], dtype=mat.dtype)
    zs2 = jnp.zeros([pad_size, rows], dtype=mat.dtype)
    eye = jnp.eye(pad_size, dtype=mat.dtype)
    mat = jnp.concatenate([mat, zs1], 1)
    mat = jnp.concatenate([mat, jnp.concatenate([zs2, eye], 1)], 0)
    return mat


class Preconditioner:
    """Compute statistics/shape from gradients for preconditioning."""

    def __init__(self, param, block_size):
        self._original_shape = param.shape
        self._transformed_shape = param.shape
        self._transformed_shape = merge_small_dims(self._original_shape, block_size)
        param = jnp.reshape(param, self._transformed_shape)
        self._shape = param.shape
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param.shape):
            if 0 < block_size < d:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._num_splits = len(split_sizes)
        self._preconditioner_shapes = []
        for t in itertools.product(*split_sizes):
            self._preconditioner_shapes.extend([[d, d] for d in t])

    def shapes_for_preconditioners(self):
        return self._preconditioner_shapes

    def num_splits(self):
        return self._num_splits

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for (i, indices) in self._splits:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
            tensors = tensors_local
        return tensors

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for (i, indices) in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(jnp.concatenate(partitions[ind:ind + n], axis=i))
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]

    def statistics_from_grad(self, grad):
        """Compute statistics from gradients.

        Args:
          grad: Gradient to compute statistics from.

        Returns:
          A list of gradient statistics for each partition.
        """
        reshaped_grad = jnp.reshape(grad, self._transformed_shape)
        partitioned_grads = self.partition(reshaped_grad)
        stats = []
        for g in partitioned_grads:
            g_stats = []
            rank = len(g.shape)
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = jnp.tensordot(g, g, axes=(axes, axes))
                g_stats.append(stat)
            stats.extend(g_stats)
        return stats

    def exponent_for_preconditioner(self):
        """Returns exponent to use for inverse-pth root M^{-1/p}."""
        return 2 * len(self._transformed_shape)

    def preconditioned_grad(self, grad, preconditioners):
        """Precondition the gradient.

        Args:
          grad: A gradient tensor to precondition.
          preconditioners: A list of preconditioners to apply.

        Returns:
          A preconditioned gradient.
        """

        reshaped_grad = jnp.reshape(grad, self._transformed_shape)
        partitioned_grads = self.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        num_splits = self.num_splits()
        for i, g in enumerate(partitioned_grads):
            preconditioners_for_grad = preconditioners[i * num_splits:(i + 1) *
                                                                      num_splits]
            rank = len(g.shape)
            precond_g = g
            for j in range(rank):
                precond_g = jnp.tensordot(
                    precond_g, preconditioners_for_grad[j], axes=[[0], [0]])
            preconditioned_partitioned_grads.append(precond_g)
        merged_grad = self.merge_partitions(preconditioned_partitioned_grads)
        return jnp.reshape(merged_grad, self._original_shape)


def shampoo(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    pth_root = functools.partial(matrix_inverse_pth_root, ridge_epsilon=ctx.optimizer.epsilon)
    param = ctx.parameters[param_name]

    def _skip_preconditioning(param):
        return len(param.shape) < 1 or any([s > ctx.optimizer.skip_preconditioning_dim_size_gt for s in param.shape])

    if ctx.is_initializing:
        preconditioner = Preconditioner(param, ctx.optimizer.block_size)
        statistics = []
        preconditioners = []
        if not _skip_preconditioning(param):
            shapes = preconditioner.shapes_for_preconditioners()
            statistics = [ctx.optimizer.epsilon * jnp.eye(s[0], dtype=ctx.model.storage_dtype) for s in shapes]
            preconditioners = [jnp.eye(s[0], dtype=ctx.model.storage_dtype) for s in shapes]

        momentum = jnp.zeros_like(param).astype(ctx.model.computation_dtype)
        ctx.parameters[f'/shampoo/{param_name}/momentum'] = momentum
        for i, stat in enumerate(statistics):
            ctx.parameters[f'/shampoo/{param_name}/statistics_{i:02d}'] = stat
        for i, prec in enumerate(preconditioners):
            ctx.parameters[f'/shampoo/{param_name}/preconditioners_{i:02d}'] = prec

        return jnp.zeros_like(grad)

    statistics = [(key, param) for key, param in ctx.parameters.items() if
                  key.startswith(f'/shampoo/{param_name}/statistics_')]
    statistics = [param for _, param in sorted(statistics, key=lambda x: int(x[0].split('_')[-1]))]
    preconditioners = [(key, param) for key, param in ctx.parameters.items() if
                       key.startswith(f'/shampoo/{param_name}/preconditioners_')]
    preconditioners = [param for _, param in sorted(preconditioners, key=lambda x: int(x[0].split('_')[-1]))]
    momentum = ctx.parameters[f'/shampoo/{param_name}/momentum'].astype(ctx.model.storage_dtype)
    param = ctx.parameters[param_name]
    step = ctx.parameters['/shampoo/count']
    preconditioner = Preconditioner(param, ctx.optimizer.block_size)
    if not _skip_preconditioning(param):
        new_stats = preconditioner.statistics_from_grad(grad)
        new_statistics = []
        for stat, stat_accumulator in zip(new_stats, statistics):
            new_statistics.append(ctx.optimizer.adam_beta2 * stat_accumulator + (1.0 - ctx.optimizer.adam_beta2) * stat)
    else:
        new_statistics = [[]] * len(statistics)

    statistics = []
    original_shapes = []
    exponents = []
    max_size = 0
    prev_preconditioners = []

    original_shapes_for_state = []
    if len(new_statistics) > 0:
        preconditioner = Preconditioner(param, ctx.optimizer.block_size)
        for statistic in new_statistics:
            exponents.append(preconditioner.exponent_for_preconditioner())
            original_shapes_for_state.append(statistic.shape)
            max_size = max(max_size, statistic.shape[0])

        statistics.extend(new_statistics)
        prev_preconditioners.extend(preconditioners)
        original_shapes.extend(original_shapes_for_state)
    num_statistics = len(new_statistics)
    if not statistics:
        return jnp.zeros_like(grad)

    # Pad statistics and exponents to next multiple of num_devices.
    packed_statistics = [pad_square_matrix(stat, max_size) for stat in statistics]
    preconditioners_flat, errors_flat = jax.vmap(pth_root)(jnp.stack(packed_statistics), jnp.stack(exponents))

    def _skip(error):
        return jnp.logical_or(jnp.isnan(error), error >= INVERSE_FAILURE_THRESHOLD).astype(error.dtype)

    def _select_preconditioner(error, new_p, old_p):
        return lax.cond(_skip(error), lambda _: old_p, lambda _: new_p, operand=None)

    new_preconditioners_flat = []
    new_errors_flat = []
    for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes, prev_preconditioners, errors_flat):
        new_preconditioners_flat.append(_select_preconditioner(error, p.astype(prev_p.dtype)[:shape[0], :shape[1]],
                                                               prev_p))
        new_errors_flat.append(error)

    # Add back empty preconditioners so we that we can set the optimizer state.
    idx = 0
    if num_statistics == 0:
        new_preconditioners = []
    else:
        preconditioners_for_state = new_preconditioners_flat[idx:idx + num_statistics]
        new_preconditioners = preconditioners_for_state
        idx += num_statistics

    if _skip_preconditioning(param):
        shampoo_update = grad
    else:
        preconditioner = Preconditioner(param, ctx.optimizer.block_size)
        precond_grad = preconditioner.preconditioned_grad(grad, new_preconditioners)
        multiplier = (jnp.linalg.norm(grad) / (jnp.linalg.norm(grad) + 1e-16))
        shampoo_update = precond_grad * multiplier

    shampoo_update_with_wd = shampoo_update

    momentum = momentum * ctx.optimizer.adam_beta1 + shampoo_update
    
    ctx.parameters[f'/shampoo/{param_name}/momentum'] = momentum.astype(ctx.model.computation_dtype)
    for i, stat in enumerate(new_statistics):
        ctx.parameters[f'/shampoo/{param_name}/statistics_{i:02d}'] = stat
    for i, prec in enumerate(new_preconditioners):
        ctx.parameters[f'/shampoo/{param_name}/preconditioners_{i:02d}'] = prec

    return momentum
