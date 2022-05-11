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
from typing import Any, List, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax


@struct.dataclass
class SlicedSymmetricMatrix:
    """A symmetric matrix represented by lower-triangular block row slices.
    For example, the symmetric matrix M = [[a, b^T], [b, c]] would be represented
    by the block rows a and [b, c].
    The matrix may be batched, in which case each entry of block_rows may have
    dimension greater than 2. The last two dimensions represent the rows and cols.
    """
    block_rows: List[jnp.ndarray]


@jax.jit
def materialize_matrix(symmetric_matrix):
    """Returns a materialized symmetric matrix.
    Args:
      symmetric_matrix: the matrix represented by lower-triangular block slices.
    """
    block_rows = symmetric_matrix.block_rows
    block_size = block_rows[0].shape[-2]
    num_blocks = len(block_rows)

    # Slice the lower-triangular and diagonal blocks into blocks.
    blocks = [[
        block_row[Ellipsis, i * block_size:(i + 1) * block_size] for i in range(k + 1)
    ] for k, block_row in enumerate(block_rows)]

    # Generate the (off-diagonal) upper-triangular blocks.
    off_diags = [[] for _ in range(num_blocks - 1)]
    for k, block_row in enumerate(block_rows[1:]):
        for i in range(k + 1):
            off_diags[i].append(
                jnp.swapaxes(
                    a=block_row[Ellipsis, i * block_size:(i + 1) * block_size],
                    axis1=-1,
                    axis2=-2))

    return jnp.block([row + row_t for row, row_t in zip(blocks[:-1], off_diags)] +
                     [blocks[-1]])


@functools.partial(jax.jit, static_argnames=("num_blocks"))
def materialize_matrix_from_concat(
        block_rows_concat,
        num_blocks=None,
):
    """Returns a materialized symmetric matrix from concatenated slices.
    Args:
      block_rows_concat: The matrix represented as the concatenated
        lower-triangular blocks.
      num_blocks: The number of block-rows used to represent the symmetric matrix.
        If not specified, it is inferred from the shape of block_rows_concat.
    """
    if num_blocks is None:
        num_blocks = find_num_blocks(block_rows_concat)

    block_size = block_rows_concat.shape[-2]

    block_rows = [
        block_rows_concat[Ellipsis, (k * (k + 1)) // 2 *
                                    block_size:(((k + 1) * (k + 2)) // 2 + 1) * block_size]
        for k in range(num_blocks)
    ]

    return materialize_matrix(SlicedSymmetricMatrix(block_rows=block_rows))


def num_blocks_from_total_blocks(total_blocks):
    """Returns the number of blocks (i.e.
    block rows) from the total blocks.
    This is the inverse of the function x -> x*(x+1)/2.
    For example, the matrix M = [[A, B^T], [B, C]] may be represented using a
    total of 3 blocks ([A, B, C]). The number of corresponding block rows is 2.
    Args:
      total_blocks: The total blocks used to represent the matrix.
    """
    num_blocks = np.round(
        (np.sqrt(8 * total_blocks + 1) - 1) / 2).astype(np.int32)
    if (num_blocks * (num_blocks + 1)) / 2 != total_blocks:
        raise ValueError(
            f"total_blocks={total_blocks} does not correspond to "
            "a symmetric matrix. It must have the form total_blocks = x*(x+1)/2.")
    return num_blocks


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
    return num_blocks_from_total_blocks(total_blocks)


# Dtype for inverse-pth root routine
# Switch to f64 if you have hardware that supports it. Enable the jax flag
# jax_enable_x64 for this to work, otherwise it will default to float32.
_MAT_INV_PTH_ROOT_DTYPE = jnp.float64


@struct.dataclass
class TrainingMetrics:
    inverse_pth_root_errors: chex.Array  # Error for inverse-pth roots.
    # TODO(rohananil): Add more important metrics to track during training.


# Per parameter optimizer state used in data-parallel training.
class ParameterStats(NamedTuple):
    """State associated to each parameter of the model being trained."""
    diagonal_statistics: chex.Array  # Accumulator for diagonal preconditioner
    statistics: List[Any]  # Statistics (QuantizedValue, chex.Array)
    preconditioners: List[Any]  # Preconditioners (QuantizedValue, chex.Array)
    diagonal_momentum: chex.Array  # Momentum for the diagonal preconditioner
    momentum: chex.Array  # Momentum for the shampoo preconditioner
    training_metrics: TrainingMetrics  # Metrics (optional for training).


def init_training_metrics(num_statistics):
    # Since the downstream apis expect a jnp.array - we create a dummy one if
    # num_statistics=0.
    if not num_statistics:
        return TrainingMetrics(jnp.array(0, jnp.float32))
    else:
        return TrainingMetrics(jnp.zeros([num_statistics], jnp.float32))


class ShampooState(NamedTuple):
    count: chex.Array
    stats: Any


def power_iteration(
        matrix,
        num_iters=100,
        error_tolerance=1e-6,
):
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


def mat_power(
        mat_m,
        p,
):
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


def matrix_inverse_pth_root(
        matrix,
        p,
        num_iters=100,
        ridge_epsilon=1e-6,
        error_tolerance=1e-6,
):
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
        init_state = tuple(
            [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
        _, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(_iter_condition, _iter_body, init_state)
        error = jnp.max(jnp.abs(mat_m - identity)).astype(jnp.float32)
        is_converged = jnp.asarray(convergence, old_mat_h.dtype)
        resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
        resultant_mat_h = jnp.asarray(resultant_mat_h, orig_dtype)
    return resultant_mat_h, error


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
        raise ValueError("Must have rows == cols, instead got "
                         f"rows={rows}, cols={cols}")
    if cols > max_size:
        raise ValueError("Must have cols <= max_size. Instead got "
                         f"cols={cols}, max_size={max_size}.")
    if rows == max_size:
        return mat
    pad_size = max_size - rows

    zs1 = jnp.zeros([rows, pad_size], dtype=mat.dtype)
    zs2 = jnp.zeros([pad_size, rows], dtype=mat.dtype)
    eye = jnp.eye(pad_size, dtype=mat.dtype)
    mat = jnp.concatenate([mat, zs1], 1)
    mat = jnp.concatenate([mat, jnp.concatenate([zs2, eye], 1)], 0)
    return mat


def efficient_cond(predicate, compute_fn, init_state, *args, **kwargs):
    """Avoids wasteful buffer allocation with XLA."""

    def _iter_body(unused_state):
        results = compute_fn(*args, **kwargs)
        return tuple([False] + list(results))

    def _iter_condition(state):
        return state[0]

    results = jax.lax.while_loop(_iter_condition, _iter_body,
                                 tuple([predicate] + init_state))
    return tuple(results[1:])


class BlockPartitioner:
    """Partitions a tensor into smaller tensors."""

    def __init__(self, param, block_size):
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
                partial_merged_tensors.append(
                    jnp.concatenate(partitions[ind:ind + n], axis=i))
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


class Preconditioner:
    """Compute statistics/shape from gradients for preconditioning."""

    def __init__(self, param, block_size):
        self._original_shape = param.shape
        self._transformed_shape = param.shape
        self._transformed_shape = merge_small_dims(self._original_shape, block_size)
        reshaped_param = jnp.reshape(param, self._transformed_shape)
        self._partitioner = BlockPartitioner(reshaped_param, block_size)

    def statistics_from_grad(self, grad):
        """Compute statistics from gradients.

        Args:
          grad: Gradient to compute statistics from.

        Returns:
          A list of gradient statistics for each partition.
        """
        reshaped_grad = jnp.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
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

    def shapes_for_preconditioners(self):
        """Returns shape from statistics."""
        return self._partitioner.shapes_for_preconditioners()

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
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        num_splits = self._partitioner.num_splits()
        for i, g in enumerate(partitioned_grads):
            preconditioners_for_grad = preconditioners[i * num_splits:(i + 1) *
                                                                      num_splits]
            rank = len(g.shape)
            precond_g = g
            for j in range(rank):
                precond_g = jnp.tensordot(
                    precond_g, preconditioners_for_grad[j], axes=[[0], [0]])
            preconditioned_partitioned_grads.append(precond_g)
        merged_grad = self._partitioner.merge_partitions(
            preconditioned_partitioned_grads)
        return jnp.reshape(merged_grad, self._original_shape)


def distributed_shampoo(
        block_size,
        beta1=0.9,
        beta2=0.999,
        matrix_epsilon=1e-6,
        start_preconditioning_step=5,
        preconditioning_compute_steps=1,
        statistics_compute_steps=1,
        inverse_failure_threshold=0.1,
        skip_preconditioning_dim_size_gt=4096):
    """Distributed Shampoo optimizer.

    Distributed Shampoo is a second-order preconditioned method (concretely, a
    variant of full-matrix Adagrad), that provides significant convergence and
    wall-clock time improvements compared to conventional first-order methods,
    and that has been shown to scale to large state-of-the-art deep learning
    models.

    References:
      Scalable Second Order Optimization for Deep Learning,
      Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer

      Preprint: https://arxiv.org/abs/2002.09018

    Args:
      block_size: Block size for large layers (if > 0). Preconditioning compute
        operation is cubic in the dimension of the tensor. Block size allows us to
        chunk the layers into sub-layers of maximal dimension dictated by this
        value. Use 128 as default (increase if you have compute budget).
      beta1: momentum parameter.
      beta2: second moment averaging parameter.
      matrix_epsilon: epsilon to add to statistics before computing inverse pth
        root. If you are running in f32 precision for inverse pth root
        (recommended today) this can go upto 1e-6. If you have latest hardware
        with native f64 precision, set this upto 1e-12.
      start_preconditioning_step: When to start Shampoo update before which
        diagonal update is used. This is because we dont have enough information
        to do stable inverse.
      preconditioning_compute_steps: How often to compute preconditioner.
        Performance tuning params for controlling memory and compute requirements.
        Ideally set this and statistics_compute_steps params to 1.
      statistics_compute_steps: How often to compute statistics.
      inverse_failure_threshold: numerics are hard and inverses fail sometimes; we
        determine that using this threshold.
      skip_preconditioning_dim_size_gt: Skip if preconditioning dim size is
        greater than this value.

    Returns:
      a GradientTransformation.
    """
    pth_root = functools.partial(matrix_inverse_pth_root, ridge_epsilon=matrix_epsilon, precision=lax.Precision.HIGHEST)

    def init_fn(params):
        """Initialise the optimiser's state."""

        def _init(param):
            preconditioner = Preconditioner(param, block_size)
            statistics = []
            preconditioners = []
            if not _skip_preconditioning(param):
                shapes = preconditioner.shapes_for_preconditioners()
                statistics = [matrix_epsilon * jnp.eye(s[0], dtype=jnp.float32) for s in shapes]
                preconditioners = [jnp.eye(s[0], dtype=jnp.float32) for s in shapes]

            diagonal_statistics = []

            momentum = jnp.zeros_like(param).astype(jnp.bfloat16)
            diagonal_momentum = jnp.zeros_like(param).astype(jnp.bfloat16)

            return ParameterStats(diagonal_statistics, statistics, preconditioners, diagonal_momentum,
                                  momentum, init_training_metrics(len(statistics)))

        return ShampooState(count=jnp.zeros([], jnp.int32), stats=jax.tree_map(_init, params))

    def _skip_preconditioning(param):
        return len(param.shape) < 1 or any([s > skip_preconditioning_dim_size_gt for s in param.shape])

    def _compute_stats(grad, state, param, step):
        """Compute per-parameter statistics."""
        preconditioner = Preconditioner(param, block_size)
        if not _skip_preconditioning(param):
            def compute_updated_statistics():
                new_stats = preconditioner.statistics_from_grad(grad)
                new_stats_accumulators = []
                for stat, stat_accumulator in zip(new_stats, state.statistics):
                    new_stats_accumulators.append(beta2 * stat_accumulator + (1.0 - beta2) * stat)
                return new_stats_accumulators

            perform_step = step % statistics_compute_steps == 0
            init_state = state.statistics
            new_statistics = list(efficient_cond(perform_step, compute_updated_statistics, init_state))
            return ParameterStats(state.diagonal_statistics, new_statistics, state.preconditioners,
                                  state.diagonal_momentum,
                                  state.momentum, state.training_metrics)

    def _matrix_inverse_pth_root_vmap(xs, ps):
        return jax.vmap(pth_root)(xs, ps)

    def _pmap_compute_preconditioners(states, step, statistics,
                                      num_statistics_per_state, original_shapes,
                                      exponents, max_size, prev_preconditioners):
        """Computes preconditioners for given statistics in states in PMAP mode.

        Args:
          states: A list of optimizer states.
          step: Current step number
          statistics: A list of statistics for all variables (for every dim)
          num_statistics_per_state: Number of statistis per state to reconstruct
            output states.
          original_shapes: A list of shapes of the statistics.
          exponents: Exponent power to use for inverse-pth roots.
          max_size: Maximum dim of the statistics to pad.
          prev_preconditioners: Previously available preconditioner.

        Returns:
          New optimizer states after computing the preconditioner.
        """
        if not statistics:
            return states

        num_statistics = len(statistics)
        # Pad statistics and exponents to next multiple of num_devices.
        packed_statistics = [pad_square_matrix(stat, max_size) for stat in statistics]

        def _internal_inverse_pth_root_all():
            return _matrix_inverse_pth_root_vmap(jnp.stack(packed_statistics), jnp.stack(exponents))

        preconditioners_init = packed_statistics
        errors_init = ([inverse_failure_threshold] * len(packed_statistics))
        init_state = [preconditioners_init, errors_init]
        perform_step = step % preconditioning_compute_steps == 0
        preconditioners_flat, errors_flat = efficient_cond(perform_step, _internal_inverse_pth_root_all, init_state)

        def _skip(error):
            return jnp.logical_or(jnp.isnan(error), error >= inverse_failure_threshold).astype(error.dtype)

        def _select_preconditioner(error, new_p, old_p):
            return lax.cond(_skip(error), lambda _: old_p, lambda _: new_p, operand=None)

        new_preconditioners_flat = []
        new_errors_flat = []
        for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes, prev_preconditioners, errors_flat):
            new_preconditioners_flat.append(_select_preconditioner(error, p[:shape[0], :shape[1]], prev_p))
            new_errors_flat.append(error)

        assert len(states) == len(num_statistics_per_state)
        assert len(new_preconditioners_flat) == num_statistics
        assert len(new_errors_flat) == num_statistics

        # Add back empty preconditioners so we that we can set the optimizer state.
        preconditioners_for_states = []
        idx = 0
        errors_for_states = []
        for num_statistics, state in zip(num_statistics_per_state, states):
            if num_statistics == 0:
                preconditioners_for_states.append([])
                errors_for_states.append(jnp.array(0, jnp.float32))
            else:
                preconditioners_for_state = new_preconditioners_flat[idx:idx + num_statistics]
                assert len(state.statistics) == len(preconditioners_for_state)
                preconditioners_for_states.append(preconditioners_for_state)

                errors_for_state = jnp.stack(new_errors_flat[idx:idx + num_statistics])
                assert len(state.statistics) == len(errors_for_state)
                errors_for_states.append(errors_for_state)

                idx += num_statistics
        new_states = []
        for state, new_preconditioners, new_errors in zip(states, preconditioners_for_states, errors_for_states):
            if state.statistics:
                new_errors = jnp.where(jnp.logical_and(new_errors > 0.0, new_errors != inverse_failure_threshold),
                                       new_errors, state.training_metrics.inverse_pth_root_errors)
            new_training_metrics = TrainingMetrics(new_errors)
            new_states.append(ParameterStats(state.diagonal_statistics, state.statistics, new_preconditioners,
                                             state.diagonal_momentum, state.momentum, new_training_metrics))

        return new_states

    def _compute_preconditioners(states, params, step):
        """Computes preconditioners for given statistics in states.

        Args:
          states: A list of optimizer states.
          params: A list of params.
          step: Current step number

        Returns:
          New optimizer states after computing the preconditioner.
        """
        statistics = []
        num_statistics_per_state = []
        original_shapes = []
        exponents = []
        max_size = 0
        prev_preconditioners = []

        for state, param in zip(states, params):
            num_statistics = len(state.statistics)
            num_statistics_per_state.append(num_statistics)
            original_shapes_for_state = []
            if num_statistics > 0:
                preconditioner = Preconditioner(param, block_size)
                for statistic in state.statistics:
                    exponents.append(preconditioner.exponent_for_preconditioner())
                    original_shapes_for_state.append(statistic.shape)
                    max_size = max(max_size, statistic.shape[0])

                statistics.extend(state.statistics)
                prev_preconditioners.extend(state.preconditioners)
                original_shapes.extend(original_shapes_for_state)

        return _pmap_compute_preconditioners(states, step, statistics, num_statistics_per_state, original_shapes,
                                             exponents, max_size, prev_preconditioners)

    def _transform_grad(grad, state, param, step):
        """Transform per-parameter gradients."""
        preconditioner = Preconditioner(param, block_size)
        sgd_update = grad
        new_diagonal_statistics = state.diagonal_statistics.to_float()
        grafting_update = sgd_update

        precond_grad = grad
        if not _skip_preconditioning(param):
            precond_grad = preconditioner.preconditioned_grad(precond_grad, state.preconditioners)
        else:
            precond_grad = grafting_update

        grafting_update_norm = jnp.linalg.norm(grafting_update)
        precond_grad_norm = jnp.linalg.norm(precond_grad)

        multiplier = (grafting_update_norm / (precond_grad_norm + 1e-16))
        shampoo_update = precond_grad * multiplier

        shampoo_update_with_wd = shampoo_update
        grafting_update_with_wd = grafting_update

        shampoo_update_with_wd_momentum = state.momentum.to_float() * beta1 + shampoo_update_with_wd
        grafting_update_with_wd_momentum = state.diagonal_momentum.to_float() * beta1 + grafting_update_with_wd
        run_shampoo = (step >= start_preconditioning_step).astype(grafting_update_with_wd_momentum.dtype)
        update = run_shampoo * shampoo_update_with_wd_momentum + (1.0 - run_shampoo) * grafting_update_with_wd_momentum

        new_diagonal_momentum = grafting_update_with_wd_momentum
        new_momentum = shampoo_update_with_wd_momentum

        param_stats = ParameterStats(new_diagonal_statistics, state.statistics, state.preconditioners,
                                     new_diagonal_momentum.astype(jnp.bfloat16), new_momentum.astype(jnp.bfloat16),
                                     state.training_metrics)

        return update, param_stats

    return init_fn, _compute_stats, _compute_preconditioners, _transform_grad
