"""
core functionality from https://github.com/google-research/google-research/blob/19c62b3e187dd7fef2c9e94067b5e2b7f5eda53f/scalable_shampoo/optax/distributed_shampoo.py
"""
import typing

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from src.backend import dot

INVERSE_FAILURE_THRESHOLD = 0.1


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


def power_iteration(matrix: jnp.ndarray, step: typing.Union[jnp.ndarray, int] = 1729, num_iters: int = 100,
                    error_tolerance: float = 1e-6):
    r"""Power iteration algorithm.

    The power iteration algorithm takes a symmetric PSD matrix `A`, and produces
    a scalar `\lambda` , which is the greatest (in absolute value) eigenvalue
    of `A`, and a vector v, which is the corresponding eigenvector of `A`.

    References:
      [Wikipedia, 2021](https://en.wikipedia.org/wiki/Power_iteration)

    Args:
      matrix: the symmetric PSD matrix
      step: Current step in training loop of model (or any other number that can be used to see an RNG).
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
        return i + 1, s_v, s_new, s_v, jnp.greater(jnp.abs(s_new - s), error_tolerance)

    # Figure out how to use step as seed for random.
    key = jax.random.PRNGKey(step)
    v_0 = jax.random.uniform(key, (matrix_size,), jnp.float32, -1.0, 1.0).astype(matrix.dtype)  # fp64 only indirectly

    init_state = (0, v_0, jnp.zeros([], dtype=matrix.dtype), v_0, True)
    _, v_out, s_out, _, _ = lax.while_loop(_iter_condition, _iter_body, init_state)
    v_out = v_out / jnp.linalg.norm(v_out)
    return v_out, s_out


def mat_power(mat_m: jnp.ndarray, p: int):
    """A simple matrix power method. M^p where p can be TracedValue."""

    def _iter_condition(state):
        i, _, _ = state
        return i > 0

    def _iter_body(state):
        i, power, mat = state
        power = jax.lax.cond(i % 2 == 1, lambda: jnp.matmul(mat, power, precision=lax.Precision.HIGHEST), lambda: power)
        i //= 2
        mat = jnp.matmul(mat, mat, precision=lax.Precision.HIGHEST)
        return i, power, mat

    initial_result = jnp.eye(mat_m.shape[0], dtype=jnp.float64)
    return lax.while_loop(_iter_condition, _iter_body, (p, initial_result, mat_m))[1]


def matrix_inverse_pth_root(matrix: jnp.ndarray, step: jnp.ndarray, p: int, num_iters: int = 100,
                            ridge_epsilon: float = 1e-6, error_tolerance: float = 1e-6):
    """Computes `matrix^(-1/p)`, where `p` is a positive integer.

    This function uses the Coupled newton iterations algorithm for
    the computation of a matrix's inverse pth root.


    References:
      [Functions of Matrices, Theory and Computation,
       Nicholas J Higham, Pg 184, Eq 7.18](
       https://epubs.siam.org/doi/book/10.1137/1.9780898717778)

    Args:
      matrix: the symmetric PSD matrix whose power it to be computed
      step: Current step in training loop of model (or any other number that can be used to see an RNG).
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

    matrix_size = matrix.shape[0]
    matrix = matrix.astype(jnp.float64)
    alpha = jnp.asarray(-1.0 / p, jnp.float64)
    identity = jnp.eye(matrix_size, dtype=jnp.float64)
    _, max_ev = power_iteration(matrix, step, num_iters=100, error_tolerance=1e-6)
    ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-6)

    def _iter_condition(state):
        i, unused_mat_m, unused_mat_h, unused_old_mat_h, error, run_step = state
        error_above_threshold = jnp.logical_and(error > error_tolerance, run_step)
        return jnp.logical_and(i < num_iters, error_above_threshold)

    def _iter_body(state):
        i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step = state
        mat_m_i = (1 - alpha) * identity + alpha * mat_m
        new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=lax.Precision.HIGHEST)
        new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=lax.Precision.HIGHEST)
        new_error = jnp.max(jnp.abs(new_mat_m - identity))
        # sometimes error increases after an iteration before decreasing and
        # converging. 1.2 factor is used to bound the maximal allowed increase.
        return i + 1, new_mat_m, new_mat_h, mat_h, new_error, new_error < error * 1.2

    if matrix_size == 1:
        resultant_mat_h = (matrix + ridge_epsilon) ** alpha
        error = jnp.array(0, jnp.float32)
    else:
        damped_matrix = matrix + ridge_epsilon * identity

        z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
        new_mat_m_0 = damped_matrix * z
        new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
        new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
        init_state = (0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True)
        _, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(_iter_condition, _iter_body, init_state)
        error = jnp.max(jnp.abs(mat_m - identity))
        is_converged = jnp.asarray(convergence, old_mat_h.dtype)
        resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    return resultant_mat_h, error


def _skip(error):
    return jnp.logical_or(jnp.isnan(error), error >= INVERSE_FAILURE_THRESHOLD)


def fallback_pth_root(prev: jnp.array, step: jnp.ndarray, stat: jnp.array, p: int, eps: float):
    new_p, error = matrix_inverse_pth_root(stat, step, p, ridge_epsilon=eps)
    failure = _skip(error)
    return lax.cond(failure, lambda: prev, lambda: new_p), failure


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


class Preconditioner:
    """Compute statistics/shape from gradients for preconditioning."""

    def __init__(self, shape: typing.Tuple[int], block_size: int, batch_dims: int):
        self.batch_dims = batch_dims
        self.original_batched_shape = tuple(shape)
        self.batch_shape = self.original_batched_shape[:batch_dims]
        self.original_slice_shape = self.original_batched_shape[batch_dims:]
        self.reshaped_slice_shape = tuple(merge_small_dims(self.original_slice_shape, block_size))
        self.reshaped_batched_shape = self.batch_shape + self.reshaped_slice_shape
        self.splits = []
        self.rank = len(self.reshaped_slice_shape)
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(self.reshaped_slice_shape):
            if 0 > block_size or block_size > d:
                continue
            # d-1, otherwise split appends a 0-size array.
            nsplit = (d - 1) // block_size
            indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
            self.splits.append((i + self.batch_dims, indices))

    def partition(self, tensor):
        return [jnp.split(tensor, indices_or_sections=indices, axis=i) for i, indices in self.splits]

    def statistics_from_grad(self, grad):
        reshaped_grad = jnp.reshape(grad, self.reshaped_batched_shape)
        partitioned_grads = self.partition(reshaped_grad)
        stats = []
        batch_dims = list(range(self.batch_dims))
        for i, slices in enumerate(partitioned_grads, self.batch_dims):
            axes = list(range(self.batch_dims, i)) + list(range(i + 1, self.rank))
            stats.extend([dot(g, g, axes, axes, batch_dims, batch_dims) for g in slices])
        return stats

    def exponent_for_preconditioner(self):
        return 2 * self.rank

    def preconditioned_grad(self, grad, preconditioners):
        grad = jnp.reshape(grad, self.reshaped_batched_shape)
        pid = 0
        for axis, indices in self.splits:
            grad = jnp.concatenate([jnp.tensordot(g, preconditioners[pid + i], axes=[[axis], [0]])
                                    for i, g in enumerate(jnp.split(grad, indices_or_sections=indices, axis=axis))],
                                   axis=axis)
            pid += len(indices)
        return jnp.reshape(grad, self.original_batched_shape)
