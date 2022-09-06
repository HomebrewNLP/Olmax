import os
import random
import typing

import jax
import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy

from .context import Context

tf1 = tf.compat.v1


def decoder(int_string: bool, data: tf.Tensor, seed: int, context_p1: int, sub_batch: int, deterministic: bool):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param int_string: whether the entire dataset is in int64 or byte
    :param data: protobuf object to decode
    :param seed: rng seed
    :param context_p1: context + 1
    :param sub_batch: number of samples should be taken from this dataset per batch
    :param deterministic: whether to use sloppy interleave (fast) or deterministic interleave (slow)
    :return: tensorflow dataset of tokens
    """
    batch_prod = context_p1 * sub_batch

    def chunk(proto):
        if int_string:
            dat = tf1.parse_single_example(proto, {'text': tf1.VarLenFeature(tf.int64)})
            dat = tf.cast(tf.sparse.to_dense(dat['text']), tf.int32)
        else:
            text_slice = tf1.parse_single_example(proto, {'text': tf1.FixedLenFeature([], tf.string)})['text']
            dat = tf.strings.reduce_join(tf.strings.bytes_split(text_slice))
            dat = tf.strings.unicode_decode(dat, 'UTF-8')
            dat = tf.cast(dat, tf.uint8)
        dat = tf.reshape(dat, (-1,))
        dat = tf.slice(dat, (0,), (tf.size(dat) // batch_prod * batch_prod,))
        dat = tf.reshape(dat, (-1, context_p1))
        dat = tf.random.shuffle(dat, seed=seed)
        dat = tf.reshape(dat, (-1, batch_prod))
        return tf.data.Dataset.from_tensor_slices(dat)

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk, cycle_length=1, deterministic=deterministic)


def debug_generator(ctx: Context) -> typing.Iterator[np.ndarray]:
    rstate = np.random.RandomState(0)
    while True:
        start = rstate.uniform(1, 2 ** 30, (ctx.training.device_steps * ctx.dims.batch,)).astype(np.int64)
        multiplier = rstate.normal(size=(ctx.training.device_steps * ctx.dims.batch,)).astype(np.int64)
        out = np.arange(0, ctx.dims.sequence + 1).astype(np.int64).reshape(1, -1)
        out += start
        yield (np.sin(out) * multiplier * ctx.dims.vocab) % ctx.dims.vocab


def text_dataset(ctx: Context, skipped_steps: int) -> typing.Iterator[np.ndarray]:
    if ctx.training.debug:
        return debug_generator(ctx)

    filenames = tf.io.gfile.glob(ctx.data.path)

    rng = random.Random(ctx.data.seed)
    rng.shuffle(filenames)

    file_slice = len(filenames) / jax.process_count()
    filenames = filenames[int(file_slice * jax.process_index()):int(file_slice * (jax.process_index() + 1))]

    dset = tf.data.Dataset.from_tensor_slices(filenames).repeat()
    sequence_length = ctx.dims.sequence
    batch_size = ctx.dims.batch
    device_steps = ctx.training.device_steps
    full_batch = device_steps * batch_size
    sequence_length_1 = sequence_length + 1
    assert full_batch % ctx.data.datasets_used_per_step == 0
    is_int64 = 'int64' in filenames[0]

    def _slice_target(x):
        """
        :param x: tensor
        :return: Shape[Steps * Batch, Sequence + 1]
        """
        x = tf.reshape(x, (device_steps * batch_size, sequence_length_1))
        x = tf.cast(x, tf.int32)
        return x

    dset = dset.interleave(lambda x: decoder(is_int64, x, rng.randint(0, 2 ** 32),
                                             sequence_length_1, full_batch // ctx.data.datasets_used_per_step,
                                             ctx.data.deterministic),
                           cycle_length=ctx.data.interleaved_datasets,
                           num_parallel_calls=ctx.data.parallel_workers,
                           deterministic=ctx.data.deterministic)
    if ctx.data.shuffle_buffer_gb > 0:
        buffer_size = ctx.data.shuffle_buffer_gb * 2 ** 30 // sequence_length_1
        if is_int64:
            buffer_size //= 4  # int32 (yes, it's not actually 64)
        dset = dset.shuffle(buffer_size, seed=rng.randint(0, 2 ** 32))
    dset = dset.batch(ctx.data.datasets_used_per_step, deterministic=ctx.data.deterministic)
    dset = dset.map(_slice_target, deterministic=ctx.data.deterministic)
    if ctx.data.prefetch_buffer > 0:
        dset = dset.prefetch(ctx.data.prefetch_buffer)
    options = tf.data.Options()
    options.deterministic = ctx.data.deterministic
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.threading.private_threadpool_size = os.cpu_count()
    options.experimental_slack = not ctx.data.deterministic
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.AUTO
    dset = dset.with_options(options)

    if skipped_steps:
        dset.skip(skipped_steps)

    return dset.as_numpy_iterator()
