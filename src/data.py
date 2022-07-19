import random

import jax
import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.python.data.ops.dataset_ops import _NumpyIterator as NumpyIterator

from .context import Context

tf1 = tf.compat.v1


def decoder(int_string: bool, data: tf.Tensor, seed: int, context_p1: int, sub_batch: int):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param int_string: whether the entire dataset is in int64 or byte
    :param data: protobuf object to decode
    :param seed: rng seed
    :param context_p1: context + 1
    :param sub_batch: number of samples should be taken from this dataset per batch
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

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk, cycle_length=1, deterministic=True)


def text_dataset(ctx: Context) -> NumpyIterator:
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
    assert not (full_batch % ctx.data.datasets_used_per_step)

    def _slice_target(x):
        """
        We're transposing here to ensure that the sampled data is balanced not only between batches but also within
        the batch.
        With 2 data loaders and a batch of 4, you'd have [[1, 1, 1, 1], [2, 2, 2, 2]] as returned sample without it and
        [[1, 2, 1, 2], [1, 2, 1, 2]] with it.
        :param x: tensor that's sliced
        :return: src/tgt Shape[Steps, Src/Tgt, Batch, Sequence]
        """
        x = tf.reshape(x, (batch_size, device_steps, sequence_length_1))
        x = tf.cast(x, tf.int32)
        x = tf.transpose(x, (1, 0, 2))
        return tf.stack([x[:, :, :sequence_length], x[:, :, 1:]], 1)

    dset = dset.interleave(lambda x: decoder('int64' in filenames[0], x, rng.randint(0, 2 ** 32),
                                             sequence_length_1, full_batch // ctx.data.datasets_used_per_step),
                           cycle_length=ctx.data.interleaved_datasets,
                           num_parallel_calls=ctx.data.parallel_workers,
                           deterministic=True)
    if ctx.data.shuffle_buffer > 0:
        dset = dset.shuffle(ctx.data.shuffle_buffer, seed=rng.randint(0, 2 ** 32))
    dset = dset.batch(ctx.data.datasets_used_per_step, deterministic=True).map(_slice_target, deterministic=True)
    if ctx.data.prefetch_buffer > 0:
        dset = dset.prefetch(ctx.data.prefetch_buffer)
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 96
    options.experimental_slack = False
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.AUTO
    dset = dset.with_options(options)
    return dset.as_numpy_iterator()
