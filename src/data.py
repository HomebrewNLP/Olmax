import random

import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.data.ops.dataset_ops import _NumpyIterator as NumpyIterator

from .context import Context

tf1 = tf.compat.v1


def decoder(int_string: bool, data: tf.Tensor, batch_prod: int):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param int_string: whether the entire dataset is in int64 or byte
    :param data: protobuf object to decode
    :param batch_prod: sub_batch * (context + 1)
    :return: tensorflow dataset of tokens
    """

    def chunk(proto):
        if int_string:
            dat = tf1.parse_single_example(proto, {'text': tf1.VarLenFeature(tf.int64)})
            dat = tf.cast(tf.sparse.to_dense(dat['text']), tf.int32)
        else:
            text_slice = tf1.parse_single_example(proto, {'text': tf1.FixedLenFeature([], tf.string)})['text']
            dat = tf.strings.unicode_decode(text_slice, 'UTF-8')
        dat = tf.reshape(dat, (-1,))
        dat = tf.slice(dat, (0,), (tf.size(dat) // batch_prod * batch_prod,))
        dat = tf.reshape(dat, (-1, batch_prod))
        return tf.data.Dataset.from_tensor_slices(dat)

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk, cycle_length=1)


def text_dataset(ctx: Context) -> NumpyIterator:
    filenames = tf.io.gfile.glob(ctx.data.path)

    random.seed(ctx.seed)
    random.shuffle(filenames)

    dset = tf.data.Dataset.from_tensor_slices(filenames).repeat()
    sequence_length = ctx.dims.sizes.sequence
    batch_size = ctx.dims.sizes.batch
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
        :return: src/tgt
        """
        data_parallel = ctx.training.tpu_size // ctx.dims.sizes.heads
        x = tf.reshape(x, (data_parallel, batch_size // data_parallel, device_steps, sequence_length_1))
        x = tf.cast(x, tf.int32)
        x = tf.transpose(x, (0, 2, 1, 3))
        return tf.stack([x[:, :, :, :sequence_length], x[:, :, :, 1:]], 2)

    dset = dset.interleave(lambda x: decoder('int64' in filenames[0], x,
                                             sequence_length_1 * full_batch // ctx.data.datasets_used_per_step),
                           cycle_length=ctx.data.interleaved_datasets,
                           num_parallel_calls=ctx.data.parallel_workers)
    if ctx.data.shuffle_buffer > 0:
        dset = dset.shuffle(ctx.data.shuffle_buffer, seed=ctx.data.seed)
    dset = dset.batch(ctx.data.datasets_used_per_step).map(_slice_target)
    if ctx.data.prefetch_buffer > 0:
        dset = dset.prefetch(ctx.data.prefetch_buffer)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.autotune = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.AUTO
    dset = dset.with_options(options)
    return dset.as_numpy_iterator()
