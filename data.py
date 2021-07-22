import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.data.ops.dataset_ops import _NumpyIterator as NumpyIterator

from context import Context

tf1 = tf.compat.v1


def decoder(int_string: bool, data: tf.Tensor, ctx: int):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param int_string: whether the entire dataset is in int64 or byte
    :param data: protobuf object to decode
    :param ctx: context size of generated dataset
    :return: tensorflow dataset of tokens
    """

    def chunk(proto):
        if int_string:
            dat = tf1.parse_single_example(proto, {'text': tf1.VarLenFeature(tf.int64)})
            dat = tf.cast(tf.sparse.to_dense(dat['text']), tf.int32)
        else:
            text_slice = tf1.parse_single_example(proto, {'text': tf1.FixedLenFeature([], tf.string)})['text']
            dat = tf.reshape(tf.strings.unicode_decode(text_slice, 'UTF-8'), (-1, 1))
        return tf.data.Dataset.from_tensor_slices(dat).batch(ctx + 1, drop_remainder=True)

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk, cycle_length=1)


def text_dataset(ctx: Context) -> NumpyIterator:
    filenames = tf.io.gfile.glob(ctx.data.path)

    dset = tf.data.Dataset.from_tensor_slices(filenames).repeat()
    sequence_length = ctx.dims.sizes.sequence
    batch_size = ctx.dims.sizes.batch
    device_steps = ctx.device_steps

    def _slice_target(x):
        x = tf.cast(tf.reshape(x, (device_steps, batch_size, sequence_length + 1)), tf.int32)
        return tf.stack([x[:, :, :sequence_length], x[:, :, 1:]], 1)

    dset = dset.interleave(lambda x: decoder('int64' in filenames[0], x, sequence_length),
                           cycle_length=ctx.data.interleaved_datasets,
                           num_parallel_calls=ctx.data.parallel_workers)
    if ctx.data.shuffle_buffer > 0:
        dset = dset.shuffle(ctx.data.shuffle_buffer, seed=ctx.data.seed)
    dset = dset.batch(device_steps * batch_size).map(_slice_target)
    if ctx.data.prefetch_buffer > 0:
        dset = dset.prefetch(ctx.data.prefetch_buffer)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.autotune = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = False
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
