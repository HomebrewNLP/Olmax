import typing

import tensorflow as tf

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
        dat = tf.data.Dataset.from_tensor_slices(dat)
        dat = dat.window(size=ctx + 1, shift=ctx, stride=1, drop_remainder=True)
        dat = dat.interleave(lambda x: x.batch(ctx + 1, drop_remainder=True), cycle_length=1)
        return dat

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk, cycle_length=1)


def text_dataset(ctx: Context) -> typing.Iterable[tf.Tensor]:
    filenames = tf.io.gfile.glob(ctx.data.path)

    dset = tf.data.Dataset.from_tensor_slices(filenames).repeat()
    sequence_length = ctx.dims.dim_sizes[ctx.dims.sequence]
    batch_size = ctx.dims.dim_sizes[ctx.dims.batch]
    device_steps = ctx.device_steps

    def _slice_target(x):
        x = tf.cast(tf.reshape(x, (device_steps, batch_size, sequence_length + 1)), tf.int32)
        return tf.stack([x[:, :sequence_length], x[:, 1:]], 1)

    dset = dset.interleave(lambda x: decoder('int64' in filenames[0], x, sequence_length),
                           cycle_length=ctx.data.interleaved_datasets,
                           num_parallel_calls=ctx.data.parallel_interleave)
    dset = dset.shuffle(ctx.data.shuffle_buffer, seed=ctx.data.seed).batch(device_steps * batch_size).map(_slice_target)
    return iter(dset)
