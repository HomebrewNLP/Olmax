import copy
import sys
import typing

import yaml
from jax import numpy as jnp, random


class DataClass:
    pass


class DataContext(DataClass):
    def __init__(self):
        self.path = "gs://obst-euw4a-aa/the-small-chunk-char-pile/*"
        self.shuffle_buffer = 0
        self.parallel_workers = 128
        self.interleaved_datasets = 1024
        self.datasets_used_per_step = self.parallel_workers
        self.prefetch_buffer = 16
        self.seed = 0
        self.vocab_size = 256  # should be divisible by 128


class DimSizes(DataClass):
    def __init__(self, data: DataContext, group_linear_factor: float, feed_forward_factor: float):
        self.batch = 256
        self.features_per_head = 512
        self.heads = 8
        self.sequence = 256
        self.vocab = data.vocab_size
        self.one = 1
        self.intermediate_attention = int(self.features_per_head * group_linear_factor)
        self.intermediate_feed_forward = int(self.intermediate_attention * feed_forward_factor)

    def __getitem__(self, item: str):
        return getattr(self, item)


class Dims(DataClass):
    def __init__(self, data: DataContext, group_linear_factor: float, feed_forward_factor: float):
        self.batch = "batch"
        self.features_per_head = "features_per_head"
        self.heads = "heads"
        self.sequence = "sequence"
        self.intermediate_attention = "intermediate_attention"
        self.intermediate_feed_forward = "intermediate_feed_forward"
        self.one = "one"
        self.vocab = "vocab"
        self.sizes = DimSizes(data, group_linear_factor, feed_forward_factor)


class TensorboardTrace(DataClass):
    """
    Defines a tensorboard profiling output (folder) on which a tensorboard can be run to measure RAM utilization and
    view the operation trace.
    """

    def __init__(self):
        self.start_step = 16
        self.stop_step = 64 + 16
        self.do_trace = False
        self.output_path = "trace"


class Optimizer(DataClass):
    def __init__(self):
        self.learning_rate = 1e-3
        self.gradient_clip = 0.1
        self.momentum_beta = 0.9
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.weight_decay = 1e-3
        self.warmup_end = 4096
        self.exponential_decay = 1e-4


class Model(DataClass):
    def __init__(self):
        self.norm_eps = 1e-5
        self.group_linear_factor = 2
        self.depth = 32
        self.leaky_relu_slope = 0.02
        self.activation_std = 0.5893595616022745
        self.masked_attention = True
        self.feed_forward_factor = 2
        self.dtype = "bfloat16"  # valid jax.numpy.dtype


class Training(DataClass):
    def __init__(self):
        self.loss_top_p = 0.4
        self.loss_top_snap = 128  # snap top_p * batch to closest multiple
        self.device_steps = 1024
        self.steps = 2 ** 16
        self.model_parallel = 8
        self.data_parallel = 1
        self.print_interval = 1
        self.trace = TensorboardTrace()


def init_class(instance: DataClass, config: typing.Dict[str, typing.Any]):
    for name, attr in instance.__dict__.items():
        if name not in config:
            continue
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])


def serialize(instance: typing.Union[typing.Dict[str, DataClass], DataClass]):
    if isinstance(instance, DataClass):
        return serialize(instance.__dict__)
    return {k: serialize(v) if isinstance(v, DataClass) else v for k, v in instance.items()}


class Context(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.data = DataContext()
        self.optimizer = Optimizer()
        self.model = Model()
        self.dims = Dims(self.data, self.model.group_linear_factor, self.model.feed_forward_factor)
        self.training = Training()

        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            with open(sys.argv[1]) as f:
                cfg = f.read()
            init_class(self, yaml.safe_load(cfg))

        self.seed = 0
        self.global_prefix = ''

        self.name_cache: typing.Dict[str, int] = {}
        self.parameters: typing.Dict[str, jnp.ndarray] = {}
        self.parameter_dims: typing.Dict[str, typing.List[str]] = {}
        self.prng_key = random.PRNGKey(self.seed)
        self.is_initializing = False

        if config is not None:
            self.__dict__.update(config)

    def add_to_prefix(self, appended="", count=True):
        new = copy.copy(self)
        if count:
            appended = self.incremental_name(appended)
        new.global_prefix = self.global_prefix + '/' + appended
        return new

    def incremental_name(self, name):
        if name not in self.name_cache:
            self.name_cache[name] = -1
        self.name_cache[name] += 1
        return f'{name}:{self.name_cache[name]:d}'

    def config(self) -> dict:
        cfg = self.__dict__.copy()
        del cfg['name_cache'], cfg['parameters'], cfg['parameter_dims'], cfg['prng_key'], cfg['is_initializing']
        return serialize(cfg)


class WhileContext(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.config = config
        self.ctx = Context()
        self.current_step = jnp.ones([], dtype=jnp.uint32)
        self.data: typing.Optional[jnp.ndarray] = None

        if self.config is not None:
            self.ctx.parameters = config['parameters']
            self.current_step = config['current_step']
            self.data = config['data']

    def _serialize(self) -> dict:
        return {'parameters': self.ctx.parameters, 'current_step': self.current_step, 'data': self.data}

    def __call__(self, data: jnp.ndarray):
        self.data = data
        return self


class WhileTrainContext(WhileContext):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__(config)
        self.loss = jnp.zeros([])
        self.top_loss = jnp.zeros([])

        if self.config is not None:
            self.loss = config['loss']
            self.top_loss = config['top_loss']

    def serialize(self):
        serialized = self._serialize()
        serialized['loss'] = self.loss
        serialized['top_loss'] = self.top_loss
        return serialized


class WhilePredictContext(WhileContext):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__(config)

        batch_dim_size = self.ctx.dims.sizes.batch
        sequence_dim_size = self.ctx.dims.sizes.sequence
        vocab_dim_size = self.ctx.dims.sizes.vocab

        self.start_pos = jnp.zeros([batch_dim_size])
        self.stop_pos = jnp.array([sequence_dim_size] * batch_dim_size)[0]
        self.sampling_temperature = jnp.zeros([batch_dim_size])
        self.top_n = jnp.array([vocab_dim_size] * batch_dim_size)

        if self.config is not None:
            self.start_pos = config['start_pos']
            self.stop_pos = config['stop_pos']
            self.sampling_temperature = config['sampling_temperature']
            self.top_n = config['top_n']

    def serialize(self):
        serialized = self._serialize()
        serialized['start_pos'] = self.start_pos
        serialized['stop_pos'] = self.stop_pos
        serialized['sampling_temperature'] = self.sampling_temperature
        serialized['top_n'] = self.top_n

        return serialized
