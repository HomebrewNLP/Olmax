import copy
import sys
import typing

import jsonpickle
from jax import numpy as jnp, random


class DataClass:
    pass


class DataContext(DataClass):
    def __init__(self):
        self.path = "gs://obst-euw4a-aa/the-char-pile/*"
        self.shuffle_buffer = 0
        self.parallel_workers = None
        self.interleaved_datasets = 1
        self.prefetch_buffer = 0
        self.seed = 0
        self.vocab_size = 256  # should be divisible by 128


class DimSizes(DataClass):
    def __init__(self, data: DataContext, group_linear_factor=2):
        self.batch = 512
        self.features_per_head = 128
        self.heads = 8
        self.sequence = 256
        self.vocab = data.vocab_size
        self.one = 1
        self.intermediate_feed_forward = self.features_per_head * group_linear_factor

    def __getitem__(self, item: str):
        return getattr(self, item)


class Dims(DataClass):
    def __init__(self, data: DataContext):
        self.batch = "batch"
        self.features_per_head = "features_per_head"
        self.heads = "heads"
        self.sequence = "sequence"
        self.intermediate_feed_forward = "intermediate_feed_forward"
        self.one = "one"
        self.vocab = "vocab"
        self.sizes = DimSizes(data)


class Optimizer(DataClass):
    def __init__(self):
        self.learning_rate = -1e3
        self.gradient_clip = 5e-3
        self.nesterov_momentum = True
        self.momentum_beta = 0.9


class Initializer(DataClass):
    def __init__(self):
        self.scale = 1.0
        self.embedding_std = 0.004
        self.norm_std = 0.02


class Model(DataClass):
    def __init__(self):
        self.norm_eps = 1e-5
        self.group_linear_factor = 2
        self.depth = 8
        self.masked_attention = True
        self.dtype = jnp.bfloat16
        self.z_loss = 1e-5
        self.initializer = Initializer()


class Training(DataClass):
    def __init__(self):
        self.device_steps = 16
        self.steps = 2 ** 16
        self.model_parallel = 8
        self.data_parallel = 1
        self.print_interval = 1


def init_class(instance: DataClass, config: typing.Dict[str, typing.Any]):
    for name, attr in instance.__dict__.items():
        if name not in config:
            continue
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])


class Context(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.data = DataContext()
        self.dims = Dims(self.data)
        self.optimizer = Optimizer()
        self.model = Model()
        self.training = Training()

        if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
            with open(sys.argv[1]) as f:
                cfg = f.read()
            init_class(self, jsonpickle.loads(cfg))

        self.seed = 0
        self.global_prefix = ''

        self.name_cache: typing.Dict[str, int] = {}
        self.parameters: typing.Dict[str, jnp.ndarray] = {}
        self.parameter_dims: typing.Dict[str, typing.List[str]] = {}
        self.prng_key = random.PRNGKey(self.seed)

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


class WhileContext(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.ctx = Context()
        self.current_step = jnp.zeros([], dtype=jnp.uint32)
        self.data: typing.Optional[jnp.ndarray] = None
        self.loss = jnp.zeros([])

        if config is not None:
            self.ctx.parameters = config['parameters']
            self.loss = config['loss']
            self.current_step = config['current_step']
            self.data = config['data']

    def serialize(self):
        return {'parameters': self.ctx.parameters, 'current_step': self.current_step, 'loss': self.loss,
                'data': self.data}

    def __call__(self, data: jnp.ndarray):
        self.data = data
        return self
