import copy
import typing

from jax import numpy as jnp, random


class DataContext:
    def __init__(self):
        self.path = "gs://obst-euw4a-aa/the-char-pile/*"
        self.shuffle_buffer = 0
        self.parallel_workers = None
        self.interleaved_datasets = 16
        self.prefetch_buffer = 0
        self.seed = 0
        self.vocab_size = 256  # should be divisible by 128


class DimSizes:
    def __init__(self, data: DataContext, group_linear_factor=2):
        self.batch = 1024
        self.features_per_head = 128
        self.heads = 8
        self.sequence = 256
        self.vocab = data.vocab_size
        self.one = 1
        self.intermediate_feed_forward = self.features_per_head * group_linear_factor

    def __getitem__(self, item: str):
        return getattr(self, item)


class Dims:
    def __init__(self, data: DataContext):
        self.batch = "batch"
        self.features_per_head = "features_per_head"
        self.heads = "heads"
        self.sequence = "sequence"
        self.intermediate_feed_forward = "intermediate_feed_forward"
        self.one = "one"
        self.vocab = "vocab"
        self.sizes = DimSizes(data)


class Context:
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.seed = 0
        self.learning_rate = -1e-3
        self.device_steps = 2 ** 13
        self.steps = 2 ** 16
        self.gradient_clip = 0.005
        self.head_count = 1
        self.nesterov_momentum = True
        self.momentum_beta = 0.9
        self.norm_eps = 1e-5
        self.group_linear_factor = 2
        self.depth = 8
        self.dtype = jnp.float32
        self.init_scale = 1.0
        self.global_prefix = ''
        self.model_parallel = 1
        self.data_parallel = 8
        self.z_loss = 1e-5
        self.embedding_std = 0.004
        self.norm_std = 0.02
        self.masked_attention = True
        self.print_interval = 1
        self.data = DataContext()
        self.dims = Dims(self.data)

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


class WhileContext:
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
