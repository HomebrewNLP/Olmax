import copy
import sys
import typing

import yaml
from jax import numpy as jnp, random


class DataClass:
    def serialize(self):
        return serialize(self)


def fn_if_dataclass(instance: typing.Any, fn: typing.Callable):
    return fn(instance) if isinstance(instance, (DataClass, list, tuple, dict)) else instance


def serialize(instance: typing.Union[DataClass, typing.Dict[str, typing.Any]]):
    if isinstance(instance, DataClass):
        attributes = {key: getattr(instance, key) for key in dir(instance) if
                      not key.startswith('_') and not key.endswith('_')}
        return serialize({key: value for key, value in attributes.items() if not isinstance(value, typing.Callable)})
    if isinstance(instance, (list, tuple)):
        return [fn_if_dataclass(itm, serialize) for itm in instance]
    if isinstance(instance, dict):
        return {k: fn_if_dataclass(v, serialize) for k, v in instance.items()}
    return instance


def init_class(instance: DataClass, config: typing.Dict[str, typing.Any]):
    for name in dir(instance):
        if name.startswith("_") or name.endswith("_") or name not in config:
            continue
        attr = getattr(instance, name)
        is_dataclass = isinstance(attr, DataClass)
        is_list = isinstance(attr, (list, tuple))
        is_dict = isinstance(attr, dict)
        if not (is_dataclass or (is_list and isinstance(attr[0], DataClass)) or (
                is_dict and isinstance(next(iter(attr.values())), DataClass))):
            setattr(instance, name, config[name])
            continue

        if is_dataclass:
            init_class(attr, config[name])
        elif is_list:
            setattr(instance, name, type(attr)(init_class_copy(attr[0], item) for item in config[name]))
        elif is_dict:
            base = next(iter(attr.values()))
            setattr(instance, name, {key: init_class_copy(base, item) for key, item in config[name].items()})
        else:
            raise ValueError(f"Unknown type {type(attr)} with given data {config[name]}")


def init_class_copy(instance: DataClass, config: typing.Dict[str, typing.Any]) -> DataClass:
    instance = copy.deepcopy(instance)
    init_class(instance, config)
    return instance


class DataContext(DataClass):
    path: str = "gs://ggpt4/the-big-char-pile/*"
    shuffle_buffer: int = 0
    parallel_workers: int = 8
    interleaved_datasets: int = 8
    prefetch_buffer: int = 2
    seed: int = 0
    vocab_size: int = 256  # should be divisible by 128
    datasets_used_per_step: int = 8


class DimSizes(DataClass):
    batch: int = 2
    full_conv_kernel: int = 7
    depthwise_conv_kernel: int = 81
    features_per_head: int = 256
    intermediate: int = 512
    heads: int = 8
    sequence: int = 65536
    one: int = 1
    depth: int = 8

    def __init__(self, data: DataContext):
        self.vocab: int = data.vocab_size

    def __getitem__(self, item: str):
        return getattr(self, item)


class Dims(DataClass):
    batch: str = "batch"
    features_per_head: str = "features_per_head"
    heads: str = "heads"
    full_conv_kernel: str = "full_conv_kernel"
    depthwise_conv_kernel: str = "depthwise_conv_kernel"
    depth: str = "depth"
    sequence: str = "sequence"
    anonymous_sequence: str = "anonymous_sequence"
    intermediate: str = "intermediate"
    one: str = "one"
    multiplier: str = "multiplier"
    vocab: str = "vocab"

    def __init__(self, data: DataContext):
        self.sizes: DimSizes = DimSizes(data)


class TensorboardTrace(DataClass):
    """
    Defines a tensorboard profiling output (folder) on which a tensorboard can be run to measure RAM utilization and
    view the operation trace.
    """
    start_step: int = 16
    stop_step: int = 64 + 16
    do_trace: bool = False
    output_path: str = "trace"


class WandB(DataClass):
    use_wandb: bool = True
    project: str = 'gpt'
    entity: str = 'homebrewnlp'
    storage: str = 'redis://0.0.0.0:51367'  # used for sweeps with external tuners
    percentile: float = 25
    log_frequency: int = 1
    median_sizes: typing.List[int] = [256]


class Optimizer(DataClass):
    momentum_beta: float = 0.1
    learning_rate: float = 0.001
    gradient_clip: float = 0.001
    adam_beta1: float = 0.1
    adam_beta2: float = 0.01
    weight_decay: float = 0.01
    warmup_end: int = 1024
    exponential_decay: float = 1e-4


class Model(DataClass):
    rezero_lr_scale: float = 0.01
    leaky_relu_slope: float = 0.02
    activation_std: float = 0.5893595616022745
    storage_dtype: str = "float32"  # valid jax.numpy.dtype
    computation_dtype: str = "bfloat16"


class Training(DataClass):
    checkpoint_path: str = "gs://ggpt4/homebrewnlp-checkpoint"
    checkpoint_interval: float = 16384
    do_checkpoint: bool = False
    z_loss: float = 0.01
    device_steps: int = 4
    device_unroll: int = 1
    steps: int = 2 ** 16
    print_interval: int = 1
    trace: TensorboardTrace = TensorboardTrace()
    minimum_relative_loss_change: float = 0.003
    maximum_spike_size: float = 3
    maximum_spike_duration: int = 24
    # Best run + ~20%: {128: 5, 256: 4, 512: 3.5, 1024: 3, 2048: 2.5, 3072: 2.1, 4096: 2, 8192: 1}
    # add another 50%, so the hyperparameter optimizer gets more information and doesn't stop everything simultaneously
    loss_thresholds: typing.Dict[int, int] = {128: 8, 256: 6, 512: 5, 1024: 4.5, 2048: 4, 3072: 3.5, 4096: 3, 8192: 2}


class Context(DataClass):
    data: DataContext = DataContext()
    optimizer: Optimizer = Optimizer()
    model: Model = Model()
    training: Training = Training()
    wandb: WandB = WandB()

    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.data = DataContext()
        self.optimizer = Optimizer()
        self.model = Model()
        self.training = Training()
        self.wandb = WandB()
        self.dims = Dims(self.data)
        self.dims.sizes.intermediate = self.dims.sizes.features_per_head * 2

        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            with open(sys.argv[1]) as f:
                cfg = f.read()
            init_class(self, yaml.safe_load(cfg))

        self.seed = 0
        self.global_prefix = ''

        self.name_cache: typing.Dict[str, int] = {}
        self.parameters: typing.Dict[str, jnp.ndarray] = {}
        self.parameter_variance: typing.Dict[str, float] = {}
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
        del cfg['parameter_variance']
        return serialize(cfg)


class WhileContext(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.config = config
        self.ctx = Context()
        self.current_step = jnp.ones([], dtype=jnp.uint32)
        self.data: typing.Optional[jnp.ndarray] = None

        if self.config is not None:
            self.ctx.parameters = config['parameters']
            self.ctx.parameter_variance = config['parameter_variance']
            self.current_step = config['current_step']
            self.data = config['data']

    def _serialize(self) -> dict:
        return {'parameters': self.ctx.parameters, 'current_step': self.current_step, 'data': self.data,
                'parameter_variance': self.ctx.parameter_variance}

    def __call__(self, data: jnp.ndarray):
        self.data = data
        return self


class WhileTrainContext(WhileContext):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__(config)
        self.loss = jnp.zeros([])
        self.current_loss = jnp.zeros([])
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
        self.top_k = jnp.array([vocab_dim_size] * batch_dim_size)

        if self.config is not None:
            self.start_pos = config['start_pos']
            self.stop_pos = config['stop_pos']
            self.sampling_temperature = config['sampling_temperature']
            self.top_k = config['top_k']

    def serialize(self):
        serialized = self._serialize()
        serialized['start_pos'] = self.start_pos
        serialized['stop_pos'] = self.stop_pos
        serialized['sampling_temperature'] = self.sampling_temperature
        serialized['top_k'] = self.top_k

        return serialized
