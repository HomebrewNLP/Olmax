import copy
import os
import typing

import jax
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
    path: str = "gs://homebrewnlp-eu/the-char-pile/*"
    shuffle_buffer: int = 0
    parallel_workers: int = 2
    interleaved_datasets: int = 2
    prefetch_buffer: int = 2
    seed: int = 0
    vocab_size: int = 256  # should be divisible by 128
    datasets_used_per_step: int = 2


class Dims(DataClass):
    batch: int = 512
    outer_bottleneck_kernel: int = 25
    inner_bottleneck_kernel: int = 49
    inner_bottleneck_features: int = 128
    pointwise_kernel: int = 5
    features: int = 256
    pointwise_features: int = 512
    moe_intermediate: int = 4096
    heads: int = jax.device_count()
    sequence: int = 4096
    one: int = 1
    depth: int = 16

    def __init__(self, data: DataContext):
        self.vocab: int = data.vocab_size

    def __getitem__(self, item: str):
        return getattr(self, item)


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
    name: typing.Optional[str] = None
    use_wandb: bool = True
    project: str = 'gpt'
    entity: str = 'homebrewnlp'
    percentile: float = 25
    log_frequency: int = 1
    median_sizes: typing.List[int] = [64, 256, 1024]


class Optimizer(DataClass):
    use_shampoo: bool = True
    block_size: int = 512
    epsilon: float = 1e-16
    start_preconditioning_step: int = 16
    preconditioning_compute_steps: int = 128
    statistics_compute_steps: int = 4
    skip_preconditioning_dim_size_gt: int = 1024
    momentum_beta: float = 0.1
    learning_rate: float = 0.01
    gradient_clip: float = 0.001
    adam_beta1: float = 0.03
    adam_beta2: float = 0.003
    shampoo_beta2: float = 0.01
    weight_decay: float = 0.01
    warmup_end: int = 16384
    exponential_decay: float = 3e-6
    norm_scale: float = 1
    bottleneck_scale: float = 1
    pointwise_scale: float = 1
    qrnn_scale: float = 1
    moe_scale: float = 1
    input_scale: float = 1
    output_scale: float = 1


class Model(DataClass):
    conv_scale: float = 4.
    conv_shift: float = 8.
    norm_eps: float = 1e-16
    qrnn_frequency: int = 8
    rezero_lr_scale: float = 0.01
    leaky_relu_slope: float = 0.01
    activation_std: float = 0.5893595616022745
    storage_dtype: str = "float32"  # valid jax.numpy.dtype
    computation_dtype: str = "bfloat16"


class ExpectedLoss(DataClass):
    offset: float = 6.165868  # <- should be fixed. It technically goes down to 0.9 with other models.
    scale: float = 39.08037
    exponent: float = -0.3642513


class EarlyStopping(DataClass):
    minimum_relative_loss_change: float = 0.003
    maximum_spike_size: float = 3
    maximum_spike_duration: int = 24
    expected_loss = ExpectedLoss()
    loss_patience = 0.875  # target = expected_loss * loss_patience^log2(step)


class Training(DataClass):
    checkpoint_path: str = "gs://homebrewnlp-eu/homebrewnlp-checkpoint"
    checkpoint_load_path: str = ""
    checkpoint_interval: float = 16384
    label_smoothing: float = 0.1
    do_checkpoint: bool = False
    z_loss: float = 0.01
    device_steps: int = 4
    device_unroll: int = 1
    steps: int = 2 ** 16
    start_step: int = 0
    print_interval: int = 1
    trace: TensorboardTrace = TensorboardTrace()
    early_stopping: EarlyStopping = EarlyStopping()


class Evaluation(DataClass):
    eos: int = 4


class Context(DataClass):
    data: DataContext = DataContext()
    optimizer: Optimizer = Optimizer()
    model: Model = Model()
    training: Training = Training()
    wandb: WandB = WandB()
    eval: Evaluation = Evaluation()

    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.data = DataContext()
        self.optimizer = Optimizer()
        self.model = Model()
        self.training = Training()
        self.wandb = WandB()
        self.dims = Dims(self.data)
        self.dims.intermediate = self.dims.features * 2

        if 'CONFIG' in os.environ:
            with open(os.environ['CONFIG']) as f:
                cfg = f.read()
            init_class(self, yaml.safe_load(cfg))

        self.seed = 0
        self.depth = 0
        self.global_prefix = ''

        self.name_cache: typing.Dict[str, int] = {}
        self.parameters: typing.Dict[str, jnp.ndarray] = {}
        self.parameter_variance: typing.Dict[str, float] = {}
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
        del cfg['name_cache'], cfg['parameters'], cfg['prng_key'], cfg['is_initializing']
        del cfg['parameter_variance'], cfg['depth']
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

        batch_dim_size = self.ctx.dims.batch
        sequence_dim_size = self.ctx.dims.sequence
        vocab_dim_size = self.ctx.dims.vocab

        self.start_pos = jnp.zeros([batch_dim_size])
        self.stop_pos = jnp.array([sequence_dim_size] * batch_dim_size)[0]
        self.temperature = jnp.zeros([batch_dim_size])
        self.max_tokens = jnp.array([vocab_dim_size] * batch_dim_size)
        self.max_probability_mass = jnp.array([1] * batch_dim_size)
        self.typical_mass = jnp.array([1] * batch_dim_size)
        self.seed = jnp.array([0] * batch_dim_size)
        self.max_probability_to_filter = jnp.array([0] * batch_dim_size)
        self.adaptive_filter_scale = jnp.array([0] * batch_dim_size)
        self.adaptive_filter_power = jnp.array([1] * batch_dim_size)

        if self.config is not None:
            self.start_pos = config['start_pos']
            self.stop_pos = config['stop_pos']
            self.temperature = config['temperature']
            self.max_tokens = config['max_tokens']
            self.max_probability_mass = config['max_probability_mass']
            self.max_probability_to_filter = config['max_probability_to_filter']
            self.adaptive_filter_scale = config['adaptive_filter_scale']
            self.adaptive_filter_power = config['adaptive_filter_power']
            self.typical_mass = config['typical_mass']
            self.ctx.seed = config['seed']

    def serialize(self):
        serialized = self._serialize()
        serialized['start_pos'] = self.start_pos
        serialized['stop_pos'] = self.stop_pos
        serialized['temperature'] = self.temperature
        serialized['max_tokens'] = self.max_tokens
        serialized['max_probability_mass'] = self.max_probability_mass
        serialized['max_probability_to_filter'] = self.max_probability_to_filter
        serialized['adaptive_filter_scale'] = self.adaptive_filter_scale
        serialized['adaptive_filter_power'] = self.adaptive_filter_power
        serialized['typical_mass'] = self.typical_mass
        serialized['seed'] = self.ctx.seed

        return serialized
