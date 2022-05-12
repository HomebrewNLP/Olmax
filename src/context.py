import copy
import os
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
    path: str = "gs://homebrewnlp-eu/the-token-pile/*"
    shuffle_buffer: int = 0
    parallel_workers: int = 2
    interleaved_datasets: int = 2
    prefetch_buffer: int = 2
    seed: int = 0
    vocab_size: int = 65536  # should be divisible by 128
    datasets_used_per_step: int = 4


class DimSizes(DataClass):
    batch: int = 128
    outer_bottleneck_kernel: int = 25
    inner_bottleneck_kernel: int = 49
    inner_bottleneck_features: int = 128
    pointwise_kernel: int = 5
    features: int = 256
    pointwise_features: int = 512
    moe_intermediate: int = 4096
    heads: int = 8
    sequence: int = 4096
    one: int = 1
    depth: int = 16

    def __init__(self, data: DataContext):
        self.vocab: int = data.vocab_size

    def __getitem__(self, item: str):
        return getattr(self, item)


class Dims(DataClass):
    batch: str = "batch"
    outer_bottleneck_kernel: str = "outer_bottleneck_kernel"
    inner_bottleneck_kernel: str = "inner_bottleneck_kernel"
    inner_bottleneck_features: str = "inner_bottleneck_features"
    pointwise_kernel: str = "pointwise_kernel"
    features: str = "features"
    pointwise_features: str = "pointwise_features"
    moe_intermediate: str = "moe_intermediate"
    heads: str = "heads"
    one: str = "one"
    sequence: str = "sequence"
    depth: str = "depth"
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
    percentile: float = 25
    log_frequency: int = 1
    median_sizes: typing.List[int] = [64, 256, 1024]


class Optimizer(DataClass):
    use_shampoo: bool = False
    block_size: int = 512
    epsilon: float = 1e-6
    start_preconditioning_step: int = 16
    preconditioning_compute_steps: int = 128
    statistics_compute_steps: int = 4
    skip_preconditioning_dim_size_gt: int = 1024
    momentum_beta: float = 0.1
    learning_rate: float = 10
    gradient_clip: float = 0.001
    adam_beta1: float = 0.1
    adam_beta2: float = 0.01
    weight_decay: float = 0.001
    warmup_end: int = 1024
    exponential_decay: float = 1e-4


class Model(DataClass):
    norm_eps: float = 1e-5
    qrnn_frequency: int = 8
    rezero_lr_scale: float = 0.01
    leaky_relu_slope: float = 0.02
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
    pretrained_embedding_path: str = ''
    checkpoint_path: str = "gs://homebrewnlp-eu/homebrewnlp-checkpoint"
    checkpoint_interval: float = 16384
    do_checkpoint: bool = False
    z_loss: float = 0.01
    device_steps: int = 4
    device_unroll: int = 1
    steps: int = 2 ** 16
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
        self.dims.sizes.intermediate = self.dims.sizes.features * 2

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

        batch_dim_size = self.ctx.dims.sizes.batch
        sequence_dim_size = self.ctx.dims.sizes.sequence
        vocab_dim_size = self.ctx.dims.sizes.vocab

        self.start_pos = jnp.zeros([batch_dim_size])
        self.stop_pos = jnp.array([sequence_dim_size] * batch_dim_size)[0]
        self.temperature = jnp.zeros([batch_dim_size])
        self.top_k = jnp.array([vocab_dim_size] * batch_dim_size)
        self.top_p = jnp.array([1] * batch_dim_size)
        self.seed = jnp.array([0] * batch_dim_size)

        if self.config is not None:
            self.start_pos = config['start_pos']
            self.stop_pos = config['stop_pos']
            self.temperature = config['temperature']
            self.top_k = config['top_k']
            self.top_p = config['top_p']
            self.ctx.seed = config['seed']

    def serialize(self):
        serialized = self._serialize()
        serialized['start_pos'] = self.start_pos
        serialized['stop_pos'] = self.stop_pos
        serialized['temperature'] = self.temperature
        serialized['top_k'] = self.top_k
        serialized['top_p'] = self.top_p
        serialized['seed'] = self.ctx.seed

        return serialized
