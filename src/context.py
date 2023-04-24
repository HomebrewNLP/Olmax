import collections
import copy
import os
from typing import Any, Callable, Union, Dict, Optional, List

import jax
import yaml
from jax import numpy as jnp, random


class DataClass:
    def serialize(self):
        return serialize(self)


def fn_if_dataclass(instance: Any, fn: Callable):
    return fn(instance) if isinstance(instance, (DataClass, list, tuple, dict)) else instance


def serialize(instance: Union[DataClass, Dict[str, Any]]):
    if isinstance(instance, DataClass):
        attributes = {key: getattr(instance, key) for key in dir(instance) if
                      not key.startswith('_') and not key.endswith('_')}
        return serialize({key: value for key, value in attributes.items() if not isinstance(value, Callable)})
    if isinstance(instance, (list, tuple)):
        return [fn_if_dataclass(itm, serialize) for itm in instance]
    if isinstance(instance, dict):
        return {k: fn_if_dataclass(v, serialize) for k, v in instance.items()}
    return instance


def init_class(instance: DataClass, config: Dict[str, Any]):
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


def init_class_copy(instance: DataClass, config: Dict[str, Any]) -> DataClass:
    instance = copy.deepcopy(instance)
    init_class(instance, config)
    return instance


class DataContext(DataClass):
    path: str = "gs://homebrewnlp-eu/the-char-pile/*"
    shuffle_buffer_gb: int = 64
    parallel_workers: int = 2
    interleaved_datasets: int = 2
    prefetch_buffer: int = 2
    seed: int = 0
    deterministic: bool = True
    datasets_used_per_step: int = 2


class Dims(DataClass):
    batch: int = 512
    outer_bottleneck_kernel: int = 25
    inner_bottleneck_kernel: int = 49
    inner_bottleneck_features: int = 128
    pointwise_kernel: int = 5
    features: int = 2 ** 11  #
    spatial_mixing_kernel: int = 512
    mixing_features: int = 4
    pointwise_features: int = 2 ** 15
    sequence: int = 4096
    vocab: int = 256


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
    group: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None
    project: str = 'gpt'
    entity: str = 'homebrewnlp'
    median_sizes: List[int] = [64, 256, 1024]


class Optimizer(DataClass):
    momentum_dtype: str = "float32"
    momentum_type: str = "debiased"  # see src.constants.MomentumType for options
    epsilon: float = 1e-16
    learning_rate: float = 0.01
    gradient_clip: float = 0.001
    adam_beta1: float = 0.03
    adam_beta2: float = 0.003
    adam_beta3: float = 0.001
    weight_decay: float = 0.01
    warmup_end: int = 16384
    exponential_decay: float = 3e-6


class Normalization(DataClass):
    eps: float = 1e-16


class Model(DataClass):
    norm: Normalization = Normalization()
    autoregressive: bool = True
    conv_scale: float = 4.
    conv_shift: float = 8.
    storage_dtype: str = "float32"  # valid jax.numpy.dtype
    computation_dtype: str = "bfloat16"


class Training(DataClass):
    debug: bool = False
    checkpoint_path: str = "gs://homebrewnlp-eu/homebrewnlp-checkpoint"
    checkpoint_load_path: str = ""
    checkpoint_interval: float = 16384
    do_checkpoint: bool = False
    z_loss: float = 0.01
    device_steps: int = 4
    device_unroll: int = 1
    steps: int = 2 ** 16
    trace: TensorboardTrace = TensorboardTrace()


class Evaluation(DataClass):
    eos: int = 4


class Context(DataClass):
    data: DataContext = DataContext()
    optimizer: Optimizer = Optimizer()
    model: Model = Model()
    training: Training = Training()
    wandb: WandB = WandB()
    eval: Evaluation = Evaluation()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.data = DataContext()
        self.optimizer = Optimizer()
        self.model = Model()
        self.training = Training()
        self.wandb = WandB()
        self.dims = Dims()

        if config is None and 'CONFIG' in os.environ:
            with open(os.environ['CONFIG']) as f:
                cfg = f.read()
            config = yaml.safe_load(cfg)
        if config is not None:
            init_class(self, config)

        self.seed = 0
        self.global_prefix = ''

        self.name_cache: Dict[str, int] = {}
        self.name_cache_offsets: Dict[str, int] = {}
        self.parameters: Dict[str, jax.Array] = {}
        self.parameter_variance: Dict[str, float] = {}
        self.parameter_usages: Dict[str, int] = collections.defaultdict(int)
        self.prng_key = random.PRNGKey(self.seed)
        self.is_initializing = False
        self.fail_on_missing_parameter = True
        self.add_depth = False
        self.depth = 0

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
        del cfg['parameter_variance']
        return serialize(cfg)

    def __str__(self):
        return yaml.dump(self.config(), indent=4)


class WhileContext(DataClass):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.ctx = Context()
        self.current_step = jnp.ones([], dtype=jnp.uint32)
        self.data: Optional[jax.Array] = None

        if config is not None:
            self.ctx.parameters = config['parameters']
            self.current_step = config['current_step']
            self.data = config['data']

    def _serialize(self) -> dict:
        return {'parameters': self.ctx.parameters, 'current_step': self.current_step, 'data': self.data}

    @property
    def step(self):
        return int(self.current_step[0])

    def __call__(self, data: jax.Array):
        self.data = data
        return self


class WhileTrainContext(WhileContext):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.scalars = jnp.zeros([2], jnp.float64)

        if config is not None:
            self.scalars = config['scalars']
            self.ctx.parameter_variance = config['parameter_variance']

    def serialize(self):
        serialized = self._serialize()
        serialized['scalars'] = self.scalars
        serialized['parameter_variance'] = self.ctx.parameter_variance
        return serialized


class WhilePredictContext(WhileContext):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        batch_dim_size = self.ctx.dims.batch
        sequence_dim_size = self.ctx.dims.sequence
        vocab_dim_size = self.ctx.dims.vocab

        self.start_pos = jnp.zeros([batch_dim_size])
        self.stop_pos = jax.Array([sequence_dim_size] * batch_dim_size)[0]
        self.temperature = jnp.zeros([batch_dim_size])
        self.max_tokens = jax.Array([vocab_dim_size] * batch_dim_size)
        self.max_probability_mass = jax.Array([1] * batch_dim_size)
        self.typical_mass = jax.Array([1] * batch_dim_size)
        self.seed = jax.Array([0] * batch_dim_size)
        self.max_probability_to_filter = jax.Array([0] * batch_dim_size)
        self.adaptive_filter_scale = jax.Array([0] * batch_dim_size)
        self.adaptive_filter_power = jax.Array([1] * batch_dim_size)

        if config is not None:
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
