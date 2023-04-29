from enum import Enum


class MomentumType(Enum):
    heavyball = "heavyball"
    nesterov = "nesterov"
    debiased = "debiased"
    ema = "ema"


class ParallelAxes(Enum):
    model = "model_parallel"


class SparseAccess(Enum):
    read = "read"
    write = "write"