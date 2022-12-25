from enum import StrEnum


class MomentumType(StrEnum):
    heavyball = "heavyball"
    nesterov = "nesterov"
    debiased = "debiased"
    ema = "ema"


class ParallelAxes(StrEnum):
    model = "model_parallel"
