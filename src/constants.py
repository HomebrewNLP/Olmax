from enum import StrEnum


class MomentumType(StrEnum):
    heavyball = "heavyball"
    nesterov = "nesterov"
    debiased = "debiased"
    ema = "ema"


class ParallelAxes(StrEnum):
    model = "model_parallel"
    # data = "data_parallel"  # discontinued in favor of pure model parallel
