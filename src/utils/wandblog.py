import collections
import time
from typing import List

import numpy as np

from src.context import WhileTrainContext


class WandbLog:
    def __init__(self, run, device_steps: int, param_count: int, tokens_per_step: int):
        self.previous_step_time = time.time()
        self.run = run
        self.scalars = collections.defaultdict(list)
        self.device_steps = device_steps
        self.param_count = param_count
        self.tokens_per_step = tokens_per_step
        self.first_step = None

    def _log(self, prefix: str, value: float, sizes: List[int]):
        scalars = self.scalars[prefix]
        value = float(value)
        scalars.append(value)
        items = {f"{prefix}/Median{s * self.device_steps}": np.median(scalars[-s:]) for s in sizes}
        self.scalars[prefix] = scalars[-max(sizes):]
        items[f"{prefix}/Current"] = value
        return items

    def __call__(self, wctx: WhileTrainContext, step: int, current_lr) -> bool:
        rate = self.device_steps / (time.time() - self.previous_step_time)
        if self.first_step is None:
            self.first_step = step - self.device_steps
            self.previous_step_time = time.time()

        ctx = wctx.ctx
        sizes = [s // self.device_steps for s in ctx.wandb.median_sizes]

        tokens_per_day = 3600 * 24 * rate * ctx.dims.batch * ctx.dims.sequence
        items = {"Optimizer/Learning Rate": current_lr,
                 "Speed/Batches per Second": rate,
                 "Speed/Tokens per Day": tokens_per_day,
                 "Speed/Parameters * Tokens per Day": tokens_per_day * self.param_count,
                 "Speed/Tokens Seen": step * self.tokens_per_step}

        items.update(self._log("Loss", wctx.scalars[0, 0], sizes))
        items.update(self._log("Accuracy", wctx.scalars[0, 1], sizes))

        self.run.log(items, step=step)

        return any(val in (float("nan"), float("inf"), float("-inf")) for val in wctx.scalars[0, :])
