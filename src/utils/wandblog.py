import time
import types
import typing

import numpy as np
from jax import numpy as jnp

from src.context import WhileTrainContext


class WandbLog:
    def __init__(self, run, device_steps: int, param_count: int, tokens_per_step: int):
        self.start_time = time.time()
        self.run = run
        self.losses = []
        self.accuracies = []
        self.loss_medians = []
        self.device_steps = device_steps
        self.param_count = param_count
        self.tokens_per_step = tokens_per_step
        self.first_step = None

    def __call__(self, wctx: WhileTrainContext, step: int, current_lr) -> bool:
        ctx = wctx.ctx
        curr_loss = wctx.loss[0]
        sizes = [s // self.device_steps for s in ctx.wandb.median_sizes]
        self.losses.append(curr_loss.astype(float))
        self.accuracies.append((wctx.accuracy[0]).astype(float))
        self.loss_medians.append(np.median(self.losses[-max(sizes):]))
        self.losses = self.losses[-max(sizes):]
        self.accuracies = self.accuracies[-max(sizes):]
        self.loss_medians = self.loss_medians[-max(sizes):]

        self.run.log({f"Loss/Median{s * self.device_steps}": np.median(self.losses[-s:]) for s in sizes}, step=step)
        self.run.log({f"Accuracy/Median{s * self.device_steps}": np.median(self.accuracies[-s:]) for s in sizes},
                     step=step)

        if self.first_step is None:
            self.first_step = step - self.device_steps
        rate = (step - self.first_step) / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * ctx.dims.batch * ctx.dims.sequence

        self.run.log({"Loss/Current": self.losses[-1], "Accuracy/Current": self.accuracies[-1],
                      "Speed/Batches per Second": rate, "Speed/Tokens per Day": tokens_per_day,
                      "Optimizer/Learning Rate": current_lr, "Optimizer/Beta1": ctx.optimizer.adam_beta1,
                      "Optimizer/Beta2": ctx.optimizer.adam_beta2,
                      "Speed/Parameters * Tokens per Day": tokens_per_day * self.param_count,
                      "Speed/Tokens Seen": step * self.tokens_per_step
                      }, step=step)

        return self.losses[-1] in (float("nan"), float("inf"), float("-inf"))

    def log_params(self, device: int, params: typing.Dict[str, jnp.ndarray], step: int,
                   functions: typing.Tuple[types.FunctionType] = (jnp.mean, jnp.std, jnp.max, jnp.min)):
        for fn in functions:
            self.run.log({f"Weight Statistics/Device={device} Fn={fn.__name__} Param={key.replace('/', '-')}": float(fn(val))
                          for key, val in params.items()}, step=step)
