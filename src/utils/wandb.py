import time

import numpy as np

from src.context import WhileContext


class WandbLog:
    def __init__(self, run):
        self.start_time = time.time()
        self.run = run
        self.losses = []
        self.accuracies = []
        self.idx = 0
        self.loss_medians = []

    def __call__(self, wctx: WhileContext, current_lr) -> bool:
        self.idx += 1
        ctx = wctx.ctx
        device_steps = ctx.training.device_steps
        curr_loss = wctx.loss / device_steps
        step = self.idx * ctx.wandb.log_frequency * device_steps
        sizes = [s // ctx.training.device_steps for s in ctx.wandb.median_sizes]
        self.losses.append(curr_loss.astype(float))
        self.accuracies.append((wctx.top_loss / device_steps).astype(float))
        self.loss_medians.append(np.median(self.losses[-max(sizes):]))
        self.losses = self.losses[-max(sizes):]
        self.accuracies = self.accuracies[-max(sizes):]
        self.loss_medians = self.loss_medians[-max(sizes):]
        self.run.log({f"Loss/Median{s * device_steps}": np.median(self.losses[-s:]) for s in sizes}, step=step)
        self.run.log({f"Accuracy/Median{s * device_steps}": np.median(self.accuracies[-s:]) for s in sizes}, step=step)

        rate = step / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * ctx.dims.sizes.batch * ctx.dims.sizes.sequence

        self.run.log({"Loss/Current": self.losses[-1], "Accuracy/Current": self.accuracies[-1],
                      "Speed/Batches per Second": rate, "Speed/Tokens per Day": tokens_per_day,
                      "Optimizer/Learning Rate": current_lr.astype(float), "Optimizer/Beta1": ctx.optimizer.adam_beta1,
                      "Optimizer/Beta2": ctx.optimizer.adam_beta2}, step=step)

        return self.loss_medians[0] < self.loss_medians[-1]
