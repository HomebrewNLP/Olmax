import time

import numpy as np

from src.context import WhileContext


class WandbLog:
    def __init__(self, run):
        self.start_time = time.time()
        self.run = run
        self.last_losses = []
        self.last_accuracy = []
        self.idx = 0

    def __call__(self, wctx: WhileContext, current_lr):
        self.idx += 1
        ctx = wctx.ctx
        curr_loss = wctx.loss / ctx.training.device_steps
        step = self.idx * ctx.wandb.log_frequency * ctx.training.device_steps
        sizes = [s // ctx.training.device_steps for s in ctx.wandb.median_sizes]
        self.last_losses.append(curr_loss.astype(float))
        self.last_accuracy.append((wctx.top_loss / ctx.training.device_steps).astype(float))
        self.last_losses = self.last_losses[-max(sizes):]
        self.last_accuracy = self.last_accuracy[-max(sizes):]
        self.run.log({f"Loss/Median{s}": np.median(self.last_losses[-s:]) for s in sizes}, step=step)
        self.run.log({f"Accuracy/Median{s}": np.median(self.last_accuracy[-s:]) for s in sizes}, step=step)

        rate = step / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * ctx.dims.sizes.batch * ctx.dims.sizes.sequence

        self.run.log({"Loss/Current": self.last_losses[-1], "Accuracy/Current": self.last_accuracy[-1],
                      "Speed/Batches per Second": rate, "Speed/Tokens per Day": tokens_per_day,
                      "Optimizer/Learning Rate": current_lr.astype(float), "Optimizer/Beta1": ctx.optimizer.adam_beta1,
                      "Optimizer/Beta2": ctx.optimizer.adam_beta2}, step=step)
