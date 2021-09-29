import time
import wandb

from src.context import WhileTrainContext
from src.optimizer import get_current_lr


class WandbLog:
    def __init__(self):
        self.mean_loss = 0
        self.start_time = time.time()
        self.idx = 0
        self.prev = 0

    def __call__(self, wctx: WhileTrainContext):
        curr_loss = float(wctx.current_loss) / wctx.ctx.wandb.log_frequency
        step = self.idx * wctx.ctx.wandb.log_frequency
        self.idx += 1
        self.mean_loss = ((wctx.loss / step) * self.prev + curr_loss * self.idx) / (self.prev + self.idx)  # LWMA
        self.prev += self.idx

        rate = step / (time.time() - self.start_time)
        tokens_per_day = step * 3600 * 24 * rate * wctx.ctx.dims.sizes.batch * wctx.ctx.dims.sizes.sequence

        wandb.log({"Loss/Current": curr_loss,
                   "Loss/Mean": self.mean_loss,
                   "Speed/Batches per Second": rate,
                   "Speed/Tokens per Day": tokens_per_day,
                   "Optimizer/Learning Rate": float(get_current_lr(wctx.ctx, wctx.current_step)),
                   "Optimizer/Beta1": wctx.ctx.model.momentumnet_beta},
                  step=step)