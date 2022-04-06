import time


class WandbLog:
    def __init__(self, run):
        self.mean_loss = 0
        self.start_time = time.time()
        self.idx = 0
        self.prev = 0
        self.run = run

    def __call__(self, wctx, current_lr):
        curr_loss = wctx.loss / wctx.ctx.training.device_steps
        step = self.idx * wctx.ctx.wandb.log_frequency * wctx.ctx.training.device_steps
        self.idx += 1
        self.mean_loss = (self.mean_loss * self.prev + curr_loss * self.idx) / (self.prev + self.idx)  # LWMA
        self.prev += self.idx

        rate = step / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * wctx.ctx.dims.sizes.batch * wctx.ctx.dims.sizes.sequence

        self.run.log({"Loss/Current": curr_loss.astype(float),
                      "Loss/Mean": self.mean_loss.astype(float),
                      "Accuracy/Current": (wctx.top_loss / wctx.ctx.training.device_steps).astype(float),
                      "Speed/Batches per Second": rate,
                      "Speed/Tokens per Day": tokens_per_day,
                      "Optimizer/Learning Rate": current_lr.astype(float),
                      "Optimizer/Beta1": wctx.ctx.optimizer.adam_beta1,
                      "Optimizer/Beta2": wctx.ctx.optimizer.adam_beta2},
                     step=step)
