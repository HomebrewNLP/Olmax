# Olmax
Optimized Language-Model (in jax)

## First steps

After SSHing into a TPU, it's recommended to run `bash setup.sh` as root to install the correct versions of libraries.\
This isn't done via a `requirements.txt` as some libraries have to be removed while others require a very specific installation order.

Once that's done, you can run `python3 model.py` to start whatever model is configured in `context.py`.\
To change the config without touching the code, you could also run `python3 model.py config.json` and perform your changes in `config.json`.

It's possible to get a tensorboard trace of memory and operations by changing `ctx.training.trace.do_trace` to `true`. With that, a file in the possibly newly created folder named `ctx.training.trace.output_path` will be created containing the trace.\
Using this trace, you can start a tensorboard server and inspect the current model performance. Something as simple as `tensorboard --logdir trace --host 0.0.0.0 --port 6006` works perfectly fine.\
If the tensorboard doesn't show up, it's likely that the firewall is misconfigured. One easy way to fix this is to run `gcloud compute firewall-rules create --network=default allow-tensorboard-tcp --allow=tcp:6006`, which creates a new firewall rule allowing anyone to access, and with that force you to pay for what's hosted on, this port on the TPU.\