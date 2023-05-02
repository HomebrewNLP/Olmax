python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install wandb smart-open[gcs] jsonpickle sharedutils
