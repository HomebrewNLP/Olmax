python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y
python3 -m pip install wandb smart-open[gcs] jsonpickle sharedutils
python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0 protobuf==3.20.1
