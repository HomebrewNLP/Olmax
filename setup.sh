python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorflow==2.4.2 jsonpickle
python3 -m pip install --upgrade "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile
python3 -m pip install tensorboard==2.4.0
python3 -m pip install tensorboard-plugin-profile
