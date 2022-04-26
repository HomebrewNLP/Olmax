python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[tpu]>=0.3.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y
sudo apt install -y libpq-dev python-dev python3-dev gcc postgresql postgresql-dev
python3 -m pip install wandb smart-open jsonpickle tpunicorn google-api-python-client google-cloud-tpu redis sqlalchemy psycopg2-binary
python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0 git+https://github.com/CompVis/taming-transformers
