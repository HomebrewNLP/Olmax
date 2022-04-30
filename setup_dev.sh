sudo apt udpate
sudo apt install -y libpq-dev python-dev python3-dev gcc libgl1-mesa-glx ffmpeg libgl-dev python3-pip git
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[tpu]>=0.3.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y
python3 -m pip install wandb smart-open jsonpickle tpunicorn google-api-python-client google-cloud-tpu redis sqlalchemy psycopg2-binary opencv-python Pillow git+https://github.com/ytdl-org/youtube-dl.git google-cloud-storage oauth2client utils scipy gdown omegaconf pyparsing==2.4.7 einops pytorch-lightning
python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0
git clone https://github.com/CompVis/taming-transformers
mv taming-transformers script/