FROM  pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV PYENV_ROOT="/root/.pyenv"
RUN apt update ; apt install -y libpq-dev python-dev python3-dev gcc libgl1-mesa-glx ffmpeg libgl-dev python3-pip git curl make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN curl https://pyenv.run | bash
RUN export PYENV_ROOT="$HOME/.pyenv" && \
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH" && \
    eval "$(pyenv init -)" && \
    pyenv install 3.10.5
RUN python3 -m pip install --upgrade pip &&\
    python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y &&\
    python3 -m pip install wandb smart-open[gcs] jsonpickle tpunicorn google-api-python-client google-cloud-tpu redis sqlalchemy psycopg2-binary opencv-python Pillow git+https://github.com/ytdl-org/youtube-dl.git google-cloud-storage oauth2client utils scipy gdown omegaconf pyparsing==2.4.7 einops pytorch-lightning fastapi uvicorn pydantic transformers boto3&&\
    python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0 protobuf==3.20.1 &&\
    git clone https://github.com/CompVis/taming-transformers
RUN git clone https://github.com/HomebrewNLP/HomebrewNLP-Jax/ &&\
    mv taming-transformers HomebrewNLP-Jax/script &&\
    cd HomebrewNLP-Jax/script
