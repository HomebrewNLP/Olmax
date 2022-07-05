  GNU nano 4.8                                                                                                                                                                          tmp/Dockerfile                                                                                                                                                                                     FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TORCHVER=v1.12.0
ENV VISIONVER=v0.13.0
ENV AUDIOVER=v0.12.0

ENV TZ=Europe/Berlin
ENV PYENV_ROOT="/root/.pyenv"


# Build dependencies for Python and PyTorch
RUN apt update && \
    apt install -y autoconf automake build-essential cpio curl ffmpeg g++ gcc git gosu libbz2-dev libffi-dev libgl-dev libgl1-mesa-glx liblapack-dev liblapacke-dev liblzma-dev libncursesw5-dev libpng-dev libpq-dev libpython3-all-dev libpython3-dev libreadline-dev libsox-dev libsqlite3-dev libssl-dev libtool libxml2-dev libxmlsec1-dev llvm make nano npm pciutils pkg-config pyt>    curl https://pyenv.run | bash && \
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH" && \
    eval "$(pyenv init -)" && \
    pyenv install 3.10.5  && \
    python3 -m pip install --upgrade pip &&\
    python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y &&\
    python3 -m pip install --upgrade --ignore-installed pyyaml &&\
    python3 -m pip install --upgrade cmake==3.18.4 moc ninja onnx onnxruntime six wheel yapf &&\
    python3 -m pip install --upgrade --force-reinstall protobuf==3.20.1


# Build PyTorch (patch to bypass CUDA and CUB version checks)
RUN git clone -b ${TORCHVER} --recursive https://github.com/pytorch/pytorch &&\
    git clone -b ${VISIONVER} --recursive https://github.com/pytorch/vision.git &&\
    git clone -b ${AUDIOVER} --recursive https://github.com/pytorch/audio.git

RUN cd /pytorch && \
    sed -i -e "/^#ifndef THRUST_IGNORE_CUB_VERSION_CHECK$/i #define THRUST_IGNORE_CUB_VERSION_CHECK" /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/config.h && \
    cat /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/config.h && \
    sed -i -e "/^if(DEFINED GLIBCXX_USE_CXX11_ABI)/i set(GLIBCXX_USE_CXX11_ABI 1)" CMakeLists.txt && \
    python3 -m pip install -r requirements.txt && \
    USE_NCCL=OFF python3 setup.py build && \
    python3 setup.py bdist_wheel

RUN cd /vision && \
    git submodule update --init --recursive && \
    python3 -m pip install /pytorch/dist/*.whl && \
    python3 setup.py build && \
    python3 setup.py bdist_wheel

RUN cd /audio && \
    git submodule update --init --recursive && \
    python3 setup.py build && \
    python3 setup.py bdist_wheel


# Install runtime dependencies
RUN python3 -m pip install --upgrade boto3 einops fastapi gdown git+https://github.com/ytdl-org/youtube-dl.git google-api-python-client google-cloud-storage google-cloud-tpu jsonpickle oauth2client omegaconf opencv-python Pillow psycopg2-binary pydantic pyparsing==2.4.7 pytorch-lightning pyyaml redis scipy smart-open[gcs] sqlalchemy tpunicorn transformers utils uvicorn wandb >    git clone https://github.com/HomebrewNLP/HomebrewNLP-Jax &&\
    git clone https://github.com/CompVis/taming-transformers &&\
    mv taming-transformers HomebrewNLP-Jax/script &&\
    cd HomebrewNLP-Jax/script



