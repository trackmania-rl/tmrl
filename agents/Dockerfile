# syntax=docker/dockerfile:1.0.0-experimental
# (experimental ssh forwarding: https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066)

# Build with DOCKER_BUILDKIT=1 docker build .


# Cuda image. Can by any image with CUDA. If "base-x11" we will build with X11 support.
ARG CUDA_BASE="nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"

# OpenAI-Gym image. Can by any image with Pytorch, Gym, etc. Special values: gym, gym-mujoco, gym-avenue
ARG GYM_BASE="gym"



FROM nvidia/opengl:1.0-glvnd-devel-ubuntu18.04 as cuda-x11

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    libglm-dev libx11-dev libegl1-mesa-dev \
    libpng-dev xorg-dev cmake libjpeg-dev \
    build-essential pkg-config git curl wget automake libtool ca-certificates \
    x11-apps imagemagick

RUN git clone https://github.com/glfw/glfw.git && cd glfw && mkdir build && cd build && cmake .. && make &&  make install



FROM ${CUDA_BASE} as pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    screen \
    htop \
    tmux \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PWD=/app
WORKDIR $PWD

RUN curl -so miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
  && bash miniconda.sh -b -p miniconda \
  && rm miniconda.sh

ENV PATH="$PWD/miniconda/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir torch==1.4.0 torchvision==0.5.0



FROM pytorch as gym

# OpenAI Gym commit hash. Shouldn't be empty.
ARG GYM_REV="c33cfd8b2cc8cac6c346bc2182cd568ef33b8821"
ARG GYM_FEATURES=''
RUN git clone https://github.com/openai/gym \
 && cd gym \
 && git reset --hard $GYM_REV \
 && pip --no-cache-dir install ."${GYM_FEATURES}" \
 && cd .. && rm -r gym



FROM pytorch as gym-mujoco

# mujoco-py requirements https://github.com/openai/mujoco-py/blob/master/Dockerfile
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN mkdir -p .mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d .mujoco \
    && mv .mujoco/mujoco200_linux .mujoco/mujoco200 \
    && rm mujoco.zip

# will compile even without a valid mjkey
ARG MJ_KEY=""
RUN echo "$MJ_KEY" > .mujoco/mjkey.txt

ENV LD_LIBRARY_PATH "$PWD/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH "/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"
ENV MUJOCO_PY_MJKEY_PATH "$PWD/.mujoco/mjkey.txt"
ENV MUJOCO_PY_MUJOCO_PATH "$PWD/.mujoco/mujoco200"

# OpenAI Gym commit hash. Shouldn't be empty.
ARG GYM_REV="c33cfd8b2cc8cac6c346bc2182cd568ef33b8821"
ARG GYM_FEATURES='[mujoco]'
RUN git clone https://github.com/openai/gym \
 && cd gym \
 && git reset --hard $GYM_REV \
 && pip --no-cache-dir install ."${GYM_FEATURES}" \
 && cd .. && rm -r gym

# we need to change the permissions of mujoco_py/generated because mujoco-py will fail if it can't modifiy this directory
RUN printf "\
try: import mujoco_py, os \n\
except: exit() \n\
p = os.path.join(os.path.dirname(mujoco_py.__file__), 'generated') \n\
print(p) \n\
os.remove(os.path.join(p, 'mujocopy-buildlock')) \n\
os.chmod(p, 0o777) \n" | python



FROM gym as gym-avenue

# download Avenue assets
ENV AVENUE_ASSETS $PWD/avenue_assets
RUN mkdir avenue_assets \
  && chmod 777 -R avenue_assets \
  && pip --no-cache-dir install gdown \
  && apt-get update && apt-get install -y --no-install-recommends unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

#RUN mkdir avenue_assets/avenue_follow_car-linux \
#  && gdown -O avenue.zip --id 1eRKQaRxp2dJL9krKviqyecNv5ikFnMrC \
#  && unzip avenue.zip -d avenue_assets/avenue_follow_car-linux \
#  && rm avenue.zip \
#  && chmod 777 -R avenue_assets

ARG AVENUE_REV=master
RUN git clone https://github.com/elementai/avenue avenue \
  && cd avenue \
  && git reset --hard ${AVENUE_REV?} \
  && pip --no-cache-dir install -e .

# download Avenue assets
#RUN mkdir avenue_assets
#ENV AVENUE_ASSETS avenue_assets
#RUN python -c 'import avenue; avenue.download("AvenueCar")'
#RUN chmod 777 -R avenue_assets



FROM ${GYM_BASE}

# installing dependencies first to allow them to be cached
COPY setup.py ./
RUN python setup.py egg_info && pip install -r *.egg-info/requires.txt && rm -r setup.py *.egg-info

COPY . agents

RUN pip --no-cache-dir install -e agents

# optional wandb installation (we do this last because old versions break quickly so we don't want them to get cached)
RUN pip --no-cache-dir install wandb --upgrade

