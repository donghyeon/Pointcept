FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git vim tmux htop python3-pip python3-dev ninja-build libgl1

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install h5py pyyaml sharedarray tensorboard tensorboardx yapf==0.40.1 addict einops scipy plyfile termcolor timm polars

RUN pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

RUN pip install torch-geometric spconv-cu117

WORKDIR /home

RUN git clone https://github.com/donghyeon/Pointcept.git

WORKDIR /home/Pointcept/libs/pointops

ARG TORCH_CUDA_ARCH_LIST=8.6

RUN python setup.py install

WORKDIR /home/Pointcept

RUN pip install open3d

CMD /bin/bash
