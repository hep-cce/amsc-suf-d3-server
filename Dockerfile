# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/tritonserver:24.05-py3
# nvcc version: 12.4 ## nvcc --version
# cudnn version: 9.1.0  ## find / -name "libcudnn*" 2>/dev/null
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver

LABEL description="Triton Server backend with other dependencies for Tracking-as-a-Service"
LABEL version="1.0"

# Install dependencies
# Update the CUDA Linux GPG Repository Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# See also https://root.cern.ch/build-prerequisites
# https://root.cern/install/dependencies/#ubuntu-and-other-debian-based-distributions
RUN apt-get update -y && apt-get install -y \
    build-essential curl git freeglut3-dev libfreetype6-dev libpcre3-dev \
    libtbb-dev ninja-build time tree \
    python3 python3-dev python3-pip python3-numpy \
    rsync zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev libb64-dev graphviz gfortran  libglu1-mesa-dev  \
    libfftw3-dev libcfitsio-dev libgraphviz-dev \
    libavahi-compat-libdnssd-dev libldap2-dev libxml2-dev libkrb5-dev \
     qtwebengine5-dev nlohmann-json3-dev libmysqlclient-dev libxxhash-dev \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

# Manual builds for specific packages
# Install CMake v3.29.4
ARG CMAKE_VERSION=3.29.4
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    | ${UNPACK_TO_SRC} \
  && rsync -ru src/ ${PREFIX} \
  && cd /tmp && rm -rf /tmp/src


# Environment variables
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV TORCH_CUDA_ARCH_LIST="80"
ENV PYTHONNOUSERSITE=True

RUN pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

RUN pip3 install torch_geometric lightning>=2.2

# FRNN
RUN cd /tmp/ \
	&& git clone https://github.com/asnaylor/prefix_sum.git \
    && git clone https://github.com/xju2/FRNN.git \
	&& cd prefix_sum \
	&& NVCC_FLAGS="--std=c++17 -gencode=arch=compute_80,code=sm_80" \
		python setup.py install \
    && cd /tmp/FRNN \
    && NVCC_FLAGS="--std=c++17 -gencode=arch=compute_80,code=sm_80" \
		python setup.py install && \
	rm -rf /tmp/prefix_sum && rm -rf /tmp/FRNN


RUN  cd / && \
     git clone -b cerati/ng2-feature-extension-triton https://github.com/cerati/nugraph.git && \
     pip3 install --no-deps -e ./nugraph && \
     pip3 install matplotlib pynvml~=11.5 seaborn~=0.13 scikit-learn~=1.5 pytorch_lightning~=2.3 pynuml~=23.11
