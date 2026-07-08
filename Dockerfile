# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/tritonserver:26.05-py3
# nvcc version: 13.2 ## nvcc --version
# cudnn version: 9.22.0  ## find / -name "libcudnn*" 2>/dev/null
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

ARG LIB_WITH_CUDA=ON
ARG NPROC=6

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

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Environment variables
ENV FORCE_CUDA=1
ENV TORCH_SITE_PATH="/opt/torch"
ENV TORCH_LIB_PATH="${TORCH_SITE_PATH}/lib"
ENV LD_LIBRARY_PATH="${TORCH_LIB_PATH}:/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV TORCH_CUDA_ARCH_LIST="80"
ENV PYTHONNOUSERSITE=True

# Manual builds for specific packages
# Install CMake v3.29.4
ARG CMAKE_VERSION=3.29.4
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    | ${UNPACK_TO_SRC} \
  && rsync -ru src/ ${PREFIX} \
  && cd /tmp && rm -rf /tmp/src

RUN pip3 install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
RUN ln -s "$(python3 -c 'import os, torch; print(os.path.dirname(torch.__file__))')" "${TORCH_SITE_PATH}"
RUN pip3 install pyg_lib -f https://data.pyg.org/whl/torch-2.12.0+cu132.html

RUN mkdir -p /torch_geometric/lib \
  && cd /tmp \
  && for repo in pytorch_cluster pytorch_scatter pytorch_spline_conv pytorch_sparse; do \
       git clone https://github.com/rusty1s/${repo}.git; \
       cd ${repo}; \
       pip3 install .; \
       mkdir build; \
       cd build; \
       cmake -DCMAKE_PREFIX_PATH=${TORCH_SITE_PATH} -DWITH_CUDA=${LIB_WITH_CUDA} ..; \
       make -j ${NPROC}; \
       mv ./*.so /torch_geometric/lib/; \
       cd /tmp; \
       rm -rf ${repo}; \
     done

RUN pip3 install torch_geometric "lightning>=2.2" numba

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
     pip3 install matplotlib pynvml~=11.5 seaborn~=0.13 scikit-learn~=1.5 pynuml~=23.11

RUN python3 <<EOF
import os, re
path = "/nugraph"
for root, _, files in os.walk(path):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            with open(fpath, "r") as r:
                content = r.read()
            if "BaseTransform" in content:
                # Replace __call__ with forward only in classes inheriting BaseTransform
                new_content = re.sub(r"def\s+__call__\s*\(", "def forward(", content)
                if new_content != content:
                    with open(fpath, "w") as w:
                        w.write(new_content)
                    print(f"Updated: {fpath}")
EOF

ENV LD_PRELOAD="/torch_geometric/lib/libtorchscatter.so /torch_geometric/lib/libtorchsparse.so /torch_geometric/lib/libtorchcluster.so /torch_geometric/lib/libtorchsplineconv.so"
