FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Build args
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV WORKER_MODEL_DIR=/app/model
ENV WORKER_USE_CUDA=True

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV WORKER_DIR=/app
RUN mkdir ${WORKER_DIR}
WORKDIR ${WORKER_DIR}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install basic tools
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client cmake g++ ninja-build && \
    apt-get install -y libaio-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user ${WORKER_DIR}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
ENV SHELL=/bin/bash

# ✅ Install PyTorch and DeepSpeed before other dependencies
RUN pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ninja

# Clean any preinstalled DeepSpeed
RUN pip uninstall -y deepspeed || true

# Install DeepSpeed with custom ops
RUN DS_BUILD_OPS=1 pip install deepspeed

# ✅ Now install the rest of your dependencies
COPY builder/requirements.txt ${WORKER_DIR}/requirements.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements.txt && \
    rm ${WORKER_DIR}/requirements.txt

COPY builder/requirements_audio_enhancer.txt ${WORKER_DIR}/requirements_audio_enhancer.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements_audio_enhancer.txt && \
    rm ${WORKER_DIR}/requirements_audio_enhancer.txt

# Confirm DeepSpeed works
RUN python3 -c "import deepspeed; print(deepspeed.__version__)"

# Install Git LFS and pull models
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install -y git-lfs
RUN git lfs install

# Clone models
RUN git clone https://huggingface.co/coqui/XTTS-v2 ${WORKER_MODEL_DIR}/xttsv2
RUN git clone https://huggingface.co/ResembleAI/resemble-enhance ${WORKER_MODEL_DIR}/audio_enhancer

# Copy worker code
ADD src ${WORKER_DIR}

ENV RUNPOD_DEBUG_LEVEL=INFO

# Entry point
CMD python3 -u ${WORKER_DIR}/rp_handler.py --model-dir="${WORKER_MODEL_DIR}"
