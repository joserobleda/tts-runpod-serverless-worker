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

# Install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client cmake g++ ninja-build && \
    apt-get install -y libaio-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
                && chown -R user:user ${WORKER_DIR}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV SHELL=/bin/bash

# Create necessary directories for PyTorch/Triton optimization
RUN mkdir -p /home/user/.triton/autotune && \
    mkdir -p /home/user/.cache/matplotlib && \
    chown -R user:user /home/user/.triton /home/user/.cache

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt ${WORKER_DIR}/requirements.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements.txt && \
    rm ${WORKER_DIR}/requirements.txt

# Install Python dependencies (Worker Template)
COPY builder/requirements_audio_enhancer.txt ${WORKER_DIR}/requirements_audio_enhancer.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements_audio_enhancer.txt && \
    rm ${WORKER_DIR}/requirements_audio_enhancer.txt

# Fetch the model
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs
RUN git lfs install

# Fetch XTTSv2 model
RUN git clone https://huggingface.co/coqui/XTTS-v2 ${WORKER_MODEL_DIR}/xttsv2
RUN git clone https://huggingface.co/ResembleAI/resemble-enhance ${WORKER_MODEL_DIR}/audio_enhancer

# Switch back to root to add files and set permissions
USER root

# Add src files and set proper permissions
ADD src ${WORKER_DIR}
RUN chown -R user:user ${WORKER_DIR} && \
    chmod +x ${WORKER_DIR}/startup.sh

# Switch back to user for runtime
USER user

# Set environment variables to reduce warnings
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV MPLBACKEND=Agg
ENV TRITON_CACHE_DIR=/home/user/.triton

CMD ${WORKER_DIR}/startup.sh