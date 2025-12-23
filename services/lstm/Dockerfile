# Official TF GPU image (includes TensorFlow + CUDA/cuDNN runtime inside the container)
FROM tensorflow/tensorflow:2.16.1-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/app

# Optional: basic utilities (git etc.)—nice to have
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements.txt

# You mount code/data via docker-compose; keep container alive for interactive work
WORKDIR /workspace
CMD ["bash"]
