FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Base service deps (can extend as needed)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Clone and install Deep Lagrangian Networks
ARG DELAN_REPO=https://github.com/milutter/deep_lagrangian_networks.git
ARG DELAN_REF=master
RUN git clone --depth 1 --branch "${DELAN_REF}" "${DELAN_REPO}" /workspace/deep_lagrangian_networks
WORKDIR /workspace/deep_lagrangian_networks
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi \
 && pip install --no-cache-dir -e .

# Return to app workspace and bring in local scaffolding (scripts/config/src are optional)
WORKDIR /workspace/app
COPY src ./src
COPY scripts ./scripts
COPY config ./config

CMD ["python", "scripts/run_delan.py"]