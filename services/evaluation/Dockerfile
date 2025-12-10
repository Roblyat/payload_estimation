FROM python:3.11-slim

WORKDIR /workspace/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY config ./config

CMD ["python", "scripts/run_evaluation.py"]
