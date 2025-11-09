# Multi-stage build for minimal image size

FROM python:3.11-slim AS builder
WORKDIR /opt/install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /opt/python

FROM python:3.11-slim
ENV LAMBDA_TASK_ROOT=/var/task \
    PYTHONPATH=/var/task \
    PATH=/var/task/bin:/usr/local/bin:$PATH \
    MODEL_BUCKET=dr-classifier-models-arnav \
    MODEL_KEY=models/best_model.pth \
    MODEL_PATH=/tmp/best_model.pth
WORKDIR /var/task

COPY --from=builder /opt/python /var/task

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY app.py .
COPY model_handler.py .

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["app.handler"]