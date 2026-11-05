FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/src
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/sgmem
COPY pyproject.toml README.md requirements-docker.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-docker.txt \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && python -m pip check \
    && transformers --help >/dev/null

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "sgmem"]
CMD ["--help"]
