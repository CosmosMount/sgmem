#!/usr/bin/env bash
# Build the project image and verify its CPU and NVIDIA GPU execution.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo -v
sudo docker version
build_args=()
if [[ -n "${SGMEM_BUILD_PROXY:-}" ]]; then
  # The proxy runs on the Linux host loopback; build steps need host networking.
  build_args+=(--network=host --build-arg "HTTP_PROXY=$SGMEM_BUILD_PROXY"
               --build-arg "HTTPS_PROXY=$SGMEM_BUILD_PROXY"
               --build-arg "http_proxy=$SGMEM_BUILD_PROXY"
               --build-arg "https_proxy=$SGMEM_BUILD_PROXY"
               --build-arg "NO_PROXY=localhost,127.0.0.1"
               --build-arg "no_proxy=localhost,127.0.0.1")
fi
sudo docker build "${build_args[@]}" -t sgmem:local .
sudo docker run --rm sgmem:local --help
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint python sgmem:local \
  -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "Container cannot access the NVIDIA GPU"; print("GPU:", torch.cuda.get_device_name(0))'
printf '\nProject image built and NVIDIA GPU verified.\n'
