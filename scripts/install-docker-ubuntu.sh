#!/usr/bin/env bash
# Install Docker Engine and NVIDIA Container Toolkit on Ubuntu 22.04/24.04.
# Run interactively as the workstation user; sudo will request a password.
set -euo pipefail

. /etc/os-release
if [[ "${ID:-}" != ubuntu || ( "${VERSION_ID:-}" != 22.04 && "${VERSION_ID:-}" != 24.04 ) ]]; then
  printf 'This script supports Ubuntu 22.04 and 24.04; found %s %s.\n' "${ID:-unknown}" "${VERSION_ID:-unknown}" >&2
  exit 1
fi
if [[ "$(dpkg --print-architecture)" != amd64 ]]; then
  printf 'This script targets amd64; review the official instructions for other architectures.\n' >&2
  exit 1
fi
for path in /etc/apt/sources.list.d/docker.sources /etc/apt/sources.list.d/docker.list \
            /etc/apt/sources.list.d/nvidia-container-toolkit.list /etc/apt/keyrings/docker.asc \
            /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /etc/docker/daemon.json; do
  if [[ -e "$path" ]]; then
    printf 'Existing configuration %s requires manual review; it will not be overwritten.\n' "$path" >&2
    exit 1
  fi
done
if systemctl is-active --quiet docker; then
  printf 'Docker is already running; review its configuration before restarting it.\n' >&2
  exit 1
fi
for package in docker.io docker-ce docker-ce-cli docker-compose docker-compose-v2 docker-doc docker-buildx \
               docker-buildx-plugin docker-compose-plugin podman-docker containerd containerd.io runc \
               nvidia-container-toolkit; do
  if [[ "$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)" == installed ]]; then
    printf 'Conflicting package %s is installed. Review it before proceeding; nothing was removed.\n' "$package" >&2
    exit 1
  fi
done

printf 'This installs Docker Engine and NVIDIA Container Toolkit, adds their apt repositories, and configures/restarts Docker.\n'
sudo -v
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg

# Official repository keys and definitions. Temporary downloads are removed on exit.
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$tmp_dir/docker.asc"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$tmp_dir/nvidia.asc"
gpg --dearmor -o "$tmp_dir/nvidia.gpg" "$tmp_dir/nvidia.asc"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "$tmp_dir/nvidia.list"
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "$tmp_dir/nvidia.list" > "$tmp_dir/nvidia-signed.list"
sudo install -m 0755 -d /etc/apt/keyrings
sudo install -m 0644 "$tmp_dir/docker.asc" /etc/apt/keyrings/docker.asc
sudo install -m 0644 "$tmp_dir/nvidia.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: amd64\nSigned-By: /etc/apt/keyrings/docker.asc\n' "${VERSION_CODENAME}" > "$tmp_dir/docker.sources"
sudo install -m 0644 "$tmp_dir/docker.sources" /etc/apt/sources.list.d/docker.sources
sudo install -m 0644 "$tmp_dir/nvidia-signed.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo docker version
printf '\nDocker and NVIDIA toolkit installed. Run the CPU and GPU checks in docs/hpc-deployment.md.\n'
