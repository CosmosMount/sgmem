#!/usr/bin/env bash
# Configure this workstation's local HTTP proxy for Docker and verify the image.
set -euo pipefail
cd "$(dirname "$0")/.."
proxy=${SGMEM_DOCKER_PROXY:-http://127.0.0.1:7890}
if [[ ! "$proxy" =~ ^http://127\.0\.0\.1:[0-9]+$ ]]; then
  printf 'Expected a local HTTP proxy like http://127.0.0.1:7890; set SGMEM_DOCKER_PROXY accordingly.\n' >&2
  exit 1
fi
status=$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 15 \
  --noproxy '' --proxy "$proxy" https://registry-1.docker.io/v2/)
if [[ "$status" != 401 && "$status" != 200 ]]; then
  printf 'Docker Hub via the local proxy returned HTTP %s; check the proxy first.\n' "$status" >&2
  exit 1
fi
printf 'Configuring docker.service to use the local HTTP proxy and restarting Docker.\n'
sudo -v
tmp_dir=$(mktemp -d)
trap 'rm -rf -- "$tmp_dir"' EXIT
printf '[Service]\nEnvironment="HTTP_PROXY=%s"\nEnvironment="HTTPS_PROXY=%s"\nEnvironment="NO_PROXY=localhost,127.0.0.1,::1"\n' "$proxy" "$proxy" > "$tmp_dir/sgmem-proxy.conf"
target=/etc/systemd/system/docker.service.d/sgmem-proxy.conf
if sudo test -e "$target" && ! sudo cmp -s "$tmp_dir/sgmem-proxy.conf" "$target"; then
  printf 'Existing %s differs; review it manually instead of overwriting it.\n' "$target" >&2
  exit 1
fi
sudo install -m 0755 -d /etc/systemd/system/docker.service.d
sudo install -m 0644 "$tmp_dir/sgmem-proxy.conf" "$target"
sudo systemctl daemon-reload
sudo systemctl restart docker
SGMEM_BUILD_PROXY="$proxy" bash scripts/verify-docker.sh
