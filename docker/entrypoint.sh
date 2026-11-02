#!/usr/bin/env bash
set -e

WORKSPACE_ROOT="/workspace/aac-duac"
mkdir -p "${WORKSPACE_ROOT}/.cache/duac" "${WORKSPACE_ROOT}/checkpoints" "${WORKSPACE_ROOT}/outputs" "${WORKSPACE_ROOT}/data/libero"

# Ensure container virtualenvs at /opt/venv-* are always accessible at /workspace/aac-duac/.venv-*
if [ -d "/opt/venv-openpi" ] && [ ! -e "${WORKSPACE_ROOT}/.venv-openpi" ]; then
  ln -s /opt/venv-openpi "${WORKSPACE_ROOT}/.venv-openpi"
fi
if [ -d "/opt/venv-libero" ] && [ ! -e "${WORKSPACE_ROOT}/.venv-libero" ]; then
  ln -s /opt/venv-libero "${WORKSPACE_ROOT}/.venv-libero"
fi

# Auto-configure non-interactive LIBERO config.yaml for /workspace/aac-duac
if [ -f "${WORKSPACE_ROOT}/tools/download_libero.py" ]; then
  python3 "${WORKSPACE_ROOT}/tools/download_libero.py" --suite libero_10 >/dev/null 2>&1 || true
fi

exec "$@"

