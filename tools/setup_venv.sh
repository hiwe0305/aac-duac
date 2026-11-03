#!/usr/bin/env bash
# Automated Local Virtualenv Setup (`.venv-openpi` + `.venv-libero`) for O-LONG on Linux GPU Server
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== [1/5] Checking/Installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:${PATH}"
fi

echo "=== [2/5] Cloning external repositories (openpi, LIBERO, aac-libero) ==="
mkdir -p "${PROJECT_ROOT}/external"
if [ ! -d "${PROJECT_ROOT}/external/openpi" ]; then
  git clone --recurse-submodules --depth 1 https://github.com/Physical-Intelligence/openpi.git "${PROJECT_ROOT}/external/openpi"
fi
if [ ! -d "${PROJECT_ROOT}/external/LIBERO" ]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${PROJECT_ROOT}/external/LIBERO"
  touch "${PROJECT_ROOT}/external/LIBERO/libero/__init__.py"
fi
if [ ! -d "${PROJECT_ROOT}/external/aac-libero" ]; then
  git clone --depth 1 https://github.com/Adaptive-Action-Chunking/libero.git "${PROJECT_ROOT}/external/aac-libero"
fi

echo "=== [3/5] Building .venv-openpi (Python 3.11.9 + JAX CUDA + OpenPI) ==="
pushd "${PROJECT_ROOT}/external/openpi" >/dev/null
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv-openpi" uv venv --python 3.11.9 "${PROJECT_ROOT}/.venv-openpi"
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv-openpi" GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv-openpi" GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "${PROJECT_ROOT}/.venv-openpi/bin/python" --no-deps -e packages/openpi-client -e .
SITE_PKGS="$("${PROJECT_ROOT}/.venv-openpi/bin/python" -c "import transformers, os; print(os.path.dirname(transformers.__file__))")"
cp -r src/openpi/models_pytorch/transformers_replace/* "${SITE_PKGS}/"
popd >/dev/null

echo "=== [4/5] Building .venv-libero (Python 3.10 + MuJoCo 3.2.3 + robosuite 1.4.1 + LIBERO) ==="
UV_LINK_MODE=copy uv venv --python 3.10 "${PROJECT_ROOT}/.venv-libero"
UV_LINK_MODE=copy uv pip install --python "${PROJECT_ROOT}/.venv-libero/bin/python" "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cpu
UV_LINK_MODE=copy uv pip install --python "${PROJECT_ROOT}/.venv-libero/bin/python" \
  "numpy<2" "scipy" "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" \
  easydict hydra-core cloudpickle einops opencv-python-headless "imageio[ffmpeg]" \
  matplotlib pyyaml tqdm future termcolor websockets msgpack "gym==0.25.2" tyro
UV_LINK_MODE=copy uv pip install --python "${PROJECT_ROOT}/.venv-libero/bin/python" -e "${PROJECT_ROOT}/external/LIBERO" --no-deps
UV_LINK_MODE=copy uv pip install --python "${PROJECT_ROOT}/.venv-libero/bin/python" -e "${PROJECT_ROOT}/external/openpi/packages/openpi-client"

echo "=== [5/5] Configuring LIBERO benchmark paths (.libero/config.yaml) ==="
python3 "${PROJECT_ROOT}/tools/download_libero.py" --suite libero_10

echo ""
echo "[OK] Setup completed! Before running experiments in venv mode, run:"
echo "  source adaptive_action_chunking/server141.env"

