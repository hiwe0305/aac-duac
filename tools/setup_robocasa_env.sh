#!/usr/bin/env bash
# Build .venv-robocasa (Python 3.12): LeRobot 0.6.1 + torch cu128 + RoboCasa365 + robosuite (official GitHub clones).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
UV=${UV:-$HOME/.local/bin/uv}
[ -x .venv-robocasa/bin/python ] || $UV venv .venv-robocasa --python 3.12
PY=.venv-robocasa/bin/python
$UV pip install --python $PY torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python $PY -r tools/requirements-robocasa.txt
$UV pip install --python $PY --no-deps pynput
[ -d external/robosuite ] || git clone https://github.com/ARISE-Initiative/robosuite.git external/robosuite
[ -d external/robocasa365 ] || git clone https://github.com/robocasa/robocasa.git external/robocasa365
$UV pip install --python $PY --no-deps -e external/robosuite -e external/robocasa365
$UV pip install --python $PY --no-deps -e external/openpi/packages/openpi-client
$PY -c "import torch, lerobot, robosuite, robocasa, mujoco; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| lerobot', lerobot.__version__, '| robosuite', robosuite.__version__, '| mujoco', mujoco.__version__)"
