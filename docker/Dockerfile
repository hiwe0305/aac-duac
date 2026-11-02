# Dockerfile for O-LONG: pi0.5 (OpenPI JAX) with DUAC vs AAC on LIBERO-Long (libero_10)
# Designed for headless NVIDIA GPU servers (Ubuntu 22.04 + CUDA 12.8 + EGL)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    XDG_CACHE_HOME=/workspace/aac-duac/.cache/duac \
    OPENPI_DATA_HOME=/workspace/aac-duac/checkpoints/openpi \
    LIBERO_CONFIG_PATH=/workspace/aac-duac/.libero \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# 1. System packages (OpenGL/EGL for headless MuJoCo, clang/headers for JAX, aria2/curl/tmux for server ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    aria2 \
    tmux \
    ca-certificates \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libegl1 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    pkg-config \
    libhidapi-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configure NVIDIA EGL ICD so headless MuJoCo (`MUJOCO_GL=egl`) works on remote GPU servers without X11
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install uv (astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/aac-duac

# 2. Clone OpenPI, LIBERO, and Adaptive-Action-Chunking LIBERO repositories
#    (LIBERO includes all BDDL files, 50 initial states/task, and 3D MuJoCo assets for libero_10)
RUN mkdir -p external && \
    git clone --recurse-submodules --depth 1 https://github.com/Physical-Intelligence/openpi.git external/openpi && \
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO && \
    touch external/LIBERO/libero/__init__.py && \
    git clone --depth 1 https://github.com/Adaptive-Action-Chunking/libero.git external/aac-libero

# 3. Create /opt/venv-openpi (and symlink /workspace/aac-duac/.venv-openpi) using OpenPI's official uv.lock
#    Stored in /opt so host bind-mounts into /workspace/aac-duac never shadow the environment
RUN cd /workspace/aac-duac/external/openpi && \
    UV_PROJECT_ENVIRONMENT=/opt/venv-openpi uv venv --python 3.11.9 /opt/venv-openpi && \
    UV_PROJECT_ENVIRONMENT=/opt/venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev && \
    UV_PROJECT_ENVIRONMENT=/opt/venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv pip install --python /opt/venv-openpi/bin/python --no-deps -e packages/openpi-client -e . && \
    SITE_PKGS=$(/opt/venv-openpi/bin/python -c "import transformers, os; print(os.path.dirname(transformers.__file__))") && \
    cp -r src/openpi/models_pytorch/transformers_replace/* "$SITE_PKGS/" && \
    ln -s /opt/venv-openpi /workspace/aac-duac/.venv-openpi

# 4. Create /opt/venv-libero (and symlink /workspace/aac-duac/.venv-libero) for LIBERO simulation evaluation
RUN uv venv --python 3.10 /opt/venv-libero && \
    . /opt/venv-libero/bin/activate && \
    uv pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install "numpy<2" "scipy" "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" \
      easydict hydra-core cloudpickle einops opencv-python-headless "imageio[ffmpeg]" \
      matplotlib pyyaml tqdm future termcolor websockets msgpack gym==0.25.2 tyro && \
    uv pip install -e /workspace/aac-duac/external/LIBERO --no-deps && \
    uv pip install -e /workspace/aac-duac/external/openpi/packages/openpi-client && \
    ln -s /opt/venv-libero /workspace/aac-duac/.venv-libero

# 5. Non-interactive LIBERO config & per-container directories
RUN mkdir -p /workspace/aac-duac/.libero /workspace/aac-duac/.cache/duac /workspace/aac-duac/checkpoints /workspace/aac-duac/outputs /workspace/aac-duac/data/libero && \
    L="/workspace/aac-duac/external/LIBERO/libero/libero" && \
    printf "benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: /workspace/aac-duac/data/libero\nassets: %s/assets\n" "$L" "$L" "$L" "$L" > /workspace/aac-duac/.libero/config.yaml

ENV PYTHONPATH=/workspace/aac-duac:/workspace/aac-duac/src:/workspace/aac-duac/flowbased_aac:/workspace/aac-duac/adaptive_action_chunking:/workspace/aac-duac/external/openpi/src:/workspace/aac-duac/external/openpi/packages/openpi-client/src:/workspace/aac-duac/external/LIBERO

# 6. Copy project source files & entrypoint
COPY configs /workspace/aac-duac/configs
COPY src /workspace/aac-duac/src
COPY tools /workspace/aac-duac/tools
COPY eval /workspace/aac-duac/eval
COPY flowbased_aac /workspace/aac-duac/flowbased_aac
COPY adaptive_action_chunking /workspace/aac-duac/adaptive_action_chunking
COPY analysis /workspace/aac-duac/analysis
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY manifest.json README.md /workspace/aac-duac/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
