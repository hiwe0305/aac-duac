#!/usr/bin/env bash
# Kiểm tra toàn bộ môi trường chạy O-LONG (Option B: venv trên server).
# Không dừng ở lỗi đầu tiên: in PASS/WARN/FAIL cho từng mục kèm gợi ý sửa, cuối cùng tổng kết.
#
# Dùng:  bash tools/check_env.sh            # kiểm tra tất cả
#        bash tools/check_env.sh openpi     # chỉ phần .venv-openpi
#        bash tools/check_env.sh libero     # chỉ phần .venv-libero

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONLY="${1:-all}"
N_PASS=0; N_WARN=0; N_FAIL=0

if [ -t 1 ]; then G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; Z=$'\e[0m'; else G=; Y=; R=; B=; Z=; fi
pass() { echo "  ${G}PASS${Z} $1"; N_PASS=$((N_PASS+1)); }
warn() { echo "  ${Y}WARN${Z} $1"; [ -n "$2" ] && echo "       -> $2"; N_WARN=$((N_WARN+1)); }
fail() { echo "  ${R}FAIL${Z} $1"; [ -n "$2" ] && echo "       -> $2"; N_FAIL=$((N_FAIL+1)); }
section() { echo; echo "${B}== $1 ==${Z}"; }

# check_file <path> <mô tả> <gợi ý sửa>
check_file() { if [ -e "$ROOT/$1" ]; then pass "$2 ($1)"; else fail "$2 thiếu: $1" "$3"; fi; }

# check_venv <tên venv> : kiểm tra venv tồn tại, đúng Python 3.11, pip và python cùng interpreter
check_venv() {
  local venv="$ROOT/$1" py="$ROOT/$1/bin/python"
  if [ ! -x "$py" ]; then
    fail "$1 chưa tồn tại" "/usr/bin/python3.11 -m venv $1"
    return 1
  fi
  local pyver pipver
  pyver="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  if [ "$pyver" = "3.11" ]; then pass "$1/bin/python là Python $pyver ($(readlink -f "$py"))"
  else fail "$1/bin/python là Python '$pyver' (cần 3.11) -> $(readlink -f "$py")" "rm -rf $1 && /usr/bin/python3.11 -m venv $1"; fi

  pipver="$("$py" -m pip -V 2>/dev/null | sed -n 's/.*(python \([0-9.]*\)).*/\1/p')"
  if [ -z "$pipver" ]; then fail "$1: pip không chạy được" "$1/bin/python -m ensurepip --upgrade"
  elif [ "$pipver" != "$pyver" ]; then fail "$1: pip dùng Python $pipver nhưng python là $pyver (venv bị lệch)" "rm -rf $1 rồi tạo lại bằng /usr/bin/python3.11"
  else pass "$1: pip và python cùng Python $pipver"; fi

  local extra
  extra="$(ls "$venv/lib" 2>/dev/null | grep -v "^python$pyver$" | tr '\n' ' ')"
  [ -n "$extra" ] && warn "$1/lib có thêm thư mục lạ: $extra" "Venv từng được tạo bằng Python khác; nên xoá và tạo lại"
  return 0
}

# check_imports <python> <module[:tên hiển thị]>... : import từng module riêng, in version
check_imports() {
  local py="$1"; shift
  local mod out err
  err="$(mktemp)"
  for mod in "$@"; do
    # stdout chỉ chứa version; cảnh báo (robosuite, gym, libEGL...) đi vào stderr và bị bỏ qua khi PASS
    out="$("$py" -c "
import importlib
m = importlib.import_module('$mod')
print(getattr(m, '__version__', '') or 'ok')
" 2>"$err")"
    if [ $? -eq 0 ]; then pass "import $mod  [$(echo "$out" | tail -1)]"
    else fail "import $mod  -> $(grep -E 'Error|error' "$err" | tail -1)"; fi
  done
  rm -f "$err"
}

echo "${B}O-LONG environment check${Z}  (ROOT=$ROOT)"

# ---------------------------------------------------------------- Hệ thống
section "Hệ thống"
if [ -x /usr/bin/python3.11 ]; then pass "/usr/bin/python3.11 có sẵn ($(/usr/bin/python3.11 -V 2>&1))"
else fail "Không có /usr/bin/python3.11" "Cài Python 3.11 (conda/miniforge nếu không có sudo)"; fi
echo "       (python3 mặc định: $(python3 -V 2>&1) -> KHÔNG dùng để tạo venv nếu khác 3.11)"
if [ -f /usr/include/python3.11/Python.h ]; then pass "Có header Python.h (build được package C)"
else warn "Không có /usr/include/python3.11/Python.h" "Package phải build C (vd evdev) sẽ lỗi; dùng --no-deps để bỏ qua"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  pass "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null | head -3 | tr '\n' ';')"
else fail "Không thấy nvidia-smi" "Server cần driver NVIDIA cho JAX/EGL"; fi
if ldconfig -p 2>/dev/null | grep -q 'libEGL.so'; then pass "Có libEGL (render MuJoCo headless)"
else warn "Không tìm thấy libEGL trong ldconfig" "MuJoCo render headless có thể lỗi"; fi

# ---------------------------------------------------------------- Repo phụ thuộc
section "Repo phụ thuộc (external/)"
check_file external/openpi/pyproject.toml "openpi" \
  "git clone --recurse-submodules --depth 1 https://github.com/Physical-Intelligence/openpi.git external/openpi"
check_file external/openpi/packages/openpi-client/pyproject.toml "openpi-client" "Clone lại external/openpi"
check_file external/openpi/src/openpi/models_pytorch/transformers_replace "openpi transformers_replace" "Clone lại external/openpi"
check_file external/LIBERO/setup.py "LIBERO" \
  "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO"
check_file external/LIBERO/libero/__init__.py "LIBERO libero/__init__.py" "touch external/LIBERO/libero/__init__.py"
check_file external/aac-libero/.git "aac-libero" \
  "git clone --depth 1 https://github.com/Adaptive-Action-Chunking/libero.git external/aac-libero"

# ---------------------------------------------------------------- .venv-openpi
if [ "$ONLY" = all ] || [ "$ONLY" = openpi ]; then
  section ".venv-openpi (policy server pi0.5)"
  if check_venv .venv-openpi; then
    PY="$ROOT/.venv-openpi/bin/python"
    check_imports "$PY" numpy jax flax orbax.checkpoint transformers torch sentencepiece websockets openpi openpi_client
    # Bản vá transformers_replace đã được copy chưa?
    REPL="$ROOT/external/openpi/src/openpi/models_pytorch/transformers_replace"
    TDIR="$("$PY" -c 'import transformers, os; print(os.path.dirname(transformers.__file__))' 2>/dev/null)"
    if [ -d "$REPL" ] && [ -n "$TDIR" ]; then
      SAMPLE="$(cd "$REPL" && find . -name '*.py' | head -1)"
      if [ -n "$SAMPLE" ] && cmp -s "$REPL/$SAMPLE" "$TDIR/$SAMPLE"; then pass "Đã áp bản vá transformers_replace"
      else fail "Chưa áp bản vá transformers_replace" "cp -r $REPL/* $TDIR/"; fi
    fi
    DEV="$("$PY" -c 'import jax; print(",".join(d.platform for d in jax.devices()))' 2>&1 | tail -1)"
    case "$DEV" in
      *gpu*|*cuda*) pass "JAX thấy GPU: $DEV" ;;
      *) fail "JAX không thấy GPU: $DEV" "Kiểm tra jax[cuda12]==0.5.3 và driver NVIDIA" ;;
    esac
  fi
fi

# ---------------------------------------------------------------- .venv-libero
if [ "$ONLY" = all ] || [ "$ONLY" = libero ]; then
  section ".venv-libero (mô phỏng LIBERO)"
  if check_venv .venv-libero; then
    PYL="$ROOT/.venv-libero/bin/python"
    check_imports "$PYL" numpy torch mujoco robosuite bddl gym easydict imageio cv2 websockets msgpack libero openpi_client
    if MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PYL" -c \
      'import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); c.free()' >/dev/null 2>&1; then
      pass "MuJoCo render headless bằng EGL OK"
    else
      fail "MuJoCo không tạo được EGL context" "MUJOCO_GL=egl .venv-libero/bin/python -c 'import mujoco; mujoco.GLContext(64,64).make_current()' để xem lỗi"
    fi
  fi
fi

# ---------------------------------------------------------------- Dữ liệu & checkpoint
section "Dữ liệu & checkpoint"
check_file .libero/config.yaml "LIBERO config" "source adaptive_action_chunking/server141.env && .venv-libero/bin/python tools/download_libero.py --suite libero_10"
check_file checkpoints/openpi/openpi-assets/checkpoints/pi05_libero/params "Checkpoint pi05_libero" \
  ".venv-openpi/bin/python tools/download_checkpoints.py --target pi05_libero"
check_file checkpoints/openpi/big_vision/paligemma_tokenizer.model "PaliGemma tokenizer" \
  ".venv-openpi/bin/python tools/download_checkpoints.py --target pi05_libero"

# ---------------------------------------------------------------- Tổng kết
echo
echo "${B}Tổng kết:${Z} ${G}${N_PASS} PASS${Z}, ${Y}${N_WARN} WARN${Z}, ${R}${N_FAIL} FAIL${Z}"
[ "$N_FAIL" -eq 0 ] && echo "${G}Môi trường sẵn sàng.${Z}" || echo "${R}Sửa các mục FAIL (theo gợi ý '->') rồi chạy lại.${Z}"
exit "$N_FAIL"
