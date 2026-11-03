# O-LONG Runbook: Đánh Giá `AAC` vs `DUAC` trên $\pi_{0.5}$ (`libero_10`)

Tài liệu hướng dẫn bàn giao (Handover Guideline) để cài đặt và chạy trọn vẹn thực nghiệm **O-LONG** ([`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) — *"Adaptive action chunking có còn work trên task dài và khó không?"*) trên **máy chủ GPU Linux (Remote Headless GPU Server)**.

- **Model**: **$\pi_{0.5}$ (`pi05_libero`)** — OpenPI JAX.
- **Benchmark**: **LIBERO-Long (`libero_10` clean)** — 10 tasks $\times$ 50 trials = **500 episodes / arm**, step budget = **520 active steps**.
- **2 Tuỳ chọn cài đặt môi trường**:
  - **Option A (Khuyên dùng — Docker)**: Đóng gói sẵn toàn bộ CUDA, EGL headless, `.venv-openpi`, `.venv-libero` và các repo phụ thuộc trong container `aac-duac-pi05`.
  - **Option B (Chạy trực tiếp bằng `venv` trên Server)**: Tự động cài đặt 2 môi trường ảo `.venv-openpi` và `.venv-libero` ngay trên máy chủ qua script [`tools/setup_venv.sh`](file:///home/hiwe/aac-duac/tools/setup_venv.sh).

---

## 1. Tóm Tắt Cấu Trúc & Trạng Thái Cài Đặt Theo [`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md)

| Hạng mục trong [`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) | Trạng thái | File thực thi |
|---|---|---|
| **Step 0.1**: Fix step budget `libero_10` = `520` active steps | Sẵn sàng | [`flowbased_aac/eval_libero.py`](file:///home/hiwe/aac-duac/flowbased_aac/flowbased_aac/eval_libero.py), [`adaptive_action_chunking/aac.py`](file:///home/hiwe/aac-duac/adaptive_action_chunking/aac.py) |
| **Step 0.2**: Hook BDDL goal predicates sau mỗi `env.step` $\rightarrow$ `subgoals.csv` | Sẵn sàng | [`eval_libero_goal_predicates()`](file:///home/hiwe/aac-duac/flowbased_aac/flowbased_aac/eval_libero.py) |
| **Step 0.3**: Đối chứng `Random-h` (`--random-horizon-from`) | Sẵn sàng | [`flowbased_aac/duac.py`](file:///home/hiwe/aac-duac/flowbased_aac/duac.py) |
| **Step 0.4**: Chuẩn hoá `episodes.csv`, `decisions.csv`, `subgoals.csv`, `state_bank/*.npz` | Sẵn sàng | [`analysis/convert.py`](file:///home/hiwe/aac-duac/analysis/convert.py) |
| **Track A (6 Arms)**: `duac`, `fixed_10`, `fixed_8`, `fixed_5`, `random_h`, `aac` | Sẵn sàng | [`manifest.json`](file:///home/hiwe/aac-duac/manifest.json) |
| **Thống kê T1–T6 & Mục 5**: McNemar + 5,000 Cluster Bootstrap + Holm $\rightarrow$ `outputs/REPORT.md` | Sẵn sàng | [`analysis/analyze_o_long.py`](file:///home/hiwe/aac-duac/analysis/analyze_o_long.py) |

### Yêu cầu phần cứng trên Server
- **GPU**: 1 GPU NVIDIA $\ge 24\text{ GB}$ VRAM (RTX 3090 / 4090 / A5000 / L40S / A100), kiểm tra bằng `nvidia-smi`.
- **Ổ cứng trống**: Tối thiểu **`50 GB`** (chứa môi trường, checkpoint `pi05_libero` ~`11.6 GiB`, và `outputs/`).

---

## 2. Bước 1: Clone Repo & Chọn Môi Trường Cài Đặt (Option A hoặc Option B)

Trên máy chủ GPU, mở `tmux` để giữ phiên làm việc không bị ngắt khi rớt SSH:
```bash
git clone https://github.com/hiwe0305/aac-duac.git
cd aac-duac
tmux new -s o_long
```

Chọn **1 trong 2** cách dưới đây:

---

### Option A (Khuyên dùng): Chạy bằng Docker (`docker/docker-compose.yml`)

Bên trong Docker image ([`docker/Dockerfile`](file:///home/hiwe/aac-duac/docker/Dockerfile)) đã cài đặt sẵn **NVIDIA EGL Headless**, **`.venv-openpi`** (`/opt/venv-openpi`), **`.venv-libero`** (`/opt/venv-libero`) và clone sẵn `external/{openpi,LIBERO,aac-libero}`.

1. **(Tuỳ chọn) Chọn GPU trong [`docker/.env`](file:///home/hiwe/aac-duac/docker/.env)** (mặc định dùng GPU `0`):
   ```bash
   cat docker/.env
   ```
2. **Build và khởi chạy container nền `aac-duac-pi05`**:
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build
   ```
3. **Truy cập vào bên trong container** (thư mục làm việc tự động là `/workspace/aac-duac`):
   ```bash
   docker exec -it aac-duac-pi05 bash
   ```

---

### Option B: Cài Đặt Trực Tiếp Bằng `python -m venv` + `pip` (Không Cần `uv`, Không Cần `gcc`/`cc`)

Đã chuẩn bị sẵn 2 file requirements chỉ dùng 100% gói cài sẵn (`.whl` trên PyPI, không cần trình biên dịch C `cc` và không cần `uv`):
- [`tools/requirements-openpi.txt`](file:///home/hiwe/aac-duac/tools/requirements-openpi.txt) (cho `.venv-openpi`)
- [`tools/requirements-libero.txt`](file:///home/hiwe/aac-duac/tools/requirements-libero.txt) (cho `.venv-libero`)

Bạn có thể chạy **1 lệnh tự động** `bash tools/setup_venv.sh`, hoặc chạy **từng lệnh rời rạc** dưới đây:

#### Bước B1: Clone 3 repo phụ thuộc (`openpi`, `LIBERO`, `aac-libero`)
```bash
export PROJECT_ROOT="$(pwd)"
mkdir -p "$PROJECT_ROOT/external"
[ -d "$PROJECT_ROOT/external/openpi" ] || git clone --recurse-submodules --depth 1 https://github.com/Physical-Intelligence/openpi.git "$PROJECT_ROOT/external/openpi"
[ -d "$PROJECT_ROOT/external/LIBERO" ] || (git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$PROJECT_ROOT/external/LIBERO" && touch "$PROJECT_ROOT/external/LIBERO/libero/__init__.py")
[ -d "$PROJECT_ROOT/external/aac-libero" ] || git clone --depth 1 https://github.com/Adaptive-Action-Chunking/libero.git "$PROJECT_ROOT/external/aac-libero"
```

#### Bước B2: Tạo `.venv-openpi` bằng `python3 -m venv` + `pip`
```bash
# Tạo môi trường ảo Python >= 3.11 (ví dụ python3 trong conda base hoặc python3.11)
python3 -m venv "$PROJECT_ROOT/.venv-openpi"

# Nâng cấp pip và cài toàn bộ thư viện từ tools/requirements-openpi.txt (100% binary wheels)
"$PROJECT_ROOT/.venv-openpi/bin/pip" install --upgrade pip setuptools wheel
"$PROJECT_ROOT/.venv-openpi/bin/pip" install --only-binary=:all: -r "$PROJECT_ROOT/tools/requirements-openpi.txt"

# Cài đặt openpi-client và openpi ở chế độ editable (--no-deps)
"$PROJECT_ROOT/.venv-openpi/bin/pip" install --no-deps \
  -e "$PROJECT_ROOT/external/openpi/packages/openpi-client" \
  -e "$PROJECT_ROOT/external/openpi"

# Copy bản vá transformers_replace của OpenPI
SITE_PKGS="$("$PROJECT_ROOT/.venv-openpi/bin/python" -c "import transformers, os; print(os.path.dirname(transformers.__file__))")"
cp -r "$PROJECT_ROOT/external/openpi/src/openpi/models_pytorch/transformers_replace/"* "$SITE_PKGS/"
```

#### Bước B3: Tạo `.venv-libero` bằng `python3 -m venv` + `pip`
```bash
# Tạo môi trường ảo cho LIBERO
python3 -m venv "$PROJECT_ROOT/.venv-libero"

# Cài đặt toàn bộ thư viện mô phỏng từ tools/requirements-libero.txt (MuJoCo 3.2.3, robosuite 1.4.1, BDDL...)
"$PROJECT_ROOT/.venv-libero/bin/pip" install --upgrade pip "setuptools<70" wheel
"$PROJECT_ROOT/.venv-libero/bin/pip" install --only-binary=:all: -r "$PROJECT_ROOT/tools/requirements-libero.txt"

# Cài đặt LIBERO và openpi-client ở chế độ editable (--no-deps)
"$PROJECT_ROOT/.venv-libero/bin/pip" install --no-deps \
  -e "$PROJECT_ROOT/external/LIBERO" \
  -e "$PROJECT_ROOT/external/openpi/packages/openpi-client"
```

#### Bước B4: Nạp biến môi trường & Khởi tạo cấu hình LIBERO
```bash
source adaptive_action_chunking/server141.env
"$PROJECT_ROOT/.venv-libero/bin/python" tools/download_libero.py --suite libero_10
```

---

## 3. Bước 2: Tải Dữ Liệu Benchmark, Tải Checkpoint `pi05_libero` & Kiểm Tra (`doctor` + `test`)

> Từ bước này trở đi, **mọi câu lệnh đều giống hệt nhau** cho cả **Option A** (chạy trong container tại `/workspace/aac-duac`) và **Option B** (chạy trên host sau khi `source adaptive_action_chunking/server141.env`).

### 3.1. Chuẩn bị & Xác minh Dữ liệu Benchmark LIBERO (`tools/download_libero.py`)
```bash
# Tự động cấu hình .libero/config.yaml và kiểm tra đủ 10/10 file BDDL,
# 10/10 file .pruned_init (500 trạng thái khởi tạo) & 106 file 3D MuJoCo XML assets của libero_10:
python3 tools/download_libero.py --suite libero_10

# (Tuỳ chọn) Chỉ chạy lệnh dưới nếu muốn tải thêm bộ dataset HDF5 demonstration gốc về data/libero/:
# python3 tools/download_libero.py --suite libero_10 --download-demos
```

### 3.2. Tải Trọng số Model `pi05_libero` (~`11.58 GiB`) & Tokenizer PaliGemma
```bash
# 1. Tải checkpoint pi05_libero + paligemma_tokenizer.model (hỗ trợ tự động resume nếu đứt mạng)
python3 tools/download_checkpoints.py --target pi05_libero

# 2. Kiểm tra trạng thái (đảm bảo metadata_ready: true, tokenizer_ready: true, weights_ready: true)
python3 tools/download_checkpoints.py --target pi05_libero --verify
```

### 3.3. Chạy `doctor` và Bộ Test Kiểm tra Pipeline (`Step 0.1` $\rightarrow$ `0.4`)
```bash
PYTHONPATH=src python3 -m aac_duac.cli doctor
PYTHONPATH=src python3 -m aac_duac.cli test
```
Khi màn hình báo `ALL TESTS PASSED` và `ALL STATISTICAL CHECKS PASSED`, hệ thống đã sẵn sàng chạy thực nghiệm.

---

## 4. Bước 3: Chạy Thực Nghiệm Chính Track A (6 Arms $\times$ 500 Episodes)

Mở **2 cửa sổ terminal** (2 pane trong `tmux`).
- Nếu dùng **Option A (Docker)**: ở mỗi terminal chạy `docker exec -it aac-duac-pi05 bash`.
- Nếu dùng **Option B (`venv`)**: ở mỗi terminal chạy `source adaptive_action_chunking/server141.env`.

---

### Giai đoạn 4.1 — Chạy 5 Arms đầu với Server DUAC (Cổng `8000`)

#### Terminal 1 (Server $\pi_{0.5}$ DUAC — cổng `8000`):
```bash
./.venv-openpi/bin/python flowbased_aac/duac.py serve-pi05 \
  --config pi05_libero \
  --checkpoint-dir checkpoints/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 8000 \
  --tau 0.9 \
  --num-probes 40 \
  --alpha 3.0
```
*(Đợi dòng thông báo `[duac.py serve-pi05] Serving pi05_libero ... on 127.0.0.1:8000` rồi chạy Terminal 2).*

#### Terminal 2 (Client đánh giá LIBERO — chạy lần lượt):
```bash
# [Smoke Test - 2 phút] Chạy thử 1 trial x 2 tasks để xác nhận GPU + MuJoCo + WebSocket hoạt động:
./.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 1 --task-ids 0,1 --seed 7 \
  --state-bank --output-dir outputs/smoke_duac

# [Arm 1/6] DUAC (tau=0.9, K=40, alpha=3.0, 500 episodes, lưu state_bank & video)
./.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --alpha 3.0 --tau 0.9 --num-probes 40 \
  --state-bank --save-videos \
  --output-dir outputs/pi05/libero_10/duac

# [Arm 2/6] Fixed-10 (Baseline mặc định h=10, 500 episodes)
./.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --fixed-horizon 10 --no-state-bank --save-videos \
  --output-dir outputs/pi05/libero_10/fixed_10

# [Arm 3/6] Fixed-8 (Đối chứng khớp horizon trung bình của DUAC, 500 episodes)
./.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --fixed-horizon 8 --no-state-bank --no-save-videos \
  --output-dir outputs/pi05/libero_10/fixed_8

# [Arm 4/6] Fixed-5 (Baseline replan dày h=5, 500 episodes)
./.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --fixed-horizon 5 --no-state-bank --no-save-videos \
  --output-dir outputs/pi05/libero_10/fixed_5

# [Arm 5/6] Random-h (Đối chứng bốc ngẫu nhiên h từ phân phối của DUAC, 500 episodes)
./.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --random-horizon-from outputs/pi05/libero_10/duac/decisions.csv \
  --no-state-bank --no-save-videos \
  --output-dir outputs/pi05/libero_10/random_h
```

---

### Giai đoạn 4.2 — Chạy Arm 6 (`AAC`) với Server Multi-Sample (Cổng `8015`)

Bấm `Ctrl+C` ở **Terminal 1** để tắt server cổng `8000`, sau đó khởi động server Multi-Sample ($N=20$) cho AAC:

#### Terminal 1 (Server $\pi_{0.5}$ Multi-Sample $N=20$ — cổng `8015`):
```bash
./.venv-openpi/bin/python adaptive_action_chunking/aac.py serve-pi05 \
  --config pi05_libero \
  --checkpoint_dir checkpoints/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 8015 \
  --n_samples 20 \
  --seed 0
```

#### Terminal 2 (Client đánh giá AAC):
```bash
# [Arm 6/6] AAC (gaussian_bernoulli, N=20, move_th=3.0, 500 episodes)
./.venv-libero/bin/python adaptive_action_chunking/aac.py eval-libero \
  --backbone pi05 \
  --suite libero_10 \
  --selector gaussian_bernoulli \
  --trials 50 \
  --seed 7 \
  --move_th 3.0 \
  --port 8015 \
  --official_openpi_protocol \
  --videos \
  --out_dir outputs/pi05/libero_10/aac
```

---

## 5. Bước 4: Phân Tích Thống Kê & Xuất Báo Cáo (`outputs/REPORT.md`)

Sau khi hoàn tất cả 6 arms trong `outputs/pi05/libero_10/{duac,fixed_10,fixed_8,fixed_5,random_h,aac}`, chạy lệnh:

```bash
PYTHONPATH=src python3 -m aac_duac.cli analyze \
  --manifest manifest.json \
  --output-dir outputs/analysis_report
```

Kết quả đầu ra tự động gồm:
1. **`outputs/REPORT.md`** & **`outputs/analysis_report/o_long_report.md`**: Báo cáo Markdown hoàn chỉnh gồm kết luận tự động theo quy tắc Mục 5 (`Adaptive work` / `Không work` / `Chưa kết luận được`) và toàn bộ bảng **T1, T2, T3, T4, T5, T6**.
2. **`outputs/analysis_report/o_long_report.json`**: File JSON chứa toàn bộ số liệu thống kê (Wilson CI, Exact McNemar, 5,000-lần Cluster Bootstrap, Holm-Bonferroni, phân rã theo Subgoal 1 & 2, và độ nhạy $\alpha$).
