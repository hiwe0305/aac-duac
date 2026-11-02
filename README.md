# Guideline Setup & Chạy Thực Nghiệm O-LONG trên Server GPU (`pi0.5`: `AAC` vs `DUAC`)

Tài liệu này là **Guideline bàn giao (Handover Runbook) trọn gói** để người vận hành (operator / cộng sự) tự setup từ đầu đến cuối và chạy toàn bộ thực nghiệm **O-LONG** ([`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) — *"Adaptive action chunking có còn work trên task dài và khó không?"*) trên **máy chủ GPU từ xa (Remote Headless GPU Server)** thông qua **Docker**.

- **Model duy nhất sử dụng**: **$\pi_{0.5}$ (`pi05`)** (đã dọn sạch toàn bộ các thư mục/file thừa của `GR00T` và `RoboCasa`).
- **Benchmark thực nghiệm**: **LIBERO-Long (`libero_10` clean)** (đã bỏ `LIBERO-PRO` theo phạm vi chốt).
- **Quy ước đường dẫn thống nhất**:
  - Trên máy chủ Host (Server): `<PROJECT_ROOT>` (ví dụ `/data/aac-duac` hoặc `/home/<user>/aac-duac`)
  - Bên trong Docker Container (`aac-duac-pi05`): **`/workspace/aac-duac`**

---

## 1. Bảng Đối Chiếu Trạng Thái [`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) Trong Repo

Toàn bộ các mục kỹ thuật (Step 0.1 $\rightarrow$ Step 0.4, Track A 6 arms, Schema CSV Section 4.2, Thống kê T1–T6 Section 4.3 & Decision Rule Section 5) trong [`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) **đã được cài đặt và kiểm thử sẵn sàng 100%**:

| Hạng mục trong [`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) | Trạng thái | File cấu hình / thực thi trong Repo |
|---|---|---|
| **Step 0.1**: Fix step budget `libero_10` = `520` bước active (`suite_step_budget_name()`) | **Đã hoàn tất** | [`flowbased_aac/eval_libero.py`](file:///home/hiwe/aac-duac/flowbased_aac/flowbased_aac/eval_libero.py), [`adaptive_action_chunking/aac.py`](file:///home/hiwe/aac-duac/adaptive_action_chunking/aac.py) |
| **Step 0.2**: Hook BDDL goal predicates sau mỗi `env.step` $\rightarrow$ ghi `first_true_step` vào `subgoals.csv` | **Đã hoàn tất** | [`eval_libero_goal_predicates()`](file:///home/hiwe/aac-duac/flowbased_aac/flowbased_aac/eval_libero.py) |
| **Step 0.3**: Thêm `--random-horizon-from <decisions.csv>` để chạy đối chứng `Random-h` | **Đã hoàn tất** | [`flowbased_aac/duac.py`](file:///home/hiwe/aac-duac/flowbased_aac/duac.py), [`flowbased_aac/eval_libero.py`](file:///home/hiwe/aac-duac/flowbased_aac/flowbased_aac/eval_libero.py) |
| **Step 0.4**: Chuẩn hoá schema `episodes.csv`, `decisions.csv`, `subgoals.csv`, `state_bank/*.npz` | **Đã hoàn tất** | [`analysis/convert.py`](file:///home/hiwe/aac-duac/analysis/convert.py) |
| **Mục 2.4 & 3 (Track A)**: Đủ 6 arms trên `libero_10` (`duac`, `fixed_10`, `fixed_8`, `fixed_5`, `random_h`, `aac`) | **Đã hoàn tất** | [`manifest.json`](file:///home/hiwe/aac-duac/manifest.json) |
| **Mục 4.3, 5, 6**: Tính bảng **T1–T6**, kiểm định **McNemar + 5,000 Cluster Bootstrap + Holm-Bonferroni**, quy tắc kết luận Section 5 và xuất `outputs/REPORT.md` | **Đã hoàn tất** | [`analysis/analyze_o_long.py`](file:///home/hiwe/aac-duac/analysis/analyze_o_long.py) |

---

## 2. Trong Docker Đã Cài Đặt Sẵn Những Gì?

File [`docker/Dockerfile`](file:///home/hiwe/aac-duac/docker/Dockerfile) và [`docker/docker-compose.yml`](file:///home/hiwe/aac-duac/docker/docker-compose.yml) **đã đóng gói trọn vẹn 100% môi trường thực thi**, người cài trên server **không cần cài Python, CUDA hay bất kỳ thư viện nào trên máy chủ Host**:

1. **Hệ thống Render Headless GPU (`NVIDIA EGL`)**:
   - Đã cài đầy đủ `libegl1`, `libgl1-mesa-glx`, `libosmesa6-dev`, `libglfw3-dev`, `ffmpeg` và cấu hình `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`) để render camera 3D trực tiếp trên GPU server không màn hình.
2. **Hai môi trường ảo Python độc lập (đặt tại `/opt/venv-*` và tự động symlink vào `/workspace/aac-duac/.venv-*` qua [`docker/entrypoint.sh`](file:///home/hiwe/aac-duac/docker/entrypoint.sh) để không bao giờ bị volume mount từ Host ghi đè)**:
   - **`/workspace/aac-duac/.venv-openpi` (`/opt/venv-openpi`, Python 3.11.9)**: Cài đặt chuẩn từ `uv.lock` chính thức của OpenPI (JAX CUDA 12, Flax, Orbax, PyTorch, `openpi`, `openpi-client` và bản vá `transformers_replace`) để chạy server suy luận $\pi_{0.5}$.
   - **`/workspace/aac-duac/.venv-libero` (`/opt/venv-libero`, Python 3.10)**: Cài đặt `mujoco==3.2.3`, `robosuite==1.4.1`, `bddl==1.0.1`, `gym==0.25.2`, `LIBERO`, và `openpi-client` để chạy mô phỏng đánh giá `libero_10`.
3. **Mã nguồn & Dữ liệu mô phỏng Benchmark (`external/`)**:
   - Đã tích hợp sẵn `external/openpi`, `external/LIBERO` (bao gồm toàn bộ `bddl_files`, `init_files` 50 trạng thái khởi tạo/task, và 3D MuJoCo `assets`), và `external/aac-libero`.

---

## 3. Yêu Cầu Hệ Thống & Ánh Xạ Đường Dẫn (`Host` $\leftrightarrow$ `Docker`)

### 3.1. Điều kiện trên Server Host
1. **GPU NVIDIA**: Tối thiểu 1 GPU $\ge 24\text{ GB}$ VRAM (RTX 3090 / 4090 / A5000 / L40S / A100), kiểm tra bằng `nvidia-smi`.
2. **Docker & NVIDIA Container Toolkit**:
   ```bash
   docker --version
   docker compose version
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
3. **Dung lượng ổ cứng trống**: Tối thiểu **`50 GB`** tại nơi đặt `<PROJECT_ROOT>`.

### 3.2. Ánh xạ thư mục (`Host` $\leftrightarrow$ `Container aac-duac-pi05`)

| Đường dẫn trên Server Host | Đường dẫn trong Docker Container | Vai trò |
|---|---|---|
| `<PROJECT_ROOT>` | `/workspace/aac-duac` | Thư mục gốc của project |
| `<PROJECT_ROOT>/data/libero` | `/workspace/aac-duac/data/libero` | Thư mục dữ liệu LIBERO |
| `<PROJECT_ROOT>/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero` | `/workspace/aac-duac/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero` | Checkpoint chính **`pi05_libero`** (~`11.58 GiB`) |
| `<PROJECT_ROOT>/checkpoints/openpi/big_vision/paligemma_tokenizer.model` | `/workspace/aac-duac/checkpoints/openpi/big_vision/paligemma_tokenizer.model` | Tokenizer PaliGemma (`4.07 MB`, đã tải sẵn) |
| `<PROJECT_ROOT>/outputs/pi05/libero_10/` | `/workspace/aac-duac/outputs/pi05/libero_10/` | Kết quả 6 arms Track A (`duac`, `fixed_10`, `fixed_8`, `fixed_5`, `random_h`, `aac`) |
| `<PROJECT_ROOT>/outputs/REPORT.md` | `/workspace/aac-duac/outputs/REPORT.md` | Báo cáo tổng hợp cuối cùng (T1–T6 + Kết luận Mục 5) |

---

## 4. Quy Trình Setup Trên Server Từ Con Số 0 (4 Bước)

### Bước 1: Kiểm tra cấu hình GPU trong `docker/.env`
Từ thư mục gốc `<PROJECT_ROOT>` trên server:
```bash
cat docker/.env
```
Mặc định sử dụng GPU `0` (`GPU_ID=0`, `NVIDIA_VISIBLE_DEVICES=0`). Nếu muốn chạy trên GPU khác (ví dụ GPU `1`), sửa `GPU_ID=1` và `NVIDIA_VISIBLE_DEVICES=1` trong [`docker/.env`](file:///home/hiwe/aac-duac/docker/.env).

### Bước 2: Build & Khởi chạy Docker Container `aac-duac-pi05`
```bash
docker compose -f docker/docker-compose.yml up -d --build
```
*(Lần đầu build sẽ mất khoảng 10–15 phút để tải CUDA image và cài đặt `.venv-openpi` + `.venv-libero`).*

### Bước 3: Mở phiên `tmux` trên Server & Truy cập vào Container
> **Quan trọng**: Luôn mở `tmux` trên server trước khi tải dữ liệu/checkpoint hoặc chạy thực nghiệm dài để không bị ngắt tiến trình khi rớt SSH.

```bash
tmux new -s o_long
docker exec -it aac-duac-pi05 bash
```
Sau lệnh này, bạn đang ở tại `/workspace/aac-duac` bên trong container.

### Bước 4: Tải Dữ Liệu Benchmark LIBERO, Tải Checkpoint `pi05_libero` & Kiểm Tra Hệ Thống

Bên trong container (`/workspace/aac-duac`), chạy lần lượt 3 bước con sau:

#### 4A. Cấu hình & Kiểm tra Dữ liệu Benchmark LIBERO (`tools/download_libero.py`)
```bash
# 1. Tự động cấu hình .libero/config.yaml và xác minh đủ 10/10 file BDDL,
#    10/10 file .pruned_init (500 trạng thái khởi tạo) & 219 file 3D MuJoCo XML assets của libero_10:
python3 tools/download_libero.py --suite libero_10

# (Tuỳ chọn) Nếu muốn tải thêm cả bộ dataset HDF5 demonstration gốc của libero_10 về data/libero/:
# python3 tools/download_libero.py --suite libero_10 --download-demos
```
Khi màn hình trả về `"benchmark_ready": true`, `"bddl_files_found": "10/10"`, `"init_files_found": "10/10"`, dữ liệu mô phỏng `libero_10` đã sẵn sàng 100%.

#### 4B. Tải Trọng số Model `pi05_libero` (~`11.58 GiB`) & Tokenizer PaliGemma
*(Hiện tại metadata `9/9` file và `paligemma_tokenizer.model` đã có sẵn trong repo; lệnh dưới sẽ tải nốt 7 file trọng số lớn vào `/workspace/aac-duac/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero`, tự động hỗ trợ resume nếu đứt mạng)*:
```bash
# Tải đầy đủ trọng số pi05_libero
python3 tools/download_checkpoints.py --target pi05_libero

# Xác minh trạng thái (phải thấy metadata_ready: true, tokenizer_ready: true, weights_ready: true)
python3 tools/download_checkpoints.py --target pi05_libero --verify
```

#### 4C. Chạy `doctor` và Bộ Test Kiểm tra Pipeline (`Step 0.1` $\rightarrow$ `0.4`)
```bash
PYTHONPATH=src python3 -m aac_duac.cli doctor
PYTHONPATH=src python3 -m aac_duac.cli test
```
Khi thấy `ALL TESTS PASSED` và `ALL STATISTICAL CHECKS PASSED`, toàn bộ môi trường, dữ liệu và checkpoint đã sẵn sàng.

---

## 5. Hướng Dẫn Chạy Thực Nghiệm Chính Track A (`pi0.5` trên `libero_10` — 6 Arms $\times$ 500 Episodes)

Theo đúng thiết kế trong [`docs/task3010.md`](file:///home/hiwe/aac-duac/docs/task3010.md) Mục 2.4 & Mục 3, Track A gồm **6 arms** (mỗi arm chạy 10 tasks $\times$ 50 trials = **500 episodes**, cùng `seed = 7`, budget `520` active steps):

1. **`duac`**: Phương pháp DUAC ($\tau=0.9, K=40, \alpha=3.0$), bật `--state-bank` để lưu quỹ đạo phục vụ phân tích offline.
2. **`fixed_10`**: Baseline mặc định của $\pi_{0.5}$ ($h=10$, thực thi hết cả chunk).
3. **`fixed_8`**: Baseline đối chứng khớp với horizon trung bình của DUAC trên LIBERO-10 ($\bar{h} \approx 8.45$), dùng để kiểm tra xem lợi ích có phải chỉ do replan dày hơn hay không.
4. **`fixed_5`**: Baseline replan dày cố định ($h=5$).
5. **`random_h`**: Baseline rút ngẫu nhiên $h$ từ phân phối thực nghiệm của `duac/decisions.csv` (kiểm tra *"replan đúng lúc"* vs *"replan ngẫu nhiên cùng tần suất"*).
6. **`aac`**: Phương pháp AAC gốc ($N=20$ quỹ đạo mẫu, `gaussian_bernoulli`, $\alpha=3.0$).

---

### Phần 5.1 — Chạy 5 Arms Dùng Server DUAC (`duac`, `fixed_10`, `fixed_8`, `fixed_5`, `random_h`)

Mở **2 cửa sổ terminal bên trong container `aac-duac-pi05`** (ví dụ 2 pane trong `tmux`):

#### Terminal 1 (Bật Server $\pi_{0.5}$ DUAC trên cổng `8000`):
```bash
docker exec -it aac-duac-pi05 bash
cd /workspace/aac-duac

/workspace/aac-duac/.venv-openpi/bin/python flowbased_aac/duac.py serve-pi05 \
  --config pi05_libero \
  --checkpoint-dir /workspace/aac-duac/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 8000 \
  --tau 0.9 \
  --num-probes 40 \
  --alpha 3.0
```
*(Đợi đến khi màn hình hiện `[duac.py serve-pi05] Serving pi05_libero ... on 127.0.0.1:8000` rồi chuyển sang Terminal 2).*

#### Terminal 2 (Chạy Client đánh giá LIBERO):
```bash
docker exec -it aac-duac-pi05 bash
cd /workspace/aac-duac

# (0) SMOKE TEST NHANH (1 trial x 2 tasks ~ 2 phút) để kiểm tra GPU + MuJoCo + WebSocket:
/workspace/aac-duac/.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 1 --task-ids 0,1 --seed 7 \
  --state-bank --output-dir /workspace/aac-duac/outputs/smoke_duac

# (1) Arm 1: DUAC (500 episodes, có --state-bank và --save-videos)
/workspace/aac-duac/.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --alpha 3.0 --tau 0.9 --num-probes 40 \
  --state-bank --save-videos \
  --output-dir /workspace/aac-duac/outputs/pi05/libero_10/duac

# (2) Arm 2: Fixed-10 (500 episodes)
/workspace/aac-duac/.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --fixed-horizon 10 --no-state-bank --save-videos \
  --output-dir /workspace/aac-duac/outputs/pi05/libero_10/fixed_10

# (3) Arm 3: Fixed-8 (500 episodes - đối chứng khớp mean h của DUAC)
/workspace/aac-duac/.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --fixed-horizon 8 --no-state-bank --no-save-videos \
  --output-dir /workspace/aac-duac/outputs/pi05/libero_10/fixed_8

# (4) Arm 4: Fixed-5 (500 episodes)
/workspace/aac-duac/.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --fixed-horizon 5 --no-state-bank --no-save-videos \
  --output-dir /workspace/aac-duac/outputs/pi05/libero_10/fixed_5

# (5) Arm 5: Random-h (500 episodes - bốc ngẫu nhiên h từ phân phối của DUAC)
/workspace/aac-duac/.venv-libero/bin/python flowbased_aac/duac.py eval-libero \
  --suite libero_10 --port 8000 --num-trials 50 --seed 7 \
  --random-horizon-from /workspace/aac-duac/outputs/pi05/libero_10/duac/decisions.csv \
  --no-state-bank --no-save-videos \
  --output-dir /workspace/aac-duac/outputs/pi05/libero_10/random_h
```

---

### Phần 5.2 — Chạy Arm 6: AAC Multi-Sample (`N = 20` trên cổng `8015`)

Sau khi chạy xong 5 arms trên, bấm `Ctrl+C` ở **Terminal 1** để tắt server cổng `8000` (giải phóng VRAM), rồi bật server Multi-Sample cho AAC:

#### Terminal 1 (Bật Server $\pi_{0.5}$ Multi-Sample $N=20$ trên cổng `8015`):
```bash
/workspace/aac-duac/.venv-openpi/bin/python adaptive_action_chunking/aac.py serve-pi05 \
  --config pi05_libero \
  --checkpoint_dir /workspace/aac-duac/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 8015 \
  --n_samples 20 \
  --seed 0
```

#### Terminal 2 (Chạy Client đánh giá AAC):
```bash
# (6) Arm 6: AAC (gaussian_bernoulli, N=20, move_th=3.0, 500 episodes)
/workspace/aac-duac/.venv-libero/bin/python adaptive_action_chunking/aac.py eval-libero \
  --backbone pi05 \
  --suite libero_10 \
  --selector gaussian_bernoulli \
  --trials 50 \
  --seed 7 \
  --move_th 3.0 \
  --port 8015 \
  --official_openpi_protocol \
  --videos \
  --out_dir /workspace/aac-duac/outputs/pi05/libero_10/aac
```

---

## 6. Tổng Hợp Thống Kê & Xuất Báo Cáo (`outputs/REPORT.md`)

Sau khi cả 6 thư mục trong `/workspace/aac-duac/outputs/pi05/libero_10/{duac,fixed_10,fixed_8,fixed_5,random_h,aac}` hoàn tất, chạy lệnh duy nhất sau:

```bash
PYTHONPATH=src python3 -m aac_duac.cli analyze \
  --manifest /workspace/aac-duac/manifest.json \
  --output-dir /workspace/aac-duac/outputs/analysis_report
```

Lệnh này thực hiện tự động:
1. Kiểm tra tính toàn vẹn ([`analysis/convert.py`](file:///home/hiwe/aac-duac/analysis/convert.py)) của toàn bộ 6 arms (`500` episodes/arm, kiểm tra header `episodes.csv`, `decisions.csv`, `subgoals.csv` và chữ ký ZIP `PK` của `state_bank/*.npz`).
2. Tính toán toàn bộ bảng **T1, T2, T3, T4, T5, T6** và kiểm định **Exact McNemar + 5,000-lần Cluster Bootstrap + Holm-Bonferroni** cùng **Quy tắc quyết định Mục 5** (`Adaptive work` / `Không work` / `Chưa kết luận được`).
3. Xuất kết quả ra:
   - `/workspace/aac-duac/outputs/REPORT.md`
   - `/workspace/aac-duac/outputs/analysis_report/o_long_report.md`
   - `/workspace/aac-duac/outputs/analysis_report/o_long_report.json`
