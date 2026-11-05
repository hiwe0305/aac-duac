# O-LONG: AAC và DUAC trên π0.5 với task dài

Repo này chạy thực nghiệm **O-LONG** ([docs/task3010.md](docs/task3010.md)): hai phương pháp tự chọn horizon AAC và DUAC có thắng horizon cố định tốt nhất trên task dài và khó không, và nếu thắng thì thắng ở đâu.

| Track | Benchmark | Policy | Trạng thái |
|---|---|---|---|
| A | LIBERO-10 clean + 2 suite nhiễu LIBERO-PRO | π0.5 `pi05_libero` (OpenPI JAX, H = 10) | Đang chạy chính (mục 4) |
| RoboCasa365 | composite (horizon trung vị 2.250 bước) | π0.5 `lerobot/pi05_robocasa` (PyTorch, chunk 50) | AAC/DUAC đã viết và kiểm; chờ GPU để smoke test (mục 7) |
| BEHAVIOR-1K | 50 task việc nhà dài | π0.5 `openpi-comet` (JAX, chunk 32) | Chưa cài (mục 8) |

**Nguồn thuật toán**
- **AAC:** gọi thẳng code chính thức [Adaptive-Action-Chunking/libero](https://github.com/Adaptive-Action-Chunking/libero) (clone ở `external/aac-libero`), qua [aac_official.py](flowbased_aac/flowbased_aac/aac_official.py). Server lấy N = 20 chunk; action chạy là sample 0, đúng như client gốc.
- **DUAC:** không có code công bố. Code ở [duac_policy.py](flowbased_aac/flowbased_aac/duac_policy.py) theo bản tái lập của hactt13 (`/mnt/ssd-storage/hactt13/UQ_action_chunking/AAC_reproduce/flowbased_aac`): phương trình 6 (vết Jacobian Hutchinson tại τ = 0.9, K = 40 probe, 7 chiều action) và phương trình 8 (bước nhảy của trung bình tiền tố), cộng cận dưới chuyển động ξ của AAC (α = 3).
- `duac.py test` so bằng số với cả code AAC chính thức lẫn bản DUAC của hactt13 (mục 3).


**Kết quả để chia sẻ:** báo cáo Track A [docs/REPORT_trackA.md](docs/REPORT_trackA.md), phân tích nguyên nhân [docs/track_a/DIAGNOSIS.md](docs/track_a/DIAGNOSIS.md), dữ liệu nhẹ (CSV theo từng episode, thống kê) trong [results/](results/README.md). `outputs/` (5.4 GB) chỉ có trên server.

---

## 1. Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `flowbased_aac/duac.py` | CLI DUAC: `doctor`, `test`, `serve-pi05`, `eval-libero`, `eval-liberopro` |
| `adaptive_action_chunking/aac.py` | CLI AAC: `serve-pi05` (multi-sample), `eval-liberopro`, `eval-libero`, `test` |
| `flowbased_aac/flowbased_aac/eval_libero.py` | Evaluator chung cho mọi arm (DUAC, Fixed-h, Random-h, AAC) |
| `adaptive_action_chunking/tools/serve_pi05_multisample.py` | Server N sample, prefix ảnh và ngôn ngữ encode 1 lần |
| `tools/run_jobs.sh` | Bật server, chạy danh sách job song song, tự resume |
| `tools/run_track_a_main.sh`, `tools/make_main_jobs.sh` | Lần chạy chính của Track A |
| `jobs/` | Danh sách job đã chạy (calibration, main) |
| `analysis/` | `calibration_summary.py`, `explore_runs.py`, `analyze_o_long.py` (T1–T6), `convert.py` |
| `manifest.json` | Manifest của Track A (suite, arm, protocol, tiêu chí mục 5, các deviation, sha256 code) |
| `external/` | `openpi`, `LIBERO`, `LIBERO-PRO`, `aac-libero`, `robocasa365`, `robosuite` (không commit) |
| `flowbased_aac/flowbased_aac/robocasa/` | π0.5 RoboCasa: policy AAC/DUAC (PyTorch), server, evaluator, self-test |
| `outputs/` → `/mnt/ceph-hot-storage/hieudd/o_long/aac-duac` | Mọi output (hot storage, không ghi vào cold storage) |
| `checkpoints/` | Symlink sang ceph-cold (chỉ đọc) |

---

## 2. Cài môi trường (server 3, không dùng Docker)

Cần GPU có ít nhất 24 GB (mỗi server π0.5 khoảng 9 GB) và Python 3.11 ở `/usr/bin/python3.11`. Mở `tmux` trước để không mất phiên khi rớt SSH.

### 2.1 Repo phụ thuộc
```bash
mkdir -p external
git clone --recurse-submodules --depth 1 https://github.com/Physical-Intelligence/openpi.git external/openpi
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO && touch external/LIBERO/libero/__init__.py
git clone --depth 1 https://github.com/Adaptive-Action-Chunking/libero.git external/aac-libero
# LIBERO-PRO: bản đã vá của hactt13 (assets là symlink sang ceph-cold). Cần file __init__.py,
# nếu không thì Python sẽ import nhầm external/LIBERO.
cp -a /mnt/ssd-storage/hactt13/UQ_action_chunking/AAC_reproduce/adaptive_action_chunking/third_party/LIBERO-PRO external/LIBERO-PRO
touch external/LIBERO-PRO/libero/__init__.py
```

### 2.2 `.venv-openpi` (policy server)
```bash
/usr/bin/python3.11 -m venv .venv-openpi
.venv-openpi/bin/pip install --upgrade pip setuptools wheel
.venv-openpi/bin/pip install --prefer-binary -r tools/requirements-openpi.txt
.venv-openpi/bin/pip install --no-deps -e external/openpi/packages/openpi-client -e external/openpi
# lerobot (OpenPI import khi nạp policy), đúng commit OpenPI ghim, KHÔNG kéo dependency (pynput -> evdev cần C compiler)
.venv-openpi/bin/pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
.venv-openpi/bin/pip install "numpy==1.26.4"   # pip có thể tự nâng numpy lên 2.x; OpenPI cần < 2
SITE=$(.venv-openpi/bin/python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
cp -r external/openpi/src/openpi/models_pytorch/transformers_replace/* "$SITE/"
```

### 2.3 `.venv-libero` (mô phỏng)
```bash
/usr/bin/python3.11 -m venv .venv-libero
.venv-libero/bin/pip install --upgrade pip "setuptools<70" wheel
grep -v '^robosuite' tools/requirements-libero.txt > /tmp/req-libero.txt
.venv-libero/bin/pip install --prefer-binary -r /tmp/req-libero.txt
.venv-libero/bin/pip install --no-deps robosuite==1.4.1 pynput           # tránh evdev
.venv-libero/bin/pip install --prefer-binary numba pillow six python-xlib "dm-tree>=0.1.8"
.venv-libero/bin/pip install --no-deps -e external/LIBERO -e external/openpi/packages/openpi-client
```

### 2.4 Checkpoint, cấu hình LIBERO và output (không cần tải gì)
```bash
COLD=/mnt/ceph-cold-storage/temporal/hactt13/adaptive_action_chunking/checkpoints
mkdir -p checkpoints/openpi/big_vision checkpoints/lerobot
ln -sfn $COLD/openpi/openpi-assets checkpoints/openpi/openpi-assets
ln -sfn openpi-assets/checkpoints/pi05_libero checkpoints/openpi/pi05_libero
# OpenPI không nhận symlink cho tokenizer: phải là file thật (4 MB)
cp $COLD/openpi/big_vision/paligemma_tokenizer.model checkpoints/openpi/big_vision/

mkdir -p /mnt/ceph-hot-storage/hieudd/o_long/aac-duac && ln -sfn /mnt/ceph-hot-storage/hieudd/o_long/aac-duac outputs

source adaptive_action_chunking/server141.env       # MUJOCO_GL=egl, không chiếm trước GPU, PYTHONPATH
.venv-libero/bin/python tools/download_libero.py --suite libero_10     # ghi .libero/config.yaml
P=$PWD/external/LIBERO-PRO/libero/libero; mkdir -p .libero-pro
printf "benchmark_root: $P\nbddl_files: $P/bddl_files\ninit_states: $P/init_files\ndatasets: $PWD/data/libero\nassets: $P/assets\n" > .libero-pro/config.yaml
```

Docker (`docker/`) có từ bản đầu nhưng **chưa được kiểm lại**: Dockerfile chưa cài `lerobot` và LIBERO-PRO.

---

## 3. Kiểm tra môi trường và code
```bash
source adaptive_action_chunking/server141.env
bash tools/check_env.sh                                  # PASS/WARN/FAIL cho từng mục
.venv-libero/bin/python flowbased_aac/duac.py doctor
.venv-libero/bin/python flowbased_aac/duac.py test       # so với code AAC chính thức + bản DUAC của hactt13
.venv-libero/bin/python adaptive_action_chunking/aac.py test
PYTHONPATH=src:$PYTHONPATH .venv-libero/bin/python -m aac_duac.cli test   # gồm cả self-test của analysis/
```
`duac.py test` kiểm:
- ξ và luật bước nhảy trùng với `action_magnitude`/`find_first_idx`/`select_chunk_size` chính thức (2.000 chunk × 3 giá trị α).
- Phương trình 6 và 8 trùng với bản của hactt13.
- Bộ ước lượng Hutchinson so với vết Jacobian chính xác. Phần này cần JAX, nên chạy thêm bằng `.venv-openpi` nếu muốn kiểm.
- Vòng lặp evaluator thật trên env giả: mọi arm, state bank, resume, và báo lỗi khi thiếu init state.

---

## 4. Track A: LIBERO-10 với `pi05_libero`

### 4.1 Protocol (mọi arm dùng chung)
OffScreenRenderEnv 256×256, env seed 7, init state = trial id, 10 bước chờ rồi tối đa 520 bước policy. Ảnh gửi nguyên bản, server tự resize về 224. Policy seed 0, RNG stateful. α = 3, τ = 0.9, K = 40, N = 20. Mỗi suite × arm chạy 50 trial/task = 500 episode.

### 4.2 Suite: kết quả kiểm dữ liệu và calibration
Tóm tắt calibration: `analysis/calibration_summary.py`. Kết quả nằm ở `outputs/calibration/`.

| Suite | Init/task | Env dựng được | Fixed-10, 10 trial/task | Dùng? |
|---|---|---|---|---|
| `libero_10` (clean) | 50 | ✅ | DUAC 95.0% (Bước 0.5, khớp mức tham chiếu 95.0%; mean h 8.42) | ✅ clean |
| `libero_10_object` | 50 | ✅ | 68.0% [58.3, 76.3] | ✅ |
| `libero_10_env` | 50 | ✅ | 74.0% [64.6, 81.6] | ✅ |
| `libero_10_with_milk` | 10 (task 3, 4: 0) | task 3, 4 lỗi | 82.5% | ❌ |
| `libero_10_with_mug` | 50 | ✅ | 96.0% | ❌ (gần trần) |
| `libero_10_lan` | 50 | ✅ | 97.0% | ❌ |
| `libero_10_swap` | 50 | ✅ | 7.0% | ❌ (gần sàn) |
| `with_red_box`, `with_red_stick`, `with_blue_stick`, `with_diffpos_stick` | 10–50 | ❌ thiếu asset `red_box`, `red_sticker`, `blue_red_sticker` (kể cả trong repo LIBERO-PRO chính thức) | — | ❌ |

Ghi chú về `libero_10_object`: task 0 và 7 đạt 0/10. Các task này gần như không giúp phân biệt các arm.

### 4.3 Chạy
```bash
bash tools/make_main_jobs.sh libero_10_object libero_10_env      # ghi jobs/main_*.txt (đã có sẵn)
CONC=10 bash tools/run_track_a_main.sh > outputs/A_main_runner.txt 2>&1 &
```
- **Phase 1:** 2 server DUAC (cổng 8000, 8001) chạy DUAC và Fixed-5/8/10.
- **Phase 2:** 1 server DUAC chạy Random-h, đồng thời 1 server AAC (cổng 8015) chạy AAC.
- Log nằm ở `outputs/A/logs/`. Chạy lại cùng lệnh thì các run dở dang được resume.
- Mỗi client giới hạn 2 luồng CPU (`CLIENT_THREADS`).

Lệnh rời cho một suite (`SUITE=libero_10_object`, `OUT=outputs/A/$SUITE`):
```bash
.venv-openpi/bin/python flowbased_aac/duac.py serve-pi05 --port 8000 --seed 0
.venv-libero/bin/python flowbased_aac/duac.py eval-liberopro --suite $SUITE --trials 50 --port 8000 --state-bank --save-videos --output-dir $OUT/duac
.venv-libero/bin/python flowbased_aac/duac.py eval-liberopro --suite $SUITE --trials 50 --port 8000 --fixed-horizon 8 --state-bank --output-dir $OUT/fixed_8
.venv-libero/bin/python flowbased_aac/duac.py eval-liberopro --suite $SUITE --trials 50 --port 8000 --random-horizon-from $OUT/duac/decisions.csv --output-dir $OUT/random_h

.venv-openpi/bin/python adaptive_action_chunking/aac.py serve-pi05 --n_samples 20 --port 8015 --seed 0 --no_seed_per_step
.venv-libero/bin/python adaptive_action_chunking/aac.py eval-liberopro --backbone pi05 --suite $SUITE --selector gaussian_bernoulli \
  --trials 50 --seed 7 --move_th 3 --port 8015 --state-bank --videos --out_dir $OUT/aac
```
Thêm `--resume-dir <run>` để chạy tiếp một run bị ngắt. Các tham số phải giống `args.json` cũ.

### 4.4 Output của mỗi run
| File | Nội dung |
|---|---|
| `episodes.csv` | 1 dòng / episode: success, end_step, n_decisions, mean_h, wall_s |
| `decisions.csv` | 1 dòng / decision: h_executed, h_method (h thuật toán đề xuất, kể cả ở arm Fixed), h_unc, move_lower_bound (ξ), infer_ms, gripper_toggle, profile `u_step_1..10`, `u_prefix_*`, `m_step_*` |
| `algo_log.jsonl` | **Toàn bộ output của thuật toán ở mỗi decision.** AAC: `H_pos`, `H_rot`, `H_grip`, `H_total`, trung bình tiền tố và bước nhảy, `h_entropy`, `motion_m`, ξ, h, tỉ lệ gripper đóng trên N sample. DUAC: `U` (phương trình 6), `divergence`, trung bình tiền tố và bước nhảy, `h_unc`, `motion_m`, ξ. Kèm theo các action đã chạy |
| `subgoals.csv` | 1 dòng / goal predicate × episode: `first_true_step` |
| `state_bank/*.npz` | MuJoCo state từng bước, lệnh gripper, warm start, action, chunk. AAC có thêm `aac_samples` (N × H × 7); DUAC có thêm `duac_uncertainty` và `duac_divergence` |
| `videos/` | Trial 0–1 của mỗi task |

---

## 5. Phân tích
```bash
.venv-libero/bin/python analysis/calibration_summary.py outputs/calibration
.venv-libero/bin/python analysis/explore_runs.py outputs/A                 # h theo giai đoạn, quanh gripper, AUROC của U
PYTHONPATH=src:$PYTHONPATH .venv-libero/bin/python -m aac_duac.cli analyze --manifest manifest.json --output-dir outputs/analysis_report
```
**Việc còn phải làm:** `analyze_o_long.py` hiện chỉ phân tích từng suite riêng. Tiêu chí mục 5 lại cần gộp 2 suite nhiễu, bootstrap theo cụm task × trial. Phần này phải mở rộng trước Bước 4.

## 6. Quan sát trên calibration (chưa phải so sánh giữa các arm)
- h do DUAC đề xuất gần như không đổi theo giai đoạn (trước, giữa, sau các bước con: khoảng 8.4–8.8).
- Quanh lúc đóng/mở gripper, h lại **dài hơn** (khoảng 9.0–9.4), vì ξ chiếm ưu thế khi tay robot chậm lại.
- U trung bình của cả episode tách được episode thất bại (AUROC 0.76–0.83). Nhưng chỉ xét 5 decision đầu và so trong cùng một task thì AUROC còn khoảng 0.5. Như vậy U cao là hệ quả của thất bại, không phải tín hiệu báo trước.

---

## 7. RoboCasa365 với `lerobot/pi05_robocasa` (AAC và DUAC)

Toàn bộ pipeline nằm trong repo này: env riêng, server riêng, evaluator riêng. Code của người khác chỉ dùng để tham khảo.

### 7.1 Thành phần
| Thành phần | Vị trí |
|---|---|
| Checkpoint `lerobot/pi05_robocasa` (revision `3206f645`; chunk 50, 3 camera 256×256, state 16, action 12), tải từ Hugging Face, đã kiểm sha256 | `/mnt/ceph-hot-storage/hieudd/checkpoints/lerobot/pi05_robocasa` ← `checkpoints/lerobot/pi05_robocasa` |
| Env `.venv-robocasa` (Python 3.12 qua `uv`): lerobot 0.6.1, torch 2.7.1+cu128, robosuite 1.5.2, robocasa 1.0.1, mujoco 3.3.1 | `tools/setup_robocasa_env.sh`, `tools/requirements-robocasa.txt` |
| RoboCasa365 và robosuite, clone từ GitHub chính thức | `external/robocasa365` (commit `3443cdf`), `external/robosuite` (`5ce6643f`) |
| Asset (textures, fixtures, objects lightwheel / objaverse / aigen) | `/mnt/ceph-hot-storage/hieudd/robocasa365_assets/`, symlink từ `external/robocasa365/robocasa/models/assets/` |
| Policy (AAC N sample + DUAC Hutchinson bằng PyTorch) | [pi05_torch.py](flowbased_aac/flowbased_aac/robocasa/pi05_torch.py) |
| Server (giao thức websocket/msgpack của OpenPI) | [serve.py](flowbased_aac/flowbased_aac/robocasa/serve.py) |
| Evaluator (env wrapper chính thức `lerobot.envs.robocasa.RoboCasaEnv`) | [eval_robocasa.py](flowbased_aac/flowbased_aac/robocasa/eval_robocasa.py) |
| Self-test (không cần GPU) | [selftest.py](flowbased_aac/flowbased_aac/robocasa/selftest.py) |
| Runner | `tools/run_robocasa_jobs.sh`, job ở `jobs/robocasa_*.txt` |

**Lưu ý về dữ liệu.** Bản checkpoint trên ceph-cold (`/mnt/ceph-cold-storage/temporal/hactt13/.../checkpoints/lerobot/pi05_robocasa`) **bị hỏng**. File `policy_preprocessor_step_3_normalizer_processor.safetensors` toàn byte 0, và sha256 của `model.safetensors` không khớp với Hugging Face. Repo này dùng bản tải lại.

### 7.2 Quy ước (theo LeRobot)
- **Action** `[base_motion 0:4, control_mode 4, eef_pos 5:8, eef_rot 8:11, gripper 11]`. **State** `[base_pos 3, base_quat 4, eef_pos_rel 3, eef_quat_rel 4, gripper_qpos 2]`.
- **Tiền xử lý:** chuẩn hoá MEAN_STD, state rời rạc hoá 256 bin vào prompt `Task: …, State: …;\nAction: `. Tokenizer SentencePiece PaliGemma có BOS, pad tới 200 token. Không dùng tokenizer HF `google/paligemma-3b-pt-224` vì repo đó bị gated; hai bên dùng cùng bộ từ vựng.
- **Episode:** độ dài bằng horizon của task trong registry RoboCasa. Seed của trial k là `--seed + k`, giống nhau cho mọi arm. Thành công tính theo `info["is_success"]` của env.
- **AAC:** gọi selector chính thức trên phần cánh tay `[pos, rot, gripper]`. Các chiều base và mode vẫn được thực thi nhưng không vào entropy hay ξ. Action chạy là sample 0.
- **DUAC:** τ = 0.9 (bước Euler thứ 9 trong 10), K = 40 probe Rademacher trên 12 chiều action thật. Vết khối được ước lượng bằng vector-Jacobian product, `E[εᵀ(Jᵀε)]ᵢ = Tr(J_ii)`, với K probe xử lý chung trong một batch. U theo phương trình 6 với D = 12. h = max(h_unc, ξ).

### 7.3 Đã kiểm (2026-11-04)
- `selftest.py`: U khớp giá trị chính xác của field tuyến tính (sai số khoảng 1e-4). Dấu và mask padding đúng; chunk trùng tích phân Euler thường; AAC trả N sample với noise khác nhau. Mọi arm chạy được ở dry-run.
- Model thật (chạy trên CPU): nạp đủ key, DUAC trả `(50, 12)` cùng U và divergence, AAC trả `(N, 50, 12)`.
- Env: `atomic_seen` `OpenDrawer` cùng 4/16 task `composite_seen` (`DeliverStraw`, `KettleBoiling`, `PackIdenticalLunches`, `WashLettuce`) chạy được với asset lightwheel. 12 task còn lại cần bộ objaverse/aigen (dùng `--obj-registries objaverse,lightwheel,aigen`).
- Tốc độ mô phỏng khoảng 8–9 bước/s mỗi client (Mesa CPU, 3 camera). Một episode composite 2.250 bước mất khoảng 4–5 phút.
- **Chưa chạy trên GPU**, vì GPU đang bận với Track A. Server cần khoảng 9.5 GB VRAM, cộng thêm phần backward của DUAC với K = 40.

### 7.4 Nhóm task (RoboCasa365 registry)
| Nhóm | Task | Horizon trung vị / tối đa |
|---|---|---|
| `atomic_seen` | 18 | 600 / 1.050 |
| `composite_seen` | 16 | **2.250 / 4.350** |
| `composite_unseen` | 16 | 2.550 / 4.350 |

### 7.5 Chạy
```bash
bash tools/setup_robocasa_env.sh                                  # tạo .venv-robocasa (chỉ lần đầu)
.venv-robocasa/bin/python flowbased_aac/flowbased_aac/robocasa/selftest.py
# smoke test (Bước 0.4): 4 task composite x 2 trial; DUAC + Fixed-10/25/50, rồi AAC
bash tools/run_robocasa_jobs.sh duac jobs/robocasa_smoke_duac.txt outputs/robocasa/smoke/logs 1 4
bash tools/run_robocasa_jobs.sh aac  jobs/robocasa_smoke_aac.txt  outputs/robocasa/smoke/logs 1 2
```
Lệnh rời:
```bash
.venv-robocasa/bin/python flowbased_aac/flowbased_aac/robocasa/serve.py --mode duac --port 8100      # hoặc --mode aac --n-samples 20 --port 8120
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  .venv-robocasa/bin/python flowbased_aac/flowbased_aac/robocasa/eval_robocasa.py --tasks WashLettuce --trials 20 \
  --port 8100 --state-bank --output-dir outputs/robocasa/WashLettuce/duac            # thêm --fixed-horizon 25, --random-horizon-from ..., --aac-selector gaussian_bernoulli
```
Output: `episodes.csv`, `decisions.csv`, `algo_log.jsonl` (đủ profile 50 bước của AAC/DUAC), `state_bank/*.npz` (MuJoCo state ở mỗi decision, chunk, action đã chạy, 20 sample của AAC hoặc U/divergence của DUAC), và `videos/`.

### 7.6 Việc tiếp theo
1. Smoke test trên GPU khi Track A xong. Kiểm success khác 0, h có thay đổi, và đo `infer_ms` của DUAC và AAC.
2. Calibration Fixed-25 trên `composite_seen`, chọn các task có success 20–70%.
3. Chạy chính: DUAC, AAC, Random-h, Fixed-10/25/50, mỗi task 20 trial (số episode LeRobot khuyến nghị). Ghi vào `manifest.json` thành một track riêng trước khi chạy.
4. Cân nhắc chọn lại α cho chunk 50, rồi ghi lại trong manifest **trước** khi chạy chính.

---

## 8. BEHAVIOR-1K với `openpi-comet` (để chạy sau)

**Chưa có gì trên server.** Đây là benchmark dài nhất: 50 task việc nhà, mỗi task gồm nhiều bước con, robot Galaxea R1 hai tay di động.

| Thành phần | Nguồn |
|---|---|
| Code policy | [mli0603/openpi-comet](https://github.com/mli0603/openpi-comet), fork OpenPI (JAX), hạng 2 BEHAVIOR Challenge 2025 ([arXiv 2512.10071](https://arxiv.org/abs/2512.10071)) |
| Checkpoint π0.5 | `pi05-b1kpt50-cs32` (50 task, chunk 32) và `pi05-b1kpt12-cs32` (12 task), theo model zoo trong README của openpi-comet; weight trên Hugging Face [sunshk/openpi_comet](https://huggingface.co/sunshk/openpi_comet) (bản PyTorch: `sunshk/openpi_comet_pytorch`) |
| Checkpoint khác | [mafangniu/Behavior-Skill-VLA-Checkpoints](https://huggingface.co/mafangniu/Behavior-Skill-VLA-Checkpoints) (50 và 12 task, task-level hoặc skill-level) |
| Simulator | BEHAVIOR-1K / OmniGibson trên NVIDIA Isaac Sim, cùng bộ asset BEHAVIOR (dung lượng lớn) |

**Các bước dự kiến** (cần kiểm lại phiên bản trong README của openpi-comet và BEHAVIOR-1K trước khi cài):
1. Kiểm phần cứng và quyền: Isaac Sim cần driver NVIDIA phù hợp và cài được không cần root; kiểm dung lượng ổ (asset BEHAVIOR lớn, SSD còn khoảng 107 GB).
2. Cài BEHAVIOR-1K/OmniGibson vào một env riêng (ví dụ `.venv-behavior`), tải asset về hot storage.
3. Clone openpi-comet vào `external/openpi-comet`, tải `pi05-b1kpt50-cs32`, chạy evaluator gốc trên 1 task × 1 episode.
4. Vì là JAX OpenPI nên dùng lại được `make_duac_sampler`, nhưng chunk là 32 và số chiều action khác LIBERO, nên `valid_action_dim` phải đặt lại. AAC cần server N sample.
5. Chi phí: theo một model card cộng đồng, episode thất bại có thể mất khoảng 30 phút. Phải đo ở smoke test trước khi lên kế hoạch số trial.

Theo mục 2.7 của task, mỗi benchmark thêm vào là một track riêng và phải ghi vào `manifest.json` trước khi chạy chính.
