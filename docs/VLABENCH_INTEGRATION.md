# VLABench: tích hợp π0.5 `VLABench/pi05-primitive-10task` với Vanilla, Fixed, AAC, DUAC

Track này thêm VLABench vào khung đánh giá O-LONG mà **không huấn luyện** và **không sửa** code AAC/DUAC. Không file cũ nào bị sửa: mọi thứ là file mới (`git status` chỉ có `??`). Giao thức thí nghiệm ở [VLABENCH_EXPERIMENT_PROTOCOL.md](VLABENCH_EXPERIMENT_PROTOCOL.md). Đề bài gốc ở [vlabench.md](vlabench.md).

## 1. Kết luận kiểm tra checkpoint

| Hạng mục | Giá trị đã kiểm | Nguồn |
|---|---|---|
| Định dạng | OpenPI **JAX/Orbax** (`params/` 12 GB + `assets/.../norm_stats.json`), không phải LeRobot PyTorch. `train_state/` (28 GB, optimizer) không cần cho suy luận nên đã xoá sau khi tải | HF `VLABench/pi05-primitive-10task@f658a8b` |
| Code chính thức | fork `Shiduo-zh/openpi` nhánh `pi05` (`external/openpi-vlabench`, commit `788ef1f`), config `pi05_ft_vlabench_primitive` | model card |
| Model | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`, Gemma 2B, 10 bước Euler, bf16 | config của fork |
| So với OpenPI upstream (`external/openpi`) | `Pi0.sample_actions`, `ModelTransformFactory`, `DataConfigFactory`, tokenizer, `Policy` **giống hệt**. Fork chỉ thêm stop-gradient lúc train và vài trường dataset. Vì vậy dùng lại `.venv-openpi` của track LIBERO | diff ở mục 6 |
| **Prediction horizon H** | **10** | |
| **Execution horizon chính thức** | **5** (`replan_steps=5` trong `examples/vlabench/eval.py`) = arm `vanilla` | |
| Camera | `rgb` = [right, left, forward, wrist], 480×480. Policy: image = forward, second_image = right, wrist_image = wrist; resize-pad về 224 trong model transform | client chính thức |
| State (7) | `[ee_pos − (0, −0.4, 0.78), euler_xyz(ee_quat), gripper_open]`, pad tới 32 | client chính thức |
| Action (7) | **Tuyệt đối**: target pos (khung robot), target euler xyz, gripper. Model dự đoán *delta so với state* (`DeltaActions` trên 6 chiều) và server cộng state lại (`AbsoluteActions`). Gripper ≥ 0.1 → mở (ngón 0.04), ngược lại đóng | config + client |
| Chuẩn hoá | Quantile (q01/q99) như mọi π0.5 OpenPI. Stats lấy từ checkpoint. Dữ liệu huấn luyện **không** được tải | `check_policy.py` mục 3 |
| Điều khiển | 1 action = 1 `env.step` (10 Hz): IK → 7 khớp + 2 ngón | evaluator chính thức |
| Độ dài episode | `add_condiment` 300, `select_poker` / `select_painting` 100, các task khác 200 | `task_config.json` |
| Thành công | `timestep.last()` (điều kiện task thoả). Progress / intention score dùng hàm chính thức (ngưỡng 0.1) | |

## 2. Track và task (đã đối chiếu với `VLABench/configs/evaluation/tracks`)

| Track | Tên chính thức | Task | Episode |
|---|---|---|---|
| T1 | `track_1_in_distribution` | 10 primitive | 50/task |
| T2 | `track_2_cross_category` | 10 primitive | 50 (riêng `insert_flower` chỉ có **10**) |
| T3 | `track_3_common_sense` | `select_poker` → `select_nth_largest_poker`, `select_mahjong` → `select_unique_type_mahjong` (tự đổi tên) | 50 |
| T4 | `track_4_semantic_instruction` | 10 primitive | 50 |
| T6 | `track_6_unseen_texture` | 10 primitive | 50 |

T6 ở đây là *unseen texture* của repo VLABench, **không phải** T6 long-horizon trong bài báo. Không có T5 (cross-task để mở). Các task primitive không kiểm tra lập kế hoạch dài hạn. Phạm vi so sánh chỉ là độ bất định ở mức action và việc thực thi thích nghi.

## 3. Kiến trúc

```
.venv-vlabench (Py 3.10, mujoco 3.2.2, dm_control 1.0.22)        .venv-openpi (Py 3.11, JAX) — dùng chung với LIBERO
  eval_vlabench.py ── websocket msgpack (OpenPI) 127.0.0.1 ──>  vlabench/serve.py --mode vanilla|duac|aac
   load_env(task, episode_config)                                 openpi_config.get_train_config()  (tái lập config fork)
   render RGB → common.policy_request                             vanilla: openpi Policy.infer
   arm: duac/fixed/random_h  → duac_policy.select_duac_horizon    duac:    duac_policy.DuacPolicy            (KHÔNG đổi)
        aac                  → aac_official.aac_decision          aac:     serve_pi05_multisample.MultiSamplePi05Policy (KHÔNG đổi)
   chunk[:h] → IK → env.step
```
Phải tách hai process vì hai env xung đột phiên bản (Python, numpy, mujoco). Pipeline chính thức của VLABench và các track LIBERO/RoboCasa cũng tách như vậy. Cổng 8700/8710/8720 (8100–8231 đã bị dịch vụ khác dùng).

| File mới | Vai trò |
|---|---|
| `flowbased_aac/flowbased_aac/vlabench/common.py` | Task/track, giá trị tham chiếu, quy ước state/action, `to_aac_frame` |
| `.../vlabench/openpi_config.py` | TrainConfig `pi05_ft_vlabench_primitive` + `VLABenchInputs/Outputs` (chép từ fork) |
| `.../vlabench/serve.py` | Server 3 mode, thêm `server_ms` và `peak_gpu_mb` vào phản hồi |
| `.../vlabench/eval_vlabench.py` | Evaluator: arm, CSV/JSONL/state bank/video, resume, ghi lỗi |
| `.../vlabench/check_policy.py` | Kiểm checkpoint, chuẩn hoá, tương đương sampler; benchmark latency |
| `.../vlabench/selftest.py` | Kiểm không cần GPU/simulator |
| `analysis/vlabench_summary.py` | Bảng + thống kê + hình V1–V7 (PNG/PDF) |
| `tools/run_vlabench_jobs.sh`, `tools/make_vlabench_jobs.sh`, `jobs/vlabench_*.txt` | Runner và danh sách job |

## 4. AAC và DUAC trên VLABench: những gì giữ nguyên, những gì phải chuyển đổi

**Giữ nguyên.**
- DUAC: phương trình 6 (Hutchinson, τ = 0.9, K = 40, D = 7 chiều action thật) và phương trình 8, đúng sampler JAX `make_duac_sampler`.
- AAC: N = 20 sample chung một prefix, selector chính thức `select_chunk_size(gaussian_bernoulli)`, chạy sample 0, α = 3.
- Entropy của AAC và U của DUAC tính trong không gian action chuẩn hoá của model. Ở đây đó là delta so với state, chuẩn hoá quantile.

**Chuyển đổi bắt buộc (chỉ ở phía adapter).** Cận dưới chuyển động ξ của AAC (`action_magnitude` / `find_first_idx`) đọc action *chưa chuẩn hoá* theo dạng LIBERO, tức delta từng bước theo đơn vị OSC của robosuite (1.0 = 0.05 m hoặc 0.5 rad). VLABench lại trả về pose tuyệt đối tính bằng mét/rad. Vì vậy `common.to_aac_frame` chuyển chunk sang:
- `Δpos_k = p_k − p_{k−1}` và `Δrot_k = rotvec(R_k R_{k−1}ᵀ)` (khung thế giới, giống LIBERO), với `p_{−1}`, `R_{−1}` là pose hiện tại. Khi đó `compose_rotations` chính thức của bước 1..h cho đúng phép quay từ pose hiện tại tới target h; selftest đã kiểm, kể cả khi euler quấn qua ±π.
- Chia cho 0.05 m và 0.5 rad, để α = 3 vẫn mang nghĩa vật lý như trên LIBERO (≈15 cm chuyển động quy đổi).
- Gripper: mở (≥ 0.1) → −1, đóng → +1 (quy ước LIBERO; entropy Bernoulli và phát hiện toggle đối xứng nên không phụ thuộc dấu).

Có thể đặt `--xi-units raw` để không đổi đơn vị. Khi đó ξ gần như luôn bằng H.

**Quan sát (smoke, chưa phải kết quả).** Một chunk 10 bước của VLABench chỉ dịch chuyển khoảng 2.5 đơn vị LIBERO (trung vị), nên ξ = 10 ở khoảng 75% decision. h của DUAC/AAC vì vậy thường bằng 10, dù h_unc tự nó dao động 2–9. Tình huống tương tự đã gặp ở LIBERO/RoboCasa (ξ chiếm ưu thế). Không chỉnh α trên episode đánh giá. Ablation không dùng ξ là việc nên làm tiếp (mục 8).

**RNG.** Mỗi server có một khoá JAX stateful (seed 0). Vanilla gọi `Policy.infer` (1 noise). DUAC tách khoá thành noise + probe. AAC dùng N noise. `check_policy.py` xác nhận: cùng noise thì chunk DUAC và sample 0 của AAC trùng chunk vanilla trong sai số bf16 (≤ 1.2% nửa khoảng chuẩn hoá, trong khi hai noise khác nhau lệch 3–210%). Nghĩa là sampler không đổi policy. Khi nhiều client dùng chung một server, chuỗi noise xen kẽ nhau nên không tái lập bit-exact; trạng thái đầu thì giống hệt giữa mọi arm.

**Gradient.** DUAC lấy JVP bằng `jax.linearize` ngay trên tham số đóng băng (JAX không có chế độ "no-grad" chặn đạo hàm theo đầu vào). Precision và JIT giống vanilla.

## 5. Cài đặt (server này, chỉ trong thư mục hieudd)

```bash
cd /mnt/ssd-storage/hieudd/aac-duac
# 5.1 Mã nguồn chính thức (external/ không commit)
git clone --depth 1 https://github.com/OpenMOSS/VLABench.git external/VLABench                 # cf588fe
git clone --depth 1 -b pi05 https://github.com/Shiduo-zh/openpi.git external/openpi-vlabench    # 788ef1f, chỉ để tham chiếu
git clone https://github.com/motion-planning/rrt-algorithms.git external/rrt-algorithms && git -C external/rrt-algorithms checkout 14e4796
# 5.2 .venv-vlabench (Python 3.10)
export UV_CACHE_DIR=/mnt/ceph-cold-storage/hieudd/.uv-cache
~/.local/bin/uv venv .venv-vlabench --python 3.10
~/.local/bin/uv pip install --python .venv-vlabench/bin/python "mujoco==3.2.2" "dm_control==1.0.22" "numpy==1.25.0" "scipy==1.14.0" \
  "h5py==3.11.0" "gym==0.26.2" "gymnasium==0.29.1" "mediapy==1.2.0" "open3d==0.18.0" opencv-python-headless colorlog colorama \
  pyyaml tqdm imageio imageio-ffmpeg gdown msgpack websockets matplotlib scikit-learn networkx pillow tyro rtree plotly openai python-xlib
~/.local/bin/uv pip install --python .venv-vlabench/bin/python --no-deps pynput -e external/VLABench -e external/rrt-algorithms \
  -e external/openpi/packages/openpi-client      # rrt-algorithms phải editable (bản thường thiếu gói con rrt)
# 5.3 Asset mô phỏng (6 GB, Google Drive) -> ceph-cold, symlink vào VLABench
A=/mnt/ceph-cold-storage/hieudd/vlabench; mkdir -p $A/assets_download $A/assets
.venv-vlabench/bin/gdown "https://drive.google.com/uc?id=1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-" -O $A/assets_download/obj.zip
.venv-vlabench/bin/gdown "https://drive.google.com/uc?id=1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw" -O $A/assets_download/scene.zip
unzip -q $A/assets_download/obj.zip -d $A/assets && unzip -q $A/assets_download/scene.zip -d $A/assets
ln -sfn $A/assets/obj external/VLABench/VLABench/assets/obj; ln -sfn $A/assets/scenes external/VLABench/VLABench/assets/scenes
# 5.4 Checkpoint (chỉ params + assets: 12 GB) -> ceph-cold
C=/mnt/ceph-cold-storage/hieudd/checkpoints/vlabench/pi05-primitive-10task
.venv-openpi/bin/huggingface-cli download VLABench/pi05-primitive-10task --revision f658a8bec801aab960fe0ceab370a123b31c1837 \
  --local-dir $C --include "params/*" "assets/*" "_CHECKPOINT_METADATA" "README.md"
mkdir -p checkpoints/vlabench && ln -sfn $C checkpoints/vlabench/pi05-primitive-10task
```
`.venv-openpi` dùng nguyên bản của track LIBERO (mục 2.2 README).

## 6. Kiểm tra (đã chạy)

```bash
.venv-vlabench/bin/python flowbased_aac/flowbased_aac/vlabench/selftest.py            # CPU
source adaptive_action_chunking/server141.env; export OPENPI_DATA_HOME=$PWD/checkpoints/openpi
.venv-openpi/bin/python flowbased_aac/flowbased_aac/vlabench/check_policy.py --reps 20  # GPU, ~3 phút
```
**`selftest.py` (2026-11-18): ALL PASSED.**
- Quy ước euler/quat trùng hàm của VLABench.
- `to_aac_frame` khôi phục đúng delta, kể cả khi euler quấn qua ±π.
- Trên chunk đã chuyển đổi, `select_chunk_size` chính thức cho kết quả bằng `aac_decision` (50 lần rút).
- DUAC chọn h = max(eq.8, ξ).
- Vòng lặp evaluator thật trên env/server giả chạy được mọi arm, kèm state bank và resume không trùng dòng. Tham số resume khác và h > H đều bị chặn.

**`check_policy.py` (RTX 5090, JAX 0.5.3) → `outputs/vlabench/validation/check_policy.json`: PASS.**

| Kiểm | Kết quả |
|---|---|
| Nạp checkpoint | 12.8 s; H = 10, action_dim 32, max_token_len 200 |
| Output | (10, 7) tuyệt đối; target đầu cách pose hiện tại 1.1–1.2 cm |
| Un-normalize thủ công (quantile + state) so với server | sai khác tối đa 3e-17 (không bị chuẩn hoá hai lần) |
| Cùng noise: DUAC so với vanilla | ≤ 0.59% nửa khoảng chuẩn hoá |
| Cùng noise: AAC sample 0 so với vanilla | ≤ 1.17% (bf16, graph batch 20 khác graph batch 1) |
| Noise khác: vanilla so với vanilla (để đối chiếu) | 1.1–210% |
| Latency ổn định (trung vị 20 lần, sau warm-up, `block_until_ready`) | vanilla 64.9 ms/gọi (model 49.1); **DUAC 71.5 ms (+6.6 ms, +10%)**; **AAC N=20 86.0 ms (+21.1 ms, +32%)** |
| Peak bộ nhớ GPU (một process giữ cả 3 sampler) | 6.98 GB |

**Smoke giai đoạn A** (`select_fruit`, 2 episode): vanilla, fixed_5, duac, aac chạy hết, không lỗi.

**Kiểm vanilla so với giá trị chính thức** (T1, 10 task × episode 0–19, h = 5; `outputs/vlabench/validate`):

| | Ở đây | Model card (50 ep/task) |
|---|---|---|
| Trung bình T1 | **71/200 = 35.5%** (Wilson 29.2–42.3%); bỏ 3 episode lỗi vật lý như evaluator chính thức: 36.0% | 40.6% |
| Kiểm định nhị thức với 0.406 | p = 0.15 (chưa tính nhiễu của chính giá trị tham chiếu) | |
| Task lệch nhiều nhất | `select_drink` 4/20 so với 0.44: p = 0.04, sau Holm (10 task) 0.40 | |

Kết luận: pipeline tái lập đánh giá chính thức trong phạm vi sai số thống kê.

Lỗi gặp khi kiểm:
- 6 lần `Offscreen framebuffer is not complete` ở bước 0, do 10 client tạo context EGL cùng lúc. Đã sửa bằng `open_env` (thử lại trước tương tác đầu tiên) và runner khởi động client lệch nhau 3 s. Cả 6 episode đã chạy lại.
- 3 lần `PhysicsError mjWARN_BADQACC` giữa episode. Đây là bất ổn thật của mô phỏng, được ghi lại và tính là thất bại.

## 7. Lệnh chạy

```bash
cd /mnt/ssd-storage/hieudd/aac-duac
# Server (runner tự bật nếu cổng chưa nghe): vanilla 8700, duac 8710 (cả Fixed-k, Random-h), aac 8720; mỗi server ~7 GB VRAM
source adaptive_action_chunking/server141.env; export OPENPI_DATA_HOME=$PWD/checkpoints/openpi
.venv-openpi/bin/python flowbased_aac/flowbased_aac/vlabench/serve.py --mode duac --port 8710
# Một run rời
MUJOCO_GL=egl .venv-vlabench/bin/python flowbased_aac/flowbased_aac/vlabench/eval_vlabench.py \
  --track T1 --tasks select_fruit --episodes 2 --port 8710 --state-bank --save-videos --output-dir outputs/vlabench/x/duac
#   --vanilla (cổng 8700) | --fixed-horizon k (k ≤ 10) | --aac-selector gaussian_bernoulli (cổng 8720)
#   --random-horizon-from <duac>/decisions.csv | --tasks primitive_10 | --episode-ids 0-9 | --resume-dir <run>
# Theo giai đoạn
bash tools/make_vlabench_jobs.sh pilot            # và: main | robust T2|T3|T4|T6
setsid nohup bash tools/run_vlabench_jobs.sh jobs/vlabench_pilot.txt outputs/vlabench/pilot/logs 10 > outputs/vlabench/pilot/runner.txt 2>&1 &
.venv-vlabench/bin/python analysis/vlabench_summary.py outputs/vlabench/pilot --baseline vanilla
```
Output mỗi run: `args.json`, `run_info.json` (metadata server, phiên bản phần mềm, commit), `episodes.csv` (success, progress, intention, bước, thống kê h, tổng thời gian theo thành phần, peak GPU, lỗi), `decisions.csv` (H, h chạy, h đề xuất, h_unc, ξ, U, thời gian model/server/round-trip/render/chọn h/env, số sample), `algo_log.jsonl` (toàn bộ profile AAC/DUAC, chunk đã chạy), `state_bank/` và `videos/` (tuỳ chọn). Chạy lại cùng lệnh là resume.

## 8. Hạn chế và việc còn mở
- **Đo hiệu năng:** `infer_ms`/round-trip trong lúc đánh giá có tính cả thời gian chờ khi nhiều client dùng chung server. Overhead sạch lấy từ `check_policy.py` (một process, sau warm-up). Không coi việc gọi model ít hơn là bằng chứng tăng tốc. So sánh nên dùng `policy ms / action`.
- **ξ thường bằng H** (mục 4). Cần ablation không dùng ξ để thấy riêng phần đóng góp của độ bất định.
- **Entropy của AAC trên chiều roll:** state roll quanh ±π, và `DeltaActions` của fork không quấn góc, nên delta chuẩn hoá của roll có thể nhảy 2π giữa các sample. Đây là tính chất của checkpoint, không sửa; `H_rot` được log riêng.
- Lỗi mô phỏng: evaluator chính thức bỏ episode lỗi ra khỏi trung bình. Ở đây episode lỗi được ghi lại và tính là thất bại; `summary.json` báo cả hai tỉ lệ.
- Render chỉ RGB ở các điểm quyết định, không ảnh hưởng vật lý. Video ghép camera forward và wrist, không phải lưới 4 camera như bản chính thức.
- Docker chưa cập nhật cho VLABench.
