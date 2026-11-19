# VLABench: giao thức thí nghiệm

Chi tiết kỹ thuật ở [VLABENCH_INTEGRATION.md](VLABENCH_INTEGRATION.md).

## 1. Biến được giữ cố định giữa các arm
| Biến | Giá trị |
|---|---|
| Checkpoint | `VLABench/pi05-primitive-10task@f658a8b`, JAX bf16, 10 bước Euler, H = 10 |
| Episode | Episode `i` của config track chính thức (giống nhau ở mọi arm). `random`/`numpy` seed = `--seed + i` (mặc định `i`) trước khi nạp env |
| Kết thúc | `timestep.last()` hoặc đạt độ dài tối đa chính thức của task (100/200/300) |
| Action / IK / gripper | Như client và evaluator chính thức |
| RNG policy | Một khoá JAX stateful, seed 0, mỗi server một khoá |
| Siêu tham số | DUAC τ = 0.9, K = 40, min h = 2. AAC N = 20, `gaussian_bernoulli`, α = 3 (ξ theo đơn vị LIBERO). Lấy nguyên từ track LIBERO, **không chỉnh** trên VLABench |
| Phần cứng | 1× RTX 5090; mỗi server ~7 GB; client render EGL |

## 2. Arm
| Arm | Server | h |
|---|---|---|
| `vanilla` | vanilla (8700) | 5, cấu hình chính thức |
| `fixed_k`, k ∈ {8, 10} (chạy đầy đủ) | duac (8710) | k (vẫn log đề xuất của DUAC). Không chạy k > 10 vì vượt H = 10 |
| `duac` | duac | max(h_unc, ξ) |
| `random_h` | duac | rút từ phân phối h đã chạy của DUAC trên cùng task (đối chứng: cùng số lần gọi model, không dùng thông tin bất định) |
| `aac` | aac (8720) | selector AAC chính thức |

`fixed_5` (server duac) và `vanilla` có cùng h. Khác biệt duy nhất là cách rút noise (khoá tách cho noise/probe), nên đây cũng là một đối chứng.

## 3. Các giai đoạn
| Giai đoạn | Nội dung | Job | Trạng thái |
|---|---|---|---|
| A. Smoke | `select_fruit` × 2 episode; vanilla, fixed_5, duac, aac | thủ công | ✅ mọi arm chạy hết, không lỗi |
| V. Kiểm vanilla | T1, 10 task × 20 episode, vanilla; so với model card (40.6%) | `jobs/vlabench_validate_vanilla.txt` | ✅ 71/200 = 35.5% (CI 29.2–42.3) so với 40.6%, p = 0.15 |
| B. Pilot | T1, 5 task × 10 episode; vanilla, duac, aac | `jobs/vlabench_pilot.txt` | ✅ 150 episode, 0 lỗi (mục 5) |
| C+D. Chạy đầy đủ | T1/T2/T3/T4/T6 × 10 task × 50 episode; vanilla, fixed_8, fixed_10, duac, aac (12.300 episode) | `tools/run_vlabench_full.sh` | ✅ 2026-11-19, [results/vlabench/full/REPORT.md](../results/vlabench/full/REPORT.md) |

Hiệu chỉnh siêu tham số, nếu sau này cần, chỉ dùng episode ngoài khoảng đánh giá. Episode 0–49 của mọi track dành cho đánh giá.

## 4. Phân tích
`analysis/vlabench_summary.py <run> --baseline vanilla` cho các phần sau.
- Success (Wilson CI) theo arm, track và task, kèm giá trị tham chiếu của model card.
- So sánh ghép cặp theo (track, task, episode): McNemar chính xác + Holm, bootstrap phân tầng theo task.
- Thống kê h: trung bình ± sd, min–max, tỉ lệ ξ = H, tỉ lệ ξ > h_unc.
- Chi phí: `policy ms / action`; overhead sạch lấy từ `check_policy.py`.
- AUROC của U trung bình trong episode với thất bại. Đây là chẩn đoán, không phải dự báo.

So sánh chéo benchmark (LIBERO 20 Hz, H = 10, delta; RoboCasa H = 50; VLABench 10 Hz, H = 10, pose tuyệt đối) chỉ nên dùng **hiệu số ghép cặp so với baseline của chính benchmark đó** và h / H. Không so success tuyệt đối giữa các benchmark.

## 5. Kết quả pilot (giai đoạn B, 2026-11-18; `outputs/vlabench/pilot/analysis/SUMMARY.md`)

Mỗi arm 5 task × 10 episode, trạng thái đầu ghép cặp. **Mẫu nhỏ, chỉ để kiểm pipeline, không dùng để kết luận.**

| arm | success | Δ so với vanilla (ghép cặp) | h trung bình ± sd | ξ = H | ξ > h_unc |
|---|---|---|---|---|---|
| vanilla (h = 5) | 19/50 (38%) | – | 5.00 | 71% | – |
| duac | 19/50 (38%) | +0.00 [−0.16, +0.16], thắng/thua 8/8 | 9.43 ± 1.23 | 68% | 78% |
| aac | 22/50 (44%) | +0.06 [−0.12, +0.24], thắng/thua 12/9, p = 0.66 | 8.92 ± 2.11 | 69% | 92% |

- DUAC và AAC chạy gần trọn chunk vì ξ thường bằng H (xem INTEGRATION mục 4). Số lần gọi model vì thế bằng khoảng một nửa vanilla.
- **Chi phí sạch mỗi lần gọi** (`check_policy.py`): vanilla 64.9 ms, DUAC 71.5 ms, AAC 86.0 ms.
- **Ước tính chi phí policy mỗi action**, nếu tính theo h trung bình đo ở pilot: vanilla 64.9 / 5 ≈ 13.0 ms; DUAC 71.5 / 9.43 ≈ 7.6 ms; AAC 86.0 / 8.92 ≈ 9.6 ms. Thời gian đo trong lúc chạy pilot không dùng được để so sánh vì các server có số client chờ khác nhau.
- Phần lợi về chi phí này đến từ việc chạy chunk dài hơn, chưa chắc là nhờ độ bất định. Cần so với `fixed_10` và `random_h` ở giai đoạn C.
- U trung bình của episode (DUAC) và entropy (AAC) tách được episode thất bại: AUROC 0.85 và 0.86. Đây là thống kê trên cả episode, chỉ mang tính chẩn đoán.
