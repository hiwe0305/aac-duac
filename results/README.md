# Kết quả O-LONG (bản nhẹ, để chia sẻ)

Đây là bản rút gọn của `outputs/`. Bản đầy đủ khoảng 5.4 GB nằm trên server (xem phần cuối). Báo cáo và hình (cũng có bản sao trong `docs/`):

- **Báo cáo Track A:** [REPORT.md](REPORT.md)
- **Phân tích nguyên nhân (8 biểu đồ):** [track_a/analysis/DIAGNOSIS.md](track_a/analysis/DIAGNOSIS.md)
- **Bảng thống kê T1–T3 và tiêu chí mục 5:** [track_a/analysis/track_a_results.md](track_a/analysis/track_a_results.md)

## Nội dung
| Đường dẫn | Nội dung |
|---|---|
| `track_a/<suite>/<arm>/episodes.csv` | 500 dòng, mỗi dòng 1 episode (10 task × 50 trial). Cột: `task_id`, `trial_id`, `success`, `end_step`, `n_decisions`, `mean_h`, `wall_s`, `instruction`, ... |
| `track_a/<suite>/<arm>/subgoals.csv` | Mỗi dòng 1 goal predicate × 1 episode: `predicate`, `first_true_step` (để trống nếu predicate chưa bao giờ đúng) |
| `track_a/<suite>/<arm>/summary.json`, `args.json` | Success, mean h, success theo task; tham số của run |
| `track_a/analysis/track_a_results.{md,json}` | T1 (Wilson CI), T2 (McNemar exact, bootstrap theo cụm task × trial 5.000 lần, Holm), T3 (theo task, bỏ task tốt nhất), kết luận theo mục 5 |
| `track_a/analysis/DIAGNOSIS.md`, `figures/F1–F8.png` | Phân tích nguyên nhân và 8 biểu đồ |
| `track_a/analysis/diagnosis.json` | Số liệu đứng sau các hình F1–F8 (h_unc/ξ, profile U, h quanh sự kiện, AUROC, các cặp khác kết quả) |
| `calibration/` | Bước 1: Fixed-10, 10 trial/task, trên các suite ứng viên; Bước 0.5: DUAC clean. Có `calibration_summary.txt` |

Suite: `libero_10` (clean), `libero_10_object`, `libero_10_env`. Arm: `duac`, `aac`, `fixed_5`, `fixed_8`, `fixed_10`, `random_h`.
Các episode ghép cặp được giữa mọi arm qua cặp (`task_id`, `trial_id`): cùng env seed 7, cùng init state.

Tính lại bảng thống kê từ chính các file này:
```python
import csv
def load(suite, arm):
    return {(int(r["task_id"]), int(r["trial_id"])): int(r["success"])
            for r in csv.DictReader(open(f"results/track_a/{suite}/{arm}/episodes.csv"))}
duac, f10 = load("libero_10_env", "duac"), load("libero_10_env", "fixed_10")
print("DUAC thắng / Fixed-10 thắng:", sum(duac[k] > f10[k] for k in duac), sum(f10[k] > duac[k] for k in duac))
```

## Dữ liệu lớn (không có trên GitHub)
Nằm trên server 3, ở `/mnt/ceph-hot-storage/hieudd/o_long/aac-duac/` (là `outputs/` trong repo):
| Dữ liệu | Dung lượng |
|---|---|
| `A/<suite>/<arm>/decisions.csv`: mỗi decision (h, h_unc, ξ, profile U hoặc entropy, infer_ms) | 240 MB |
| `A/<suite>/<arm>/algo_log.jsonl`: toàn bộ thông số toán học của AAC/DUAC ở mỗi decision | 754 MB |
| `A/<suite>/<arm>/state_bank/*.npz`: MuJoCo state, chunk, action, 20 sample của AAC | khoảng 3 GB |
| `A/<suite>/<arm>/videos/`: trial 0–1 của mỗi task | 79 MB |

Các file này được sinh bằng `analysis/track_a_results.py` và `analysis/diagnose_track_a.py`. Cả 2 script đọc từ `outputs/`.
