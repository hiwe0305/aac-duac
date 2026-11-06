# O-LONG, Track A: AAC và DUAC trên LIBERO-10 với π0.5

**Phạm vi:** Track A của [docs/task3010.md](task3010.md): LIBERO-10 với `pi05_libero`, 3 suite × 6 arm × 500 episode = 9.000 episode, chạy xong ngày 2026-11-04 lúc 09:10.
**Code và số liệu:** repo `aac-duac`, output ở `/mnt/ceph-cold-storage/hieudd/o_long/aac-duac/A/`. Bảng thống kê được sinh bằng `analysis/track_a_results.py` (kết quả ở `A/analysis/track_a_results.{md,json}`).

> **Phân tích nguyên nhân chi tiết (8 biểu đồ, có bằng chứng định lượng):** [A/analysis/DIAGNOSIS.md](track_a/DIAGNOSIS.md)

---

## 1. Trả lời ngắn

- **AAC: không work. DUAC: không work** (theo tiêu chí mục 5, gộp 2 suite có nhiễu).
- Trên 2 suite nhiễu gộp lại (1.000 cặp episode), **Fixed-10 tốt nhất với 70.5%**. DUAC đạt 68.5% (kém −2.0 pp, CI 95% [−4.7, +0.7]). AAC đạt 67.5% (kém −3.0 pp, CI [−5.4, −0.7]).
- Cận trên CI của chênh lệch so với Fixed tốt nhất đều nhỏ hơn 3 pp (DUAC +0.7, AAC −0.7), nên theo tiêu chí là "không work".
- Trên LIBERO-10 clean, cả hai cũng không hơn: DUAC 94.2%, AAC 94.0%, Fixed-10 95.4%.
- **Lợi ích không đến từ việc chọn h theo state.** DUAC không hơn Fixed-8 (−0.8 pp), dù h trung bình gần như bằng nhau (8.5 so với 8). DUAC cũng không hơn Random-h (−1.1 pp), arm rút h ngẫu nhiên từ chính phân phối h của DUAC.
- AAC và DUAC **gọi model nhiều hơn** Fixed-10 (DUAC +16–19%, AAC +33–39%) mà không đạt success cao hơn.

---

## 2. Setup

### 2.1 Policy và thuật toán
| | |
|---|---|
| Policy | π0.5 `pi05_libero` (OpenPI, JAX), H = 10, 10 bước denoising |
| AAC | Code chính thức [Adaptive-Action-Chunking/libero](https://github.com/Adaptive-Action-Chunking/libero) (`select_chunk_size`, `gaussian_bernoulli`). N = 20 sample, α = 3, chạy sample 0 |
| DUAC | Không có code công bố. Viết theo paper: phương trình 6 (vết Jacobian Hutchinson tại τ = 0.9, K = 40, 7 chiều action) và phương trình 8 (bước nhảy của trung bình tiền tố); h = max(h_unc, ξ) với α = 3. Đã kiểm bằng số với bản tái lập của hactt13 và với code AAC chính thức |
| Fixed-5/8/10 | Chạy trên server DUAC, nên h mà DUAC đề xuất vẫn được ghi lại ở mọi decision |
| Random-h | Mỗi decision rút h ngẫu nhiên từ phân phối h thực thi của DUAC, cùng suite và cùng task |

### 2.2 Protocol (giống nhau ở mọi arm)
OffScreenRenderEnv 256×256, env seed 7, init state = trial id (trial 0–49, ghép cặp được giữa các arm), 10 bước chờ + tối đa 520 bước policy. Policy seed 0, RNG stateful. 50 trial/task, 10 task, nên mỗi suite × arm có 500 episode.

### 2.3 Chọn suite (Bước 1, calibration Fixed-10, 10 trial/task)
| Suite | Calibration | Ghi chú |
|---|---|---|
| `libero_10` (clean) | DUAC 95.0% | Bước 0.5: khớp mức tham chiếu 95.0%; mean h 8.42 (tham chiếu 8.45) |
| **`libero_10_object`** | 68.0% | Được chọn: suite duy nhất nằm trong 30–70% |
| **`libero_10_env`** | 74.0% | Được chọn: gần 70% nhất trong các suite còn lại |
| `libero_10_with_mug` | 96.0% | Loại: gần trần |
| `libero_10_with_milk` | 82.5% | Loại: chỉ có 10 init/task, task 3 và 4 không có init nào |
| `libero_10_lan` / `libero_10_swap` | 97.0% / 7.0% | Loại: trần / sàn |
| `with_red_box`, `with_red_stick`, `with_blue_stick`, `with_diffpos_stick` | — | Không chạy được: thiếu asset `red_box`, `red_sticker`, `blue_red_sticker`, kể cả trong repo LIBERO-PRO chính thức |

### 2.4 Khác với task (đã ghi ở mục `deviations` trong `manifest.json`)
1. Suite nhiễu là `libero_10_object` và `libero_10_env` (nhiễu chính thức của LIBERO-PRO), không phải `libero_10_with_*`. Lý do ở mục 2.3.
2. `libero_10_env` ở mức 74%, hơi cao hơn ngưỡng 70%.
3. Chạy chính bắt đầu trước khi Hà Cao duyệt manifest.
4. Lúc 02:00 đổi cách xếp lịch: chia việc theo (suite, arm, task) và chạy trên một hàng đợi chung, **protocol không đổi**. 1.138 episode đã xong được giữ nguyên. 10 episode đang chạy dở được chạy lại từ đầu. Server khởi động lại nên RNG bắt đầu lại từ seed 0.

---

## 3. Kết quả

### T1. Tỉ lệ thành công (%), Wilson 95% CI, mỗi ô 500 episode
| Arm | `libero_10` (clean) | `libero_10_object` | `libero_10_env` | **2 suite nhiễu gộp (n = 1.000)** |
|---|---|---|---|---|
| DUAC | 94.2 [91.8, 95.9] | 67.8 [63.6, 71.7] | 69.2 [65.0, 73.1] | 68.5 [65.6, 71.3] |
| AAC | 94.0 [91.6, 95.8] | 67.0 [62.8, 71.0] | 68.0 [63.8, 71.9] | 67.5 [64.5, 70.3] |
| Fixed-5 | 93.8 [91.3, 95.6] | 67.6 [63.4, 71.6] | 69.4 [65.2, 73.3] | 68.5 [65.6, 71.3] |
| Fixed-8 | 94.6 [92.3, 96.3] | 65.0 [60.7, 69.1] | **73.6** [69.6, 77.3] | 69.3 [66.4, 72.1] |
| **Fixed-10** | **95.4** [93.2, 96.9] | **68.2** [64.0, 72.1] | 72.8 [68.7, 76.5] | **70.5** [67.6, 73.2] |
| Random-h | 95.0 [92.7, 96.6] | 69.0 [64.8, 72.9] | 70.2 [66.0, 74.0] | 69.6 [66.7, 72.4] |

Mọi khoảng tin cậy đều chồng lên nhau. Khoảng cách giữa arm tốt nhất và kém nhất chỉ là 3 pp trên suite nhiễu gộp, và 1.6 pp trên clean.

### T2. Chênh lệch ghép cặp (2 suite nhiễu gộp)
Chênh lệch tính bằng pp. CI lấy bằng bootstrap theo cụm (task, trial), 5.000 lần. p là McNemar exact; cột "Holm" hiệu chỉnh cho 6 phép so sánh {AAC, DUAC} × {Fixed-5, 8, 10}.

| So sánh | Chênh | 95% CI | Chỉ A thắng / chỉ B thắng | p | p Holm |
|---|---|---|---|---|---|
| DUAC − Fixed-5 | +0.0 | [−2.5, +2.4] | 83 / 83 | 1.00 | 1.00 |
| DUAC − Fixed-8 | −0.8 | [−3.3, +1.6] | 77 / 85 | 0.58 | 1.00 |
| DUAC − Fixed-10 | −2.0 | [−4.7, +0.7] | 78 / 98 | 0.15 | 0.76 |
| DUAC − Random-h | −1.1 | [−3.6, +1.5] | 73 / 84 | 0.43 | — |
| AAC − Fixed-5 | −1.0 | [−3.4, +1.5] | 80 / 90 | 0.49 | 1.00 |
| AAC − Fixed-8 | −1.8 | [−4.3, +0.6] | 69 / 87 | 0.17 | 0.76 |
| AAC − Fixed-10 | −3.0 | [−5.4, −0.7] | 63 / 93 | 0.020 | 0.12 |
| AAC − Random-h | −2.1 | [−4.4, +0.2] | 63 / 84 | 0.10 | — |
| AAC − DUAC | −1.0 | [−3.4, +1.4] | 80 / 90 | 0.49 | — |

**Từng suite** (pp, McNemar p):
| So sánh | clean | object | env |
|---|---|---|---|
| DUAC − Fixed-10 | −1.2 (0.44) | −0.4 (0.91) | −3.6 (0.09) |
| DUAC − Fixed-8 | −0.4 (0.87) | +2.8 (0.12) | **−4.4 (0.03)** |
| AAC − Fixed-10 | −1.4 (0.35) | −1.2 (0.53) | **−4.8 (0.02)** |
| AAC − Fixed-8 | −0.6 (0.75) | +2.0 (0.27) | **−5.6 (<0.01)** |

Ghi chú về các chênh lệch:
- Chênh lệch có ý nghĩa thống kê duy nhất (chưa hiệu chỉnh) là AAC và DUAC **kém hơn** Fixed-8 và Fixed-10 trên `libero_10_env`.
- Trên `libero_10_object`, DUAC hơn Fixed-8 +2.8 pp nhưng không có ý nghĩa (p = 0.12), và lại kém Fixed-10.

### T3. Chênh lệch theo từng task (2 suite nhiễu gộp, pp) và khi bỏ task có lợi ích lớn nhất
| So sánh | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | **bỏ task tốt nhất** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DUAC − Fixed-10 | +1 | +0 | −5 | −1 | +3 | +4 | −3 | −1 | −13 | −5 | **−2.7** (bỏ t5) |
| DUAC − Fixed-8 | +1 | +2 | −9 | +3 | −4 | +6 | +2 | +0 | −3 | −6 | **−1.6** (bỏ t5) |
| DUAC − Fixed-5 | +4 | +2 | −9 | −5 | −4 | +7 | −2 | −1 | +10 | −2 | **−1.1** (bỏ t8) |
| AAC − Fixed-10 | −3 | +0 | −4 | +1 | +7 | −5 | +0 | +0 | −15 | −11 | **−4.1** (bỏ t4) |
| AAC − Fixed-8 | −3 | +2 | −8 | +5 | +0 | −3 | +5 | +1 | −5 | −12 | **−2.6** (bỏ t3) |
| AAC − Fixed-5 | +0 | +2 | −8 | −3 | +0 | −2 | +1 | +0 | +8 | −8 | **−2.0** (bỏ t8) |

- **Không có task nào mà phương pháp adaptive thắng ổn định** trước mọi arm Fixed. Dấu của chênh lệch đổi theo task và theo arm so sánh, trong khoảng ±15 pp, đúng như mức nhiễu của 100 cặp episode mỗi task.
- Task 8 (*put both moka pots on the stove*) và task 9 (*put the yellow and white mug in the microwave and close it*) là nơi AAC và DUAC kém Fixed-10 nhiều nhất (−5 đến −15 pp).
- Khác với kết quả trước đây trên LIBERO-Object (+2.7 pp, gần như toàn bộ đến từ task cream cheese), ở đây không có task nào kéo kết quả lên.

**Success theo task trên suite nhiễu (%), 50 episode mỗi ô:**
| Task | Suite | DUAC | AAC | F-5 | F-8 | F-10 | Rand-h |
|---|---|---|---|---|---|---|---|
| 0 soup + tomato sauce → giỏ | object | 2 | 0 | 0 | 0 | 0 | 0 |
| 7 soup + cream cheese → giỏ | object | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 mug → microwave, đóng | object | 46 | 26 | 40 | 34 | 40 | 36 |
| 5 sách → caddy | object | 88 | 76 | 90 | 72 | 78 | 86 |
| 2 bật bếp + moka | env | 44 | 40 | 54 | 60 | 44 | 42 |
| 3 bát → ngăn kéo, đóng | env | 28 | 28 | 40 | 24 | 26 | 32 |
| 8 2 moka → bếp | env | 18 | 12 | 4 | 24 | 36 | 20 |
| 9 mug → microwave, đóng | env | 40 | 48 | 50 | 64 | 56 | 50 |
| Các task còn lại | cả 2 | 72–100 | 72–100 | 62–100 | 82–100 | 80–100 | 74–100 |

Task 0 và 7 của `object` thất bại ở mọi arm (0–2%). Có 100/1.000 cặp episode không phân biệt được gì, làm giảm lực thống kê của phép so sánh.

### T4. Hiệu quả
| Arm | h trung bình | Decision mỗi episode (clean / object / env) | infer_ms p50 / p95 | Phân bố h |
|---|---|---|---|---|
| Fixed-10 | 10 | 27.2 / 34.6 / 33.1 | 130 / 169 | 10: 100% |
| Fixed-8 | 8 | 33.6 / 44.2 / 41.0 | 130 / 165 | 8: 100% |
| DUAC | 8.45 / 8.65 / 8.59 | 32.5 / 40.3 / 39.5 | 131 / 168 | 9: 54–58%, 10: 19–26%, phần còn lại rải từ 2 đến 8 |
| Random-h | 8.44 / 8.65 / 8.60 | 32.2 / 39.9 / 39.0 | 131 / 167 | giống DUAC |
| AAC | 7.23 / 7.58 / 7.46 | 37.7 / 46.0 / 45.7 | 117–167 / 176–203 | rải từ 4 đến 10, đỉnh ở 9–10 (khoảng 45%) |
| Fixed-5 | 5 | 55.3 / 70.0 / 67.6 | 129 / 165 | 5: 100% |

- `infer_ms` đo trong lúc 14 client dùng chung 2 server, nên lớn hơn thời gian một lần gọi chạy riêng (DUAC khoảng 61 ms, AAC khoảng 78 ms). Các arm Fixed cũng chạy trên server DUAC (để ghi h đề xuất), nên `infer_ms` của chúng bằng DUAC.
- Episode thành công dài như nhau ở mọi arm (khoảng 251–269 bước).
- **DUAC gọi model nhiều hơn Fixed-10 khoảng 16–19%, AAC nhiều hơn khoảng 33–39%**, cả hai đều không đạt success cao hơn.

### T5. Lợi ích (nếu có) nằm ở đâu

**a) Tỉ lệ hoàn thành bước con (%)**
"Bước con 1" là có ít nhất một goal predicate đúng; "bước con 2" là mọi predicate đều đúng.

| Arm | clean: con 1 / con 2 | object: con 1 / con 2 | env: con 1 / con 2 |
|---|---|---|---|
| DUAC | 99.4 / 94.6 | 85.2 / 69.0 | 92.6 / 69.8 |
| AAC | 100 / 95.2 | 84.2 / 68.2 | 92.0 / 68.4 |
| Fixed-5 | 99.6 / 95.4 | 85.2 / 68.4 | 90.2 / 69.8 |
| Fixed-8 | 99.8 / 95.2 | 82.0 / 65.4 | **95.2 / 74.2** |
| Fixed-10 | 99.4 / 95.4 | 84.8 / 68.8 | 93.2 / 73.0 |
| Random-h | 99.4 / 96.0 | 85.2 / 69.6 | 92.4 / 71.4 |

- Bước con 1 hoàn thành ở khoảng bước 123–141, bước con 2 ở khoảng 252–269 (tính trên episode thành công), gần như nhau ở mọi arm.
- Phần mất mát nằm ở **bước con 2** (chênh 15–24 pp giữa "con 1" và "con 2" trên suite nhiễu). Không arm nào cải thiện riêng bước con 2.

**b) h mà thuật toán chọn theo giai đoạn của episode**
Các cột "Trước con 1", "Giữa", "Sau con 2" là h trung bình theo giai đoạn so với các bước con. Hai cột cuối là h trung bình khi chunk có hoặc không có đóng/mở gripper.

| Suite | Arm | Trước con 1 | Giữa | Sau con 2 | Chunk có đóng/mở gripper | Chunk không có |
|---|---|---|---|---|---|---|
| clean | DUAC | 8.42 | 8.43 | 8.75 | **9.31** | 8.35 |
| clean | AAC | 7.28 | 7.12 | 7.52 | **9.05** | 7.05 |
| object | DUAC | 8.62 | 8.70 | 8.60 | **9.34** | 8.55 |
| object | AAC | 7.65 | 7.55 | 7.22 | **9.17** | 7.41 |
| env | DUAC | 8.58 | 8.56 | 8.85 | **9.38** | 8.50 |
| env | AAC | 7.53 | 7.35 | 7.72 | **9.15** | 7.29 |
| (đối chứng) | Random-h | 8.36–8.67 | 8.46–8.61 | 8.73–9.08 | 8.78–9.09 | 8.40–8.59 |

- **h gần như không đổi giữa các giai đoạn**: lệch tối đa khoảng 0.4 bước. Thuật toán không phản ứng theo cấu trúc bước con của task dài.
- **Ngược với giả thuyết**, quanh lúc đóng/mở gripper cả AAC lẫn DUAC chọn h **dài hơn** (khoảng 9.1–9.4 so với 7.1–8.6), chứ không ngắn hơn. Nguyên nhân:
  - Khi gắp, tay robot di chuyển chậm, nên chuyển động cộng dồn khó vượt ngưỡng α = 3.
  - Vì vậy cận dưới ξ bị đẩy lên gần H = 10 và quyết định h.
  - Kết quả là đúng ở những thời điểm mà giả thuyết O-LONG cho rằng cần replan sớm, ξ với α = 3 lại ngăn việc đó.

**c) U của DUAC có báo trước thất bại không**
AUROC dự đoán thất bại từ U trung bình, tính trên arm DUAC.
| Suite | Cả episode | 5 decision đầu, trong cùng task |
|---|---|---|
| clean | 0.48 | 0.51 |
| object | 0.74 | 0.38 |
| env | 0.76 | 0.61 |

Nhìn cả episode, U cao hơn ở episode thất bại (AUROC khoảng 0.75). Nhưng chỉ xét giai đoạn đầu và so trong cùng task thì AUROC về khoảng 0.4–0.6. Như vậy U cao chủ yếu là **hậu quả** của thất bại (robot kẹt ở trạng thái lạ đến hết 520 bước) cộng với khác biệt giữa các task, không phải tín hiệu báo trước. Điều này giải thích vì sao chọn h theo U không cứu được episode.

### T6. Phân loại thất bại (tự động, dựa trên bước con)
Task yêu cầu đánh nhãn tay bằng cách xem video cho 20 episode mỗi arm (không với tới vật, gắp trượt, rơi vật, sai vật, hết giờ ở bước con 2). **Việc này chưa làm.** Bảng dưới là phân loại tự động từ `subgoals.csv`. Mọi episode thất bại đều chạy hết 520 bước; không có episode nào kết thúc sớm.

| Suite | Arm | Số episode fail | Không đạt bước con nào | Chỉ đạt 1 predicate | Đạt cả hai nhưng không cùng lúc |
|---|---|---|---|---|---|
| object | DUAC / AAC / F-10 | 161 / 165 / 159 | 74 / 79 / 76 | 79 / 79 / 79 | 8 / 7 / 4 |
| env | DUAC / AAC / F-10 | 154 / 160 / 136 | 37 / 40 / 34 | 86 / 94 / 87 | 31 / 26 / 15 |

- Trên `object`, khoảng một nửa số episode thất bại không đạt được bước con nào. Phần lớn số này rơi vào task 0 và 7, nơi mọi arm đều thất bại.
- Trên `env`, phần lớn thất bại xảy ra **ở bước con 2**.
- Fixed-10 có ít episode thất bại ở nhóm "đạt cả hai nhưng không cùng lúc" hơn (15 so với 26–31). Có thể vật đã đặt đúng chỗ rồi bị xô lệch đi; cần xem video để xác nhận.
- Video của trial 0–1 mỗi task nằm ở `A/<suite>/<arm>/videos/`.

---

## 4. Diễn giải

1. **Trên task dài hơn và khó hơn, chọn horizon thích ứng không mang lại lợi ích.** Kết quả trước đây trên LIBERO-Object có dịch vị trí (+2.5–2.7 pp, gần như toàn bộ từ 1 task) không lặp lại được. Trên LIBERO-10 có nhiễu, chênh lệch là 0 đến −3 pp.
2. **Hai thuật toán gần như không thích ứng theo state.** Phân bố h của DUAC tập trung ở 9–10 và gần như giống nhau ở mọi giai đoạn của episode. Random-h với cùng phân phối h cho kết quả tương đương hoặc tốt hơn DUAC. Kết quả này khớp với thí nghiệm oracle E3 trước đây (chọn h theo state chỉ mang lại khoảng +2.7 pp).
3. **Cận dưới chuyển động ξ (α = 3) chiếm ưu thế ở thời điểm quan trọng nhất.** Quanh lúc đóng/mở gripper nó buộc h dài hơn. Đây là ứng viên rõ nhất nếu muốn sửa thuật toán, nhưng task cấm chỉnh α trên dữ liệu chạy chính, nên chỉ có thể thử ở một thí nghiệm riêng.
4. **Với π0.5 trên LIBERO-10, horizon dài lại tốt hơn.** Fixed-10 (đúng bằng toàn bộ chunk) tốt nhất ở clean và trên 2 suite gộp, đồng thời gọi model ít nhất. Điều này giống pilot RoboCasa trước đây, nơi Fixed-25 tốt hơn Fixed-10.

## 5. Hạn chế
- **Lực thống kê:** 1.000 cặp episode trên suite nhiễu đủ để loại trừ lợi ích ≥ 3 pp. Chênh lệch dưới khoảng 2 pp thì không phân biệt được.
- **Chọn suite:** 2 suite nhiễu khác với kế hoạch (mục 2.4). `libero_10_env` ở mức 74%. `libero_10_object` có 2 task luôn thất bại, nên 20% số cặp episode không mang thông tin.
- **Một seed policy duy nhất.** Noise của mỗi decision không cố định theo episode, vì nhiều client dùng chung server. Server khởi động lại một lần khi đổi lịch chạy.
- **T6 mới là phân loại tự động.** Chưa đánh nhãn tay bằng video.
- **DUAC tự cài đặt theo paper**, vì không có code công bố. Đã kiểm bằng số với bản tái lập của hactt13 (phương trình 6, 8 và ξ khớp) và với vết Jacobian chính xác (sai số 0.13).
- `infer_ms` đo khi GPU dùng chung, nên không phải latency của một lần gọi riêng lẻ.

## 6. Câu hỏi mở và bước tiếp theo
1. **Thử α nhỏ hơn hoặc bỏ ξ** (arm "DUAC chỉ dùng phương trình 8"; cần thêm một cờ vào evaluator) trong một thí nghiệm riêng, có manifest riêng, để xem quanh lúc gắp h có ngắn lại và success có thay đổi không.
2. **Đánh nhãn tay T6:** 20 episode thất bại mỗi arm, ưu tiên `env` task 2, 3, 8, 9.
3. **RoboCasa365 với π0.5** (chunk 50, horizon trung vị 2.250 bước): pipeline AAC/DUAC đã có. Smoke test đang chạy. Trên RoboCasa, h của DUAC thay đổi từ 4 đến 49, nên đây là nơi kiểm giả thuyết tốt hơn LIBERO (H = 10).
4. Dùng state bank (`A/*/*/state_bank/*.npz`) cho thí nghiệm phản thực: từ cùng một state, so sánh nhánh replan sớm với nhánh replan muộn ở các bước gần thời điểm thất bại.

---

### Phụ lục: file và lệnh tái lập
| | |
|---|---|
| Kết quả từng run | `A/<suite>/<arm>/{episodes,decisions,subgoals}.csv`, `algo_log.jsonl`, `state_bank/`, `videos/`, `summary.json` |
| Thống kê T1–T3 và tiêu chí mục 5 | `.venv-libero/bin/python analysis/track_a_results.py` → `A/analysis/track_a_results.{md,json}` |
| Calibration | `analysis/calibration_summary.py outputs/calibration` |
| Manifest và các deviation | `manifest.json` |
| Lịch chạy | `tools/schedule_track_a.py`, log ở `A/logs_sched/` |
