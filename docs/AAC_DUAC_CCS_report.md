# AAC, DUAC và CCS (medoid): chạy lại, khoảng trống, phương pháp và hướng tiếp theo

- **Người thực hiện:** Hiếu · **Cập nhật:** 2026-11-26 · **Repo:** `aac-duac`, nhánh `ccs-chunk-selection` (commit `b1e0cb6`)
- **Phạm vi:** mọi con số dưới đây là kết quả đo, ghép cặp theo episode (cùng cấu hình episode, cùng seed), trừ khi ghi "theo paper".
- **Tài liệu chi tiết:**
  - [CCS_proposal.md](CCS_proposal.md)
  - [REPORT_trackA.md](REPORT_trackA.md), [track_a/DIAGNOSIS.md](track_a/DIAGNOSIS.md)
  - `results/vlabench/full/REPORT_CCS.md`, `results/track_a/SELECT_REPORT.md`, `results/robocasa/main/REPORT.md`

---

## Tóm tắt

1. **Chạy đúng setting của paper thì khớp paper; đổi setting thì AAC/DUAC không thắng h cố định tốt nhất.**
   - Trên LIBERO-Pro dịch vị trí (setting của paper DUAC), cả hai hơn Fixed-10 khoảng +2.5 pp, đúng như paper.
   - Trên ba bộ dữ liệu khó và dài hơn mà mình chạy (LIBERO-10 có nhiễu, RoboCasa365 composite, VLABench), AAC và DUAC đều không thắng h cố định tốt nhất.
   - Lý do chính: paper so với h mặc định hoặc Fixed-10 (không phải h cố định tốt nhất); lợi ích dồn vào một task; tín hiệu bất định không phản ánh trạng thái.
2. **CCS / medoid:** từ cùng 20 mẫu mà AAC đã sinh, chạy mẫu ở trung tâm (medoid) thay vì mẫu 0, với h cố định.
   - VLABench: 25.5% → **36.8%** (+11.3 pp).
   - LIBERO-10: chỉ +0.7 pp, không có ý nghĩa thống kê.
   - Medoid **không phải ý tưởng mới** (KeyStone, KDPE), nhưng hơn bản cài KeyStone 6 pp trên VLABench.
3. **Chi phí:** medoid dùng đúng server của AAC (khoảng 86 ms mỗi lần gọi, +33% so với gọi 1 mẫu). Phần chọn mẫu chạy trên CPU, khoảng 4 ms. Không cần train.
4. **Hướng tiếp:** mục 4, do agent Fable 5.2 khảo sát tài liệu và rút ra khoảng trống nghiên cứu.

---

## 1. Vì sao kết quả chạy lại của AAC và DUAC khác paper, và khoảng trống của hai paper

### 1.1 Hai paper claim gì, trên setting nào

| | **AAC** (Liang et al., CVPR 2026, arXiv 2604.04161) | **DUAC** (Ben-Ami et al., RoboPAD @ NeurIPS 2026 workshop) |
|---|---|---|
| Tín hiệu | Entropy của action trên N = 20 mẫu, tính theo từng bước trong chunk | Độ bất định của từng action, tính từ divergence của velocity field (vết Jacobian ước lượng Hutchinson, τ = 0.9, K = 40) trên **một** chunk |
| Luật chọn h | h = điểm có bước nhảy lớn nhất của entropy trung bình tiền tố, với cận dưới chuyển động ξ (α = 3) | Giống AAC (dùng lại luật và ξ của AAC) |
| Chunk được chạy | **Mẫu 0** | Chunk duy nhất |
| Backbone chính | **GR00T N1.5** đã fine-tune (π0.5 chỉ ở Table 2–3) | π0.5 `pi05_libero`; LeRobot `pi05_robocasa` |
| Benchmark | LIBERO 4 suite (50 lần chạy/task), RoboCasa Kitchen-24 (100 lần chạy/task), LIBERO-Pro (π0.5), robot thật 3 task × 20 lần | LIBERO 4 suite, LIBERO-Pro `libero_object_temp_x0.1–0.4`, LIBERO-Plus, RoboCasa Kitchen-24 |
| Baseline | **h mặc định** (GR00T h = 16; π0.5 "vanilla"), cộng quét h cố định cho GR00T | Fixed-10 (LIBERO), Fixed-25 / Fixed-50 (RoboCasa), AutoHorizon, AAC |
| Kết quả nêu bật | LIBERO 94.1 → 95.0; RoboCasa 59.7 → 62.0; π0.5 LIBERO 97.0 → 97.9; LIBERO-Pro (π0.5) 30.9 → 34.8; robot thật 67 → 82 | LIBERO-Pro: Fixed-10 47.2, AAC 48.7, **DUAC 49.2**; LIBERO 97.4 / 97.5 / **98.0**; RoboCasa: Fixed-25 51.79, **DUAC 52.33** |
| Thống kê | Không có CI, không kiểm định ghép cặp | Không có CI, không kiểm định ghép cặp |
| Chi phí theo paper | Bảng 4: N = 1 → 20 tốn 83 → 106 ms | Bảng II: Fixed-10 68 ms (1.0×), **AAC 583 ms (8.5×)**, DUAC 204 ms (3.0×) |

**Paper AAC tự cho thấy AAC chỉ ngang h cố định tốt nhất** (Bảng 1, GR00T):

| | h mặc định = 16 | h = 8 | AAC |
|---|---|---|---|
| LIBERO avg | 94.1 | 94.7 | 95.0 (+0.3 so với h = 8) |
| LIBERO-Long | 88.8 | **93.4** | 92.8 (−0.6) |
| RoboCasa avg | 59.7 | 61.2 | 62.0 (+0.8) |

Mức tăng nêu ở phần chính (+0.9 và +2.3 pp) là so với h mặc định.

### 1.2 Các bộ dữ liệu đã chạy

| Bộ | Benchmark / suite | Policy | Episode mỗi arm | Đặc điểm | Arm chính |
|---|---|---|---|---|---|
| **R0: tái lập đúng setting paper** (do anh Hà Cao, hactt13) | LIBERO-Pro `libero_object_temp_x{0.1,0.2,0.3,0.4}` + LIBERO 4 suite | π0.5 `pi05_libero`, H = 10 | 2.000 (LIBERO-Pro); 500 mỗi suite | Đúng protocol của DUAC: seed 7, `max_steps` + 10 bước chờ | Fixed-10, AAC, DUAC |
| **A: Track A, task dài có nhiễu** | LIBERO-10 clean, `libero_10_object`, `libero_10_env` (LIBERO-PRO) | π0.5 `pi05_libero`, H = 10 | 500 mỗi suite (10 task × 50) | Task 2 bước con, tối đa 520 bước; 2 suite nhiễu được chọn trước sao cho Fixed-10 đạt 30–70% | Fixed-5/8/10, DUAC, AAC, Random-h, medoid_h8/h10, sample0_h8 |
| **B: RoboCasa365 composite** | 5 task composite (`KettleBoiling`, `LoadDishwasher`, `PreSoakPan`, `ScrubCuttingBoard`, `StackBowlsCabinet`) | LeRobot `pi05_robocasa` (PyTorch), **H = 50** | 100 (5 × 20) | Episode dài 1.500–2.400 bước | Fixed-10/25/50, DUAC, AAC |
| **C: VLABench** | 5 track: T1 in-distribution, T2 cross-category, T3 common sense, T4 semantic instruction, T6 unseen texture; 10 primitive task | `VLABench/pi05-primitive-10task` (OpenPI JAX), H = 10, action là pose tuyệt đối | 2.460 (5 × 10 × 50; T2 insert_flower 10) | Success cơ sở thấp (khoảng 25%) | vanilla (h = 5), fixed_8/10, AAC, DUAC (+ ablation, random-h), các arm chọn chunk |

### 1.3 Chỗ setting trùng nhau: chạy lại khớp paper

| Setting | Paper | Chạy lại | Khớp? |
|---|---|---|---|
| LIBERO-Pro dịch vị trí x, tổng 4 mức (DUAC Bảng I) | Fixed-10 47.2 / AAC 48.7 / DUAC 49.2 | 47.05 / 49.75 / 49.5 (n = 2.000) | ✅ |
| Hiệu so với Fixed-10 trên LIBERO-Pro | DUAC +2.0, AAC +1.5 | DUAC **+2.5 pp, p = 0.011**; AAC **+2.7 pp, p = 0.005** | ✅ (còn có ý nghĩa thống kê) |
| LIBERO chuẩn, DUAC (Bảng VI) | 98.0 | 97.9 | ✅ |
| LIBERO-Pro x0.2–0.4, π0.5 (AAC Bảng 3) | 30.9 → 34.8 | Fixed-10 32.4 → AAC 36.4 | ✅ cùng chiều, cùng độ lớn |
| Task moka pot (DUAC Bảng IV) | DUAC 75%, h trung bình 8.83 | DUAC 80% (50 episode), h 8.88 | ✅ |
| VLABench, AAC so với h mặc định (h = 5) | (không có trong paper) | **+2.7 pp [+0.6, +4.8]** | Cùng kiểu với mức tăng trong paper |

**Chi tiết quan trọng của R0:** trên LIBERO-Pro, phần lớn lợi thế đến từ **một task** (cream cheese): Fixed-10 54.5%, AAC 77.5%, DUAC 73.5%. Chín task còn lại chỉ chênh vài điểm.

### 1.4 Chỗ setting khác: AAC và DUAC không thắng h cố định tốt nhất

| Bộ | h cố định tốt nhất | AAC | DUAC | Random-h | Kết luận (CI 95% so với fixed tốt nhất) |
|---|---|---|---|---|---|
| A: LIBERO-10, 2 suite nhiễu (n = 1.000) | Fixed-10 **70.5%** | 67.5% (−3.0 [−5.4, −0.7]) | 68.5% (−2.0 [−4.7, +0.7]) | 69.6% | AAC kém hơn; DUAC chưa kết luận |
| A: LIBERO-10 clean (n = 500) | Fixed-10 **95.4%** | 94.0% | 94.2% | 95.0% | Không hơn |
| B: RoboCasa365 (n = 100) | Fixed-50 **25%** | 15% | 20% | — | Chưa kết luận (cỡ mẫu nhỏ) |
| C: VLABench (n = 2.460) | fixed_8/10 **25.9%** | 25.4% | 24.3% | 25.2% | Không hơn; DUAC kém hơn ở T2, T3 |

### 1.5 Phân tích nguyên nhân khác biệt

| # | Nguyên nhân | Bằng chứng |
|---|---|---|
| 1 | **Baseline yếu hơn trong paper.** Paper so với h mặc định (AAC) hoặc Fixed-10 (DUAC), không so với h cố định tốt nhất | Trong chính paper AAC, h = 8 gần bằng AAC (+0.3 / +0.8 pp, LIBERO-Long còn thắng). Trên VLABench, AAC hơn h = 5 mặc định +2.7 pp nhưng thua fixed_8 −0.5 pp |
| 2 | **Lợi ích dồn vào một task / một kiểu nhiễu.** LIBERO-Pro x-position có task cream cheese hưởng lợi +19 đến +23 pp | Đổi sang LIBERO-10 (task 2 bước, nhiễu object/env) thì lợi ích biến mất |
| 3 | **Tín hiệu bất định không phản ánh trạng thái.** | DUAC: profile U hình chữ U cố định; U₁₀ > U₉ ở 93% decision nên h_unc = 9 ở 72%; tự tương quan giữa hai decision liền nhau −0.04 (LIBERO), khoảng 0.1 (VLABench). AAC: h_entropy dồn về 2 hoặc 9 |
| 4 | **Ngưỡng chuyển động ξ quyết định h, theo hướng ngược mục tiêu.** | ξ quyết định h ở khoảng 74% decision (VLABench). Quanh lúc gắp/nhả, ξ tăng từ khoảng 5.7 lên 8.1 nên robot replan **chậm hơn** đúng lúc cần chính xác. Bỏ ξ: DUAC không đổi (24.6%), AAC sụp còn 16.3% (h̄ ≈ 2.9) |
| 5 | **Không có đối chứng random-h hay h trung bình.** Paper không tách được "chọn h theo state" khỏi "đổi h trung bình" | Random-h cùng phân phối h: LIBERO 69.6% (≥ DUAC 68.5%), VLABench 25.2% (≈ DUAC không ξ 24.6%) |
| 6 | **Bất định không báo trước được thất bại.** | AUROC trong cùng task: 0.49 ở 5 decision đầu (LIBERO). U chỉ tăng *sau khi* episode đã hỏng |
| 7 | **Task dài và khó làm lộ điểm yếu.** Paper chủ yếu dùng task ngắn gần trần (LIBERO 94–98%) | Ở RoboCasa365 (h = 10 chỉ 11%), adaptive cũng không thắng; h ngắn có hại rõ |
| 8 | **Thống kê.** Paper báo chênh 0.5–2 pp không kèm CI. Với 500 episode mỗi suite, CI khoảng ±2–3 pp | Ví dụ LIBERO-10: paper Fixed-10 = 92.8 nhưng chạy lại 95.4 [93.2, 96.9]. Riêng con số này đảo dấu kết luận "DUAC hơn Fixed-10" trên LIBERO-10 (p khoảng 0.08) |
| 9 | **Chi phí phụ thuộc cách cài đặt.** | Paper DUAC báo AAC 583 ms (8.5×): con số này chỉ đúng nếu encode lại prefix cho từng mẫu (đo lại: 577 ms). Dùng chung KV cache thì AAC chỉ 82–86 ms (+31–33%). DUAC đo lại chỉ +4–10% (65–72 ms), không phải 3.0× |
| 10 | **Khác backbone và checkpoint.** | Phần lớn kết quả của AAC là trên GR00T N1.5 fine-tune riêng. RoboCasa của mình là composite với chunk 50, còn paper dùng Kitchen-24 atomic |
| 11 | **Khác protocol nhỏ.** | Theo protocol AAC (không cộng 10 bước chờ, seed 0/1/2, baseline Fixed-5): LIBERO-10 85.1 / 88.5, so với paper 92.5 / 95.2. Bước chờ ảnh hưởng mạnh tới task dài |

### 1.6 Khoảng trống của AAC và DUAC (chỉ xét hai paper này)

| # | Khoảng trống | Cơ sở |
|---|---|---|
| G1 | **Đánh giá:** không so với h cố định tốt nhất, không có random-h, không có CI hay kiểm định ghép cặp | 1.5 (#1, #5, #8) |
| G2 | **Tín hiệu không phụ thuộc state:** vị trí trong chunk quyết định h_unc (hiệu ứng rìa), ξ quyết định phần còn lại | 1.5 (#3, #4) |
| G3 | **Đo bất định trên thứ không được chạy:** AAC tính entropy trên 20 mẫu nhưng chạy mẫu 0 bất kỳ; độ phân tán giữa các mẫu được dùng để chọn h, không dùng để chọn **mẫu nào** | 20 mẫu khác nhau thật: hai mẫu xa nhất cách nhau 6.9 cm, ngang quãng đường cả chunk (7.7 cm); 16% decision bất đồng về gripper |
| G4 | **Giả định "bất định cao thì replan sớm" chưa được kiểm chứng:** bất định không báo trước thất bại, còn replan dày có hại | 1.5 (#4, #6); medoid_h5 kém medoid_h8 −3.1 pp |
| G5 | **Chỉ đánh giá task ngắn, gần trần.** Không có task composite dài hay benchmark success thấp | 1.4 |
| G6 | **Chi phí báo cáo không công bằng:** AAC bị tính chi phí theo cách cài đặt kém hiệu quả | 1.5 (#9) |
| G7 | **Không có giải thích cơ chế:** paper đưa ra hình minh hoạ h theo giai đoạn task, nhưng phân bố h thực tế giống hệt nhau ở mọi giai đoạn | DIAGNOSIS F2: h trung bình 8.60 / 8.63 / 9.19 qua các giai đoạn |

**Kết luận mục 1:** AAC và DUAC đúng trong phạm vi họ đo, nhưng claim "chọn h theo bất định tốt hơn h cố định" không đứng vững khi so với h cố định tốt nhất, khi có đối chứng random-h, và khi đổi bộ dữ liệu. Khoảng trống G3 (độ phân tán giữa các mẫu nên được dùng để **chọn mẫu**) là điểm xuất phát của CCS.

---

## 2. CCS: chọn action chunk thay vì chọn horizon

### 2.1 Ý tưởng

- Ở mỗi decision, π0.5 sinh **20 chunk** từ cùng một quan sát (khác noise ban đầu của flow). Đây là đúng các mẫu mà AAC đã sinh để tính entropy.
- AAC dùng chúng để chọn **h**, rồi chạy **mẫu 0**, tức là chọn ngẫu nhiên.
- CCS giữ **h cố định** và chuyển câu hỏi sang: **chạy chunk nào** trong 20 chunk?
- Phiên bản hiệu quả nhất là **medoid**: chunk có tổng khoảng cách tới các chunk khác nhỏ nhất, tức "thành viên điển hình nhất" của tập mẫu.
  - Khác với **trung bình**: trung bình có thể rơi vào giữa hai phương án hợp lệ (ví dụ đường đi thẳng vào vật cản giữa hai cách vòng trái và vòng phải).
  - Medoid luôn là **một mẫu thật** mà policy đã sinh ra.

### 2.2 Động cơ (đo trước khi làm)

| Quan sát | Số liệu |
|---|---|
| Các mẫu khác nhau thật | VLABench: khoảng cách xa nhất giữa 2 mẫu ở cuối chunk có trung vị 6.9 cm (quãng đường chunk 7.7 cm); LIBERO: 5.7 cm |
| Bất đồng về gripper | 15.8% (VLABench), 16.6% (LIBERO) decision có ≥ 20% mẫu khác ý về đóng/mở |
| Chọn mẫu ngẫu nhiên gây đổi kế hoạch | sample0: 40% decision có độ lệch kế hoạch > 2 cm giữa chunk mới và phần chưa chạy của chunk trước; medoid: 20% |
| Medoid có tác động thật | Medoid khác mẫu 0 ở 95% decision (gần như tất yếu với 1/20); khi khác, chunk lệch trung vị 1.8 cm (p90 khoảng 10 cm), và gripper khác hẳn ở 8.7% decision |

### 2.3 Thuật toán

**Ký hiệu.**
- N mẫu a⁽¹⁾…a⁽ᴺ⁾ ∈ ℝ^{H×7}: vị trí (3), hướng (3), gripper (1).
- h là số action sẽ chạy (cố định).
- Mọi khoảng cách chỉ tính trên h action đầu.

**Khoảng cách giữa hai chunk:**

  d(a, b) = (1/h) Σ_{t=1..h} [ ‖p_t(a) − p_t(b)‖₂ + w_rot · θ_t(a, b) ],  w_rot = 0.05 m/rad

trong đó θ là góc geodesic giữa hai hướng trên SO(3).

| Không gian action | Đưa về quỹ đạo | Gripper "mở" |
|---|---|---|
| VLABench: pose tuyệt đối (vị trí, euler xyz) | Dùng trực tiếp; góc geodesic, an toàn khi góc quấn ±π | g ≥ 0.1 |
| LIBERO: delta OSC (1.0 = 0.05 m / 0.5 rad) | Vị trí: cộng dồn × 0.05 m. Hướng: nhân rotation như robosuite, R_t = Exp(0.5 δ_t)···Exp(0.5 δ_1) | g < 0 |

**Medoid (phương pháp chính):**

  j\* = argmin_j (1/(N−1)) Σ_{k≠j} d(a⁽ʲ⁾, a⁽ᵏ⁾),  chạy a⁽ʲ\*⁾[:h], rồi gọi policy lại.

**CCS đầy đủ** (phiên bản đầu, thêm hai tiêu chí):
1. **Bỏ phiếu gripper:** chỉ giữ các mẫu lệch đa số ≤ 1 bước. Bước hoà 10/10 không bỏ phiếu.
2. **Liền mạch:** B_j = d(a⁽ʲ⁾[:L], phần chưa chạy của chunk trước), với L = H − h_prev.
3. j\* = argmin [ z(D_j) + λ·z(B_j) ], λ = 1, z là chuẩn hoá trong tập còn lại.

**Pseudo-code:**

```
mỗi decision:
    A = policy.sample(obs, N=20)          # 1 lần encode ảnh + lệnh (KV cache dùng chung), 20 noise
    j* = medoid(A[:, :h], d)              # CPU, khoảng 4 ms
    chạy A[j*][:h]                        # h = 8
```

### 2.4 Cách làm (triển khai)

| Thành phần | File |
|---|---|
| Thuật toán chọn: medoid, CCS, EGC, XDC, KeyStone, consensus horizon; self-test | `flowbased_aac/flowbased_aac/ccs.py` (`python -m flowbased_aac.ccs`) |
| Server N mẫu, dùng chung KV cache | `adaptive_action_chunking/tools/serve_pi05_multisample.py` (server của AAC, không sửa) |
| Evaluator VLABench | `flowbased_aac/flowbased_aac/vlabench/eval_vlabench.py --select {sample0,ccs,medoid,egc,xdc,mcons,keystone} --fixed-horizon K [--smooth-pos-max M]` |
| Evaluator LIBERO / LIBERO-PRO | `flowbased_aac/duac.py eval-liberopro --select ... --fixed-horizon K` |
| Chạy hàng loạt | `tools/run_vlabench_full.sh` (arm `<select>_h<K>`), `tools/schedule_track_a.py --stage select` |
| Report | `analysis/vlabench_ccs_report.py`, `analysis/track_a_select_report.py` |

**Protocol:**
- Mọi arm chọn chunk chạy trên **cùng** server AAC N = 20.
- `sample0_hK` là đối chứng trực tiếp: cùng 20 mẫu, cùng h, nhưng không chọn.
- Ghép cặp với arm cũ theo (track/suite, task, episode). Kiểm định McNemar exact, bootstrap phân tầng theo (track × task), hiệu chỉnh Holm.
- Tiêu chí đặt trước:
  - VLABench: CI so với cả sample0 và fixed tốt nhất nằm hoàn toàn trên 0.
  - Track A: tiêu chí trong `manifest.json` (hơn mọi Fixed ≥ 3 pp, vẫn hơn ≥ 1 pp khi bỏ task có lợi nhất, trên clean không kém quá 2 pp).
- Giới hạn VRAM ≤ 27 GB: chạy server với `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55`.

### 2.5 Kết quả

**VLABench** (2.460 episode mỗi arm; h = 8 trừ khi ghi khác):

| Arm | T1 | T2 | T3 | T4 | T6 | **Gộp** | So với sample0 |
|---|---|---|---|---|---|---|---|
| sample0 | 40.4 | 22.8 | 21.2 | 18.8 | 24.2 | 25.5 | — |
| **medoid** | 55.0 | 33.0 | 26.2 | 27.0 | 42.4 | **36.8** [35–39] | **+11.3** [+9.1, +13.3] |
| CCS đầy đủ | 54.4 | 28.7 | 23.4 | 28.4 | 36.6 | 34.4 | +8.9 (kém medoid −2.4) |
| KeyStone | 50.2 | 27.6 | 23.2 | 18.8 | 33.8 | 30.8 | +5.2 (kém medoid −6.0) |
| fixed_8 | 38.0 | 26.5 | 23.2 | 15.8 | 26.2 | 25.9 | — |

- Medoid hơn fixed_8 ở cả 5 track (+3.0 đến +17.0 pp; T3 có CI chứa 0).
- Medoid thắng sample0 ở 43/50 ô (track × task).
- Bỏ các episode lỗi mô phỏng thì mức chênh đổi không quá 0.4 pp.

**LIBERO-10, Track A** (1.500 episode mỗi arm):

| Arm | Clean | 2 suite nhiễu | Tất cả |
|---|---|---|---|
| medoid_h8 | 94.6 | **71.2** | **79.0** |
| sample0_h8 | 94.8 | 69.4 | 77.9 |
| medoid_h10 | 94.8 | 68.9 | 77.5 |
| Fixed-10 | **95.4** | 70.5 | 78.8 |

- Medoid so với Fixed-10 (nhiễu): **+0.7 pp [−1.6, +3.0]**; so với sample0: +1.8 [−0.6, +4.2].
- Không đạt tiêu chí manifest.

**Kiểm tra cơ chế:**

| Câu hỏi | Thí nghiệm | Kết quả | Kết luận |
|---|---|---|---|
| Có phải nhờ chuyển động mượt? | 2×2: sample0/medoid × có/không giới hạn tốc độ lệnh 1.4 cm/bước | sample0 25.5 → 25.7; medoid 36.8 → 37.1 | **Không.** Lợi ích đến từ chọn kế hoạch tốt hơn |
| Có phải nhờ chọn đúng vật? | Intention score của VLABench | 0.603 → 0.615 | Không |
| Chọn chunk có tương tác với h? | medoid ở h = 5 / 8 / 10 | 33.7 / 36.8 / 35.2 (fixed: 22.7 / 25.9 / 25.9) | Hai hiệu ứng cộng dồn: medoid +10–11 pp ở mọi h |
| Phần đổi kế hoạch còn lại do đâu? | Offline trên 7.780 decision | 54% lần đổi lớn không có mẫu nào trong 20 mẫu tiếp nối kế hoạch cũ | Policy đổi ý theo quan sát mới, ngoài tầm của việc chọn mẫu |

**Các mở rộng không thành công** (đều đã sửa theo code review rồi chạy lại):

| Mở rộng | Ý tưởng | VLABench | So với medoid |
|---|---|---|---|
| EGC | Giữ kế hoạch cũ nếu mode của nó còn ≥ 0.5 × mode lớn nhất (mode = cụm láng giềng) | 36.6% | −0.2 (không khác) |
| XDC | Đồng thuận với cả mẫu của decision trước | 36.3% | −0.4 (không khác) |
| Consensus horizon | Medoid + h = số bước các mẫu còn đồng ý (pose + gripper) | 33.4% | −3.4 (kém hơn) |
| ACS | Tái dùng noise của chunk vừa chạy để giữ mode | Dừng ở smoke test | Cùng noise không giữ được mode trên π0.5 |

### 2.6 Vị trí so với prior work

- **KeyStone** (arXiv 2605.08638): lấy medoid, phân cụm k-means 2 cụm khi có nhiều mode, thử trên π0.5 LIBERO (+1.0 pp).
- **KDPE** (CoRL 2025): chọn action có mật độ kernel cao nhất trên pose.
- Hai bài này đã làm ý tưởng chọn theo đồng thuận, nên **medoid không phải đóng góp thuật toán mới**.
- Trên VLABench, medoid (metric pose + geodesic trên 8 bước được chạy, không phân cụm) hơn bản cài KeyStone (L2 trên action chuẩn hoá cả chunk, có phân cụm) **+6.0 pp [+4.0, +8.1]**. Chưa tách được yếu tố nào quyết định.

---

## 3. So sánh AAC, DUAC và CCS (medoid)

### 3.1 Success rate

| Bộ | Fixed tốt nhất | AAC | DUAC | **Medoid (CCS)** |
|---|---|---|---|---|
| VLABench gộp (n = 2.460) | 25.9 | 25.4 | 24.3 | **36.8** |
| LIBERO-10 clean (n = 500) | **95.4** | 94.0 | 94.2 | 94.6 |
| LIBERO-10 nhiễu (n = 1.000) | 70.5 | 67.5 | 68.5 | **71.2** |
| LIBERO-Pro x (R0, n = 2.000) | 47.05 | **49.75** | 49.5 | chưa chạy |
| RoboCasa365 (n = 100) | **25** | 15 | 20 | chưa chạy |

**So sánh ghép cặp, VLABench:**
- Medoid so với AAC: **+11.4 pp**. So với DUAC: **+12.5 pp**. Đây là hiệu số tỉ lệ gộp; kiểm định ghép cặp của CCS so với AAC là +9.0 [+6.9, +11.1], và của medoid so với sample0/fixed có CI khoảng ±2 pp.
- LIBERO nhiễu: medoid so với AAC **+3.7 pp [+1.4, +6.0]** (p Holm 0.043); so với DUAC +2.7 pp [+0.1, +5.2].

### 3.2 Chi phí tính toán

Đo sạch bằng một client sau warm-up, trên RTX 5090, π0.5 JAX bf16.

| | Vanilla / Fixed (1 mẫu) | DUAC | AAC | **Medoid** |
|---|---|---|---|---|
| Gọi model, VLABench (ms) | 64.9 | 71.5 | 86.0 | 86.0 (server AAC) |
| Gọi model, LIBERO (ms) | 62.8 | 65.1 | 81.9 (KV cache chung) / 577 (encode lại mỗi mẫu) | 81.9 |
| So với 1 mẫu | 1.0× | 1.04–1.10× | 1.30–1.33× | 1.30–1.33× |
| Xử lý phía client mỗi decision (ms, đã log, gồm chuyển đổi dữ liệu) | 2.2–2.9 | 2.8 | 9.7 | 7.4 (riêng medoid đo offline 4.3; KeyStone 0.2) |
| Decision mỗi episode, VLABench | 21.3 (h = 8) / 17.0 (h = 10) / 34.3 (h = 5) | 18.6 | 19.7 | 20.5 |
| Decision mỗi episode, LIBERO | 39.6 (h = 8) / 31.6 (h = 10) | 37.4 | 43.1 | 39.3 |
| Thời gian policy mỗi action, VLABench (ms) | 8.9 (h = 8, server DUAC) / 8.1 (vanilla) | 7.6 | 9.5 | khoảng 11.3 |
| Bộ nhớ GPU đỉnh của model (MB) | khoảng 6.8k | khoảng 6.8k | khoảng 7.1k | khoảng 7.0k |
| Cần train | Không | Không | Không | Không |

Ghi chú:
- Theo paper DUAC, AAC tốn 8.5× và DUAC tốn 3.0×. Đo lại thì AAC chỉ 1.3× (nếu dùng chung KV cache) và DUAC khoảng 1.05×. Chi phí phụ thuộc mạnh vào cách cài đặt.
- Nếu server chạy không giới hạn bộ nhớ, JAX tự nở cache lên khoảng 25 GB. Con số trên là mức dùng thực.

### 3.3 Metric riêng (VLABench, h = 8 trừ khi ghi khác)

| Metric | sample0 | fixed_8 | AAC | DUAC | **Medoid** |
|---|---|---|---|---|---|
| Progress score | 0.420 | 0.416 | 0.413 | 0.408 | **0.510** |
| Intention score | 0.603 | 0.602 | 0.598 | 0.580 | 0.615 |
| Bước nhảy lệnh ở chỗ nối chunk (cm) | 3.31 | 3.33 | 3.13 | 3.34 | **1.38** |
| Gia tốc lệnh (mm/bước²) | 9.58 | 9.48 | 8.81 | 8.39 | **4.24** |
| Số lần bật/tắt gripper mỗi episode | 2.59 | 2.65 | 2.45 | 2.47 | **2.11** |
| Decision có độ lệch kế hoạch > 2 cm | 40% | — | — | — | **20%** |
| Bước tới khi thành công | 109 | 110 | 112 | 114 | 112 |
| Episode lỗi mô phỏng | 45 | 39 | 38 | 43 | **17** |

Trên LIBERO-10 (Track A):
- Medoid có ít thất bại "đạt một phần" nhất (188, so với 199–222 ở các arm khác).
- Bước nhảy ở chỗ nối chunk chỉ giảm 8% (0.147 → 0.135 đơn vị robosuite), so với 58% trên VLABench.

### 3.4 Đánh đổi

| | AAC | DUAC | Medoid |
|---|---|---|---|
| Lợi ích | Chỉ hơn h mặc định; không hơn h cố định tốt nhất | Như AAC; rẻ hơn AAC | +11 pp trên VLABench; khoảng +1 pp trên LIBERO |
| Chi phí | 20 mẫu; replan nhiều hơn (+33–39% decision trên LIBERO) | 1 mẫu + JVP; +16–19% decision | 20 mẫu; số decision bằng h cố định |
| Rủi ro | Tín hiệu không phụ thuộc state; ξ kéo dài h lúc gắp | Như AAC | Nếu đa số mẫu cùng sai thì medoid cũng sai; lợi ích phụ thuộc benchmark (chưa giải thích được vì sao) |
| Siêu tham số | N, α, luật khuỷu | τ, K, α | N, h, w_rot (đều cố định, không chỉnh trên dữ liệu chính) |
| Mức sẵn sàng triển khai | Cần server N mẫu | Cần JVP trong flow | Cần server N mẫu (như AAC); phần chọn chỉ vài ms |

**Kết luận mục 3:** cùng một server và cùng 20 mẫu, đổi câu hỏi từ "chạy bao nhiêu bước" sang "chạy mẫu nào" mang lại +11 pp trên VLABench mà gần như không tốn thêm so với AAC. Nhưng lợi ích không lặp lại trên LIBERO, và nguyên nhân chưa được giải thích (giả thuyết "làm mượt lệnh" đã bị bác bỏ).

---

## 4. Hướng cải thiện tiếp theo: literature review → research gap → hướng đề xuất

> Mục này do agent chạy model **Fable 5.2** khảo sát tài liệu (26/11/2026) rồi rút ra khoảng trống và hướng đề xuất. Bài quan trọng nhất (arXiv 2610.37196, *How Much Can Best-of-N Help a World Action Model?*, Gao et al., 27/10/2026) đã được kiểm lại trực tiếp trên arXiv: các con số trích dẫn khớp với bản HTML v1; riêng hệ số tương quan 0.01 của value head là trung bình trên 11 điều kiện của RoboCasa.

Quy ước trích dẫn kết quả của chính chúng tôi: **F1** (adaptive horizon không thắng fixed horizon tốt nhất), **F2** (medoid +11.3 pp trên VLABench, +0.7 pp n.s. trên LIBERO-10 nhiễu), **F3** (gain không phải do smoothing; selection và horizon cộng tính), **F4** (các kết quả âm: CCS, EGC, XDC, consensus-horizon, noise anchoring), **F5** (54% plan switch lớn không có sample nào tiếp nối plan cũ; plan discrepancy không dự báo sớm failure; chênh lệch VLABench–LIBERO chưa giải thích được).

### 4.1 Phạm vi và mức độ đọc

Tìm kiếm thực hiện ngày 26/11/2026, ưu tiên công trình 2025–2026. **Mức độ đọc cần nói rõ:** với hầu hết các bài mới dưới đây chúng tôi chỉ đọc *abstract* trên arXiv (đánh dấu **[A]**); riêng arXiv 2610.37196 đã đọc khoảng 2/3 đầu bản HTML (đánh dấu **[H]**); một số bài chỉ thấy qua đoạn trích kết quả tìm kiếm (đánh dấu **[S]**) và không được dùng làm căn cứ cho lập luận chính. Mọi con số trong bảng là con số do tác giả công bố trong abstract/bài, chúng tôi **chưa tái lập**. Các công trình đã biết trong danh sách của báo cáo (KeyStone, KDPE, BID, CTP, PAINT, WarmPrior, RTC, MG-Select, RoboMonkey, VGAS, VLA-ATTC, ElevateHead, Sentinel/STAC, PACT-VLA, "Restoring the Right Stream", A3, AutoHorizon) không được đọc lại ở vòng này; phần định vị ở 4.3 dựa trên mô tả đã có ở các mục trước.

### 4.2 Bảng tổng hợp các công trình mới tìm được

| Phương pháp (arXiv / venue) | Quyết định | Tín hiệu | Train? | Có yếu tố thời gian? | Bằng chứng (theo tác giả) | Hạn chế liên quan tới chúng tôi |
|---|---|---|---|---|---|---|
| **Selectable headroom cho World Action Model** (2610.37196, 10/2026) [H] | Đo trần của best-of-N; fail-gated selector | Branching từ snapshot bitwise-reproducible, K=16 candidate, J=4–17 continuation, held-out oracle | Có (logistic gate + ridge ranker trên outcome của branch) | Không | RoboCasa, LIBERO-Plus, 4 WAM: oracle 1-continuation +12.3 pp nhưng held-out chỉ ~0.9 pp; KeyStone offline −0.004 / +0.006 so với trung bình candidate; fail-gated 76.9→81.6% | Chỉ WAM, không có π0.5/VLABench; đo lệch **một lần** rồi trả quyền cho policy, không đo luật chọn áp dụng **mọi** decision |
| **ADS – Flow-support-weighted proposals** (2610.14588, đang review AAAI 2027) [A] | Sửa phân phối proposal lúc train | Flow support (Hutchinson trace) | Có (fine-tune policy) | Không | RoboCasa oracle best-of-8 57.6→63.0% | Cần train; nhưng nêu hiện tượng "proposal contraction": policy fine-tune hẹp cho sample gần trùng nhau |
| **TACO** (2512.02834) [A] | Chọn 1 chunk trong N | Pseudo-count (anti-exploration) | Có (estimator nhẹ) | Không | RoboTwin 2.0, RoboTwin, LIBERO, SimplerEnv; abstract không nêu số | Cần dữ liệu downstream; không phân tích khi nào không lợi |
| **CoVer-VLA** (2602.12281) [A] | Chọn instruction rephrase + chunk | Contrastive verifier | Có | Không | SIMPLER +22% ID / +13% OOD; PolaRiS +9% SR | Verifier ngoài; tăng đa dạng bằng rephrase chứ không bằng noise |
| **SDN – Selected Diffusion Noise** (2606.14084) [A] | Chọn chunk theo 2 tầng | Grounding filter với counterfactual negative set, rồi JerkRMS | Không | Không | +8–10% sim, +18.33% real, 3 VLA | Tầng 2 chọn theo độ mượt; F3 cho thấy mượt không phải cơ chế ở π0.5/VLABench |
| **Golden Ticket** (2603.15757) [A] | Thay noise ngẫu nhiên bằng 1 vector noise cố định | Monte-Carlo policy evaluation theo reward | Không train policy, nhưng cần search theo task | Hằng số theo episode | Cải thiện 46/51 task; tới +79 pp sim | Cần reward và episode search per-task; cho thấy noise quyết định outcome |
| **Probe trên VLA frozen** (2605.28527) [A] | Chọn prefix action | Linear probe trên feature π0.5, target Monte-Carlo outcome | Có (probe) | Không | LIBERO-Goal push-plate 26.7→44.3%; tác giả ghi rõ gain "không phổ quát" | Gain chỉ ở task có base thấp; chưa giải thích vì sao |
| **DREAMSTEER** (2607.02865) [A] | Chọn chunk từ VLA + motion primitive | Latent world model + value model | Không fine-tune VLA (dùng WM/critic có sẵn) | Không | Real, 4 benchmark: 23.75→66.25% | Cần world model; vượt ngân sách 27 GB là rủi ro |
| **QPILOTS** (2606.14801) [A] | Steering denoising | Gradient critic tại clean-action estimate | Có (critic) | Không | 6 task sim với VLA frozen | Cần critic/RL |
| **DVAC** (2606.03847) [A] | h | Variance của clean-action estimate ở các bước denoise cuối | Không | Rolling buffer chuẩn hoá ngưỡng | LIBERO π0.5 94.75→98.00%, −43% replan | Abstract không nói baseline là fixed-h nào; chưa có control random-h |
| **Cross-attention entropy** (2609.00908) [A] | h | Entropy cross-attention của action expert theo vị trí trong chunk | Không | Không | π0.5, X-VLA; RoboTwin 2.0, LIBERO; không có số trong abstract | Tín hiệu "tăng dần theo vị trí" có thể là profile cố định như h_unc của DUAC (F1) |
| **PACE** (NeurIPS 2026 poster) [A] | h | Thung lũng tốc độ trong chunk dự đoán | Không (ngưỡng từ demo) | Không | RoboTwin 2.0: 57.8→64.2% so với fixed-h mạnh nhất | Bài duy nhất nêu so với best fixed; chưa có trên LIBERO-PRO/VLABench |
| **Spatial Attention** (2607.04739) [A] | h | E‖∇_o log π(a\|o)‖² (độ nhạy theo observation), được dự báo kèm chunk | Có (head dự báo) | Dự báo tương lai | Sim + real, không có số trong abstract | Dùng độ nhạy observation để chọn h, không để chọn chunk hay chẩn đoán switch |
| **BCP** (2608.03483) [A] | continue / replan từng bước | Head RL trên VLA frozen | Có (RL online) | Có (tuần tự) | LIBERO-PRO π0.5 +6.8%; RoboTwin 89.88→93.94% | Kết quả LIBERO-PRO/π0.5 sát nhất với setup của chúng tôi; cần RL online |
| **DEHP** (2606.11408) [A] | h | Nhánh dự báo h, train bằng RL online | Có | Không | Không có số trong abstract | Như BCP |
| **PAV** (2610.21434) [A] | h qua "prefix còn cho phép hoàn thành task?" | Verifier trong không gian biểu diễn VLA, supervision từ terminal outcome | Có | Không | π0/π0.5, LIBERO, RoboTwin 2.0; không có số | Verifier outcome có thể dùng cho selection, nhưng cần nhãn |
| **IntentVLA** (2605.14712, EMNLP 2026) [A] | Giữ intent giữa các lần replan | Lịch sử visual gần → biểu diễn intent | Có (train VLA) | Có | AliasBench, SimplerEnv, LIBERO, RoboCasa; không có số | Thừa nhận switch do observation aliasing nhưng sửa bằng train lại |
| **CoRe** (2608.14822) [A] | Phục hồi sau lệch | Counterfactual tưởng tượng từ trạng thái "viable" gần nhất | Không | Có | Tới +85 pp dưới nhiễu loạn online | Nhắm nhiễu loạn ngoài, không phải switch nội sinh |
| **PDF** (2604.18107, CVPR 2026) [A] | Vote action qua các view augment + module perturbation | Uncertainty + delayed feedback | Có (module nhẹ, online) | Có (feedback trễ) | LIBERO +7.4% | Action logits (VLA rời rạc); chưa rõ áp dụng cho flow chunk |
| **FIPER** (2510.09459, NeurIPS 2025) [A] | Cảnh báo failure | RND trên embedding + action-chunk entropy, conformal | Calibrate trên rollout thành công | Cửa sổ thời gian | 5 môi trường | Chúng tôi không biết bài có khống chế độ dài episode hay không (xem F5) |
| **Hide-and-Seek** (2605.30834, NeurIPS 2026) [A] | Cảnh báo failure | Contrastive từ nhãn mức trajectory | Có | Có | LIBERO, **VLABench**, π0/π0.5 | Cùng benchmark và policy; là baseline bắt buộc nếu làm failure gate |
| DiVeR (2610.04933), SV-VLA (2604.02965), VLA-Corrector (2607.01804) [S] | verifier / trigger replan | — | — | — | Chỉ thấy đoạn trích | Không dùng làm căn cứ |

### 4.3 Định vị so với các công trình đã biết

Ba nhận xét rút ra từ bảng và từ danh sách đã biết:

1. **Selection:** mọi phương pháp chọn chunk hoặc cần verifier/critic/probe có train (RoboMonkey, VGAS, VLA-ATTC, TACO, CoVer, PAV, probe 2605.28527), hoặc dùng tiêu chí hình học trong không gian action (KeyStone, KDPE, MG-Select, SDN). Medoid của chúng tôi thuộc nhóm sau và về bản chất là KeyStone bỏ bước k-means; đóng góp thực nghiệm là nó hơn bản KeyStone tái lập (36.8% vs 30.8%, F2), không phải một ý tưởng mới.
2. **Temporal:** BID, CTP, PAINT, RTC, WarmPrior, IntentVLA đều ngầm giả định mode cũ *vẫn nằm trong support* của phân phối hiện tại (để chọn/kéo về), hoặc train lại để nó nằm trong đó.
3. **Horizon:** từ 6/2026 có ít nhất bảy tín hiệu mới cho h (DVAC, cross-attention, PACE, Spatial Attention, BCP, DEHP, PAV). Chỉ abstract của PACE nêu so với "strongest fixed-horizon baseline"; chúng tôi không tìm thấy bài nào báo cáo control random-h khớp phân phối như ở F1 (kết luận này giới hạn ở mức abstract).

### 4.4 Research gap

**G1 — Chưa có mô hình định lượng dự đoán *khi nào* consensus selection có lợi.**
Căn cứ: F2 và F5 (+11.3 pp ở base ~25%, ~0 ở base 70–95%). Gần nhất: 2610.37196 ghi nhận headroom "tập trung ở decision mà policy chunk thất bại" và nhỏ ở nơi policy đã mạnh; ADS nêu proposal contraction; probe 2605.28527 nói gain "không phổ quát". Vì sao chưa đóng: 2610.37196 là quan sát exploratory trên WAM, không đưa ra đại lượng đo trước được để dự báo gain; ADS mô tả contraction như lý do phải train lại chứ không dùng nó để dự báo; không bài nào tách ba yếu tố *tỉ lệ sample lệch mode*, *độ hại của một chunk lệch*, *số decision mỗi episode*.

**G2 — Mâu thuẫn chưa giải giữa "headroom một lần" và "luật chọn áp dụng liên tục".**
Căn cứ: F2, F3. 2610.37196 đánh giá KeyStone offline và thu được gain ≈ 0 so với trung bình candidate, rồi kết luận native sample hầu như không có headroom chọn được; chúng tôi đo closed-loop medoid +11.3 pp [+9.1, +13.3] và KeyStone +5.2 pp trên cùng 20 native sample. Hai kết quả khác policy class (WAM vs flow VLA) và khác benchmark nên chưa thể gọi là mâu thuẫn trực tiếp, nhưng khác biệt giao thức là có thật: họ lệch **một** decision rồi để policy gốc chạy tiếp; chúng tôi đổi luật ở **mọi** decision, tức là đổi cả phân phối trạng thái về sau. Chưa bài nào đo cả hai trên cùng một policy. F3 (cộng tính với h, không phải smoothing, intention score không đổi) loại bớt giải thích nhưng chưa chỉ ra cơ chế dương tính.

**G3 — Plan switch do observation trên policy frozen: chưa ai tách "nhiễu" khỏi "thông tin", và chưa ai đo switch có hại hay không.**
Căn cứ: F5 (54% switch lớn không có sample tiếp nối) và F4 (CCS, EGC, XDC, consensus-horizon, noise anchoring đều không hơn medoid). Gần nhất: IntentVLA (quy switch cho observation aliasing, sửa bằng train với lịch sử), CTP (belief tracking có train), BID/PAINT/RTC (kéo về chunk cũ), Spatial Attention (độ nhạy theo observation nhưng để chọn h), "Restoring the Right Stream" và PDF (consensus qua augmentation). Vì sao chưa đóng: nhóm kéo-về-chunk-cũ bất lực đúng ở 54% trường hợp của F5; nhóm augmentation nhắm robustness với nhiễu ngoài chứ không dùng để phân loại switch nội sinh; và quan trọng nhất, F5 cho thấy plan discrepancy không dự báo failure, nên giả định ngầm "switch là xấu" của cả dòng này **chưa được kiểm chứng nhân quả**. Chúng tôi có state restore để làm việc đó.

**G4 — Thiếu control chuẩn cho adaptive horizon.**
Căn cứ: F1 (random-h khớp phân phối hoà DUAC; profile chữ U theo vị trí; xi quyết định 74%). Gần nhất: PACE (so với fixed-h mạnh nhất), DVAC và cross-attention entropy (cùng π0.5, training-free). Vì sao chưa đóng: tín hiệu cross-attention được chính tác giả mô tả là tăng dần theo vị trí trong chunk — đúng dạng profile cố định mà F1 đã chỉ ra ở DUAC; không abstract nào báo cáo control random-h khớp phân phối hay control "profile trung bình theo vị trí". Đây là gap về phương pháp đánh giá; chúng tôi đã có sẵn harness.

**G5 (thận trọng) — Dự báo failure sớm.** F5 cho thấy AUROC 0.87 toàn episode là artifact độ dài episode, còn 5 decision đầu chỉ 0.54–0.59. Chúng tôi **không** khẳng định FIPER/SAFE/Hide-and-Seek mắc lỗi này vì chưa đọc phần đánh giá; chỉ ghi nhận rằng bất kỳ failure gate nào chúng tôi xây (Hướng C) phải báo cáo AUROC theo cửa sổ cố định và so với Hide-and-Seek trên VLABench.

### 4.5 Hướng đề xuất (xếp hạng)

#### Hướng A (ưu tiên 1) — Mô hình "ε–q–T" cho consensus selection và phép đo headroom một-lần vs liên-tục
*Nhắm G1, G2; trả lời câu hỏi VLABench vs LIBERO. Không train.*

**Giả thuyết.** Gain của medoid đến từ việc loại các chunk lệch mode hiếm nhưng gây hại không hồi phục. Gọi ε = xác suất sample 0 là outlier tại một decision, q = xác suất một chunk outlier làm hỏng episode (so với chunk medoid), T = số decision mỗi episode. Khi đó

```
SR_medoid − SR_sample0 ≈ SR_medoid · [1 − (1 − ε·q)^T]
```

LIBERO cho gain ≈ 0 vì ε nhỏ (proposal contraction ở policy đã fine-tune sát phân phối, theo ADS) và/hoặc q nhỏ (task dung sai lớn), không phải vì "base cao" tự thân.

**Phương pháp.**
1. *Offline, từ log sẵn có:* với mỗi decision tính d_i = khoảng cách pose trung bình trên prefix thực thi từ sample i tới medoid; chuẩn hoá bằng MAD trong decision; outlier nếu d_0 > median + 3·MAD. Ước lượng ε và T theo từng task trên cả VLABench và LIBERO.
2. *Branching trên VLABench:* chọn ngẫu nhiên phân tầng 300 decision state. Tại mỗi state thực thi 3 candidate {medoid, sample ở trung vị d, sample có d lớn nhất}, mỗi candidate J=4 continuation với common random numbers, dưới hai chính sách tiếp diễn {sample0, medoid}. Báo cáo held-out oracle theo cách của 2610.37196 để tránh lạc quan do chọn và chấm trên cùng rollout.
3. Ước lượng q từ bước 2, thay vào công thức, so với gain đo được theo từng task/track.

**Thí nghiệm quyết định và kill criterion.** Bước 1 gần như không tốn compute và làm trước.
- *Kill 1:* nếu ε_LIBERO ≥ 0.5·ε_VLABench thì giải thích "contraction" bị bác; chỉ còn q, và phải dựng state restore cho LIBERO (MuJoCo, cần audit bitwise replay) trước khi đi tiếp.
- *Kill 2:* nếu trên VLABench hiệu success held-out (medoid − outlier) < 3 pp với CI chứa 0, thì gain +11 pp **không** đến từ chất lượng từng chunk; mô hình ε–q–T sai và giả thuyết thay thế là hiệu ứng tích luỹ lên phân phối trạng thái (so sánh hai chính sách tiếp diễn ở bước 2 sẽ đo trực tiếp điều này).
- *Tiêu chí đạt:* công thức dự đoán gain theo task với tương quan hạng ≥ 0.6 trên VLABench và dự đoán |gain| < 2 pp trên LIBERO.

**Compute (ước lượng, chưa đo).** 300 × 3 × 4 × 2 = 7 200 rollout từng phần; nếu trung bình còn nửa episode thì tương đương khoảng 3 600 episode 20-sample, tức cỡ 1.5 lần một lượt đánh giá VLABench n=2460 mà chúng tôi đã chạy. Bộ nhớ không đổi so với cấu hình medoid hiện tại.

**Rủi ro novelty: trung bình–cao.** 2610.37196 đã có giao thức branching và khái niệm selectable headroom. Khác biệt: (i) flow VLA π0.5 thay vì WAM; (ii) so sánh trực tiếp lệch-một-lần với luật-liên-tục, thứ mà bài đó không đo; (iii) mô hình dự báo regime từ đại lượng đo offline. Nếu Kill 2 xảy ra, kết quả vẫn công bố được như một phản ví dụ có kiểm soát cho kết luận "native sample không có headroom".

#### Hướng B (ưu tiên 2) — Phân xử nhân quả plan switch và consensus biên hoá theo observation
*Nhắm G3. Không train. Kỳ vọng tiên nghiệm: có thể bị kill ngay ở bước 1, và đó là kết quả có giá trị.*

**Giả thuyết.** (B1) Một phần đáng kể trong 54% switch "không có sample tiếp nối" là do policy nhạy với biến thiên không mang thông tin task (nhiễu ảnh, lệch proprio nhỏ), không phải do trạng thái thế giới thay đổi. (B2) Các switch loại này có hại, còn switch do thông tin thật thì có lợi; gộp chung nên plan discrepancy không dự báo failure (F5) và mọi cơ chế "giữ plan cũ" đều hoà (F4).

**Phương pháp.**
1. *Phân xử outcome:* tại các decision có switch lớn trên VLABench, restore state và so hai nhánh: (a) thực thi nốt phần đuôi H−h của chunk cũ rồi mới replan; (b) thực thi medoid mới. J=4 continuation mỗi nhánh, chính sách tiếp diễn là medoid.
2. *Phân loại switch:* tại cùng state, sinh M=4 view observation tương đương về task (dịch/crop vài pixel, jitter quang học nhẹ, nhiễu proprio ở mức nhiễu cảm biến), mỗi view 5 sample (giữ nguyên ngân sách 20). Định nghĩa
```
s_nuis = 1 − (khoảng cách medoid giữa các view) / (khoảng cách medoid mới ↔ chunk cũ)
```
Switch "do nhiễu" nếu một số view cho medoid tiếp nối plan cũ; "do thông tin" nếu mọi view đều rời plan cũ.
3. *Phương pháp, chỉ khi bước 1–2 ủng hộ:* medoid trên tập gộp 4×5 sample (consensus qua cả noise lẫn nuisance của observation), cùng NFE với medoid hiện tại.

**Kill criterion.**
- *Kill 1 (quyết định):* nếu nhánh (a) không hơn nhánh (b) ít nhất 3 pp success held-out trên toàn bộ switch **và** trên nhóm 54% không có sample tiếp nối, thì switch không có hại; dừng toàn bộ dòng nghiên cứu "giữ plan", ghi nhận là kết quả âm nhất quán với F4 và F5.
- *Kill 2:* nếu dưới 20% switch được phân loại "do nhiễu", B1 bị bác: phân phối dịch chuyển vì trạng thái thật, policy frozen không sửa được bằng consensus.
- *Tiêu chí đạt cho bước 3:* medoid gộp view hơn medoid thường ≥ 2 pp trên VLABench (n=2460, paired) và không giảm trên LIBERO-PRO.

**Compute (ước lượng).** Bước 1: 200 state × 2 nhánh × 4 = 1 600 rollout từng phần, cỡ 1/3 lượt VLABench. Bước 2 chỉ là inference (200 × 20 sample), không đáng kể. Bước 3: một lượt VLABench đầy đủ; M=4 view làm tăng chi phí encode prefix VLM khoảng 4 lần nếu không chia sẻ cache, cần kiểm tra giới hạn 27 GB trước.

**Rủi ro novelty: trung bình.** Consensus qua augmentation đã có ("Restoring the Right Stream", PDF), nên bước 3 riêng lẻ không mới. Phần mới là bước 1–2: phép phân xử nhân quả bằng state restore cho switch nội sinh của policy frozen, và phân rã switch thành nhiễu/thông tin. Spatial Attention dùng độ nhạy observation nhưng cho h; IntentVLA nêu cùng hiện tượng nhưng sửa bằng train.

#### Hướng C (ưu tiên 3, có train nhẹ) — Fail-gated ranking từ nhãn branch
*Chỉ mở nếu Hướng A cho thấy còn headroom held-out sau medoid.*

**Giả thuyết.** Feature của π0.5 frozen chứa đủ thông tin outcome để xếp hạng candidate *trong cùng một decision* (2605.28527 báo ~92% pairwise ordering trên LIBERO-Goal), và chỉ cần lệch khỏi medoid tại các decision bị gate dự báo là rủi ro (thiết kế fail-gated của 2610.37196).

**Phương pháp.** Dùng outcome từ branching của Hướng A làm nhãn; train logistic gate g(feature state) và ranker tuyến tính r(feature state, chunk); thực thi argmax r nếu g vượt ngưỡng conformal, ngược lại thực thi medoid. π0.5 giữ frozen; phần train chỉ là probe tuyến tính/MLP nhỏ trên CPU/GPU trong vài phút — **có train**, lý do: F4 cho thấy các tín hiệu hình học không train đã cạn.

**Kill criterion.** (i) Điều kiện mở: oracle held-out − medoid ≥ 5 pp trong Hướng A; nếu không, bỏ. (ii) Tương quan hạng trong-decision của ranker trên task held-out ≤ 0.1 thì dừng (2610.37196 báo value head của Cosmos Policy chỉ đạt ρ ≈ 0.01 trong-decision dù AUROC mức state ~0.9 — đây đúng là kiểu thất bại cần đề phòng). (iii) Gate phải được đánh giá theo cửa sổ cố định, không theo toàn episode (F5), và so với Hide-and-Seek trên VLABench.

**Compute.** Nhãn tái dùng từ Hướng A; cần thêm khoảng 1 lượt VLABench để đánh giá closed-loop.

**Rủi ro novelty: cao.** VGAS, VLA-ATTC, TACO, DiVeR, PAV, 2605.28527 và 2610.37196 đều đã chiếm vùng này. Giá trị chỉ còn ở việc kết hợp với medoid làm mặc định và ở nhãn counterfactual thay cho nhãn mức trajectory.

#### Việc phụ chi phí thấp (không xếp hạng) — control cho adaptive horizon (G4)
Áp dụng nguyên control của F1 (random-h khớp phân phối; profile trung bình theo vị trí; best fixed-h) cho hai tín hiệu training-free cùng chạy trên π0.5 là DVAC và cross-attention entropy. Nếu cả hai cũng hoà random-h, F1 được tổng quát hoá từ "AAC/DUAC" thành nhận định về cả lớp tín hiệu nội tại; nếu một trong hai thắng, đó là ứng viên h để ghép với medoid (F3 cho thấy hai hiệu ứng cộng tính).

### 4.6 Tóm tắt thứ tự thực hiện

| Bước | Việc | Chi phí (đơn vị: 1 lượt VLABench n=2460) | Quyết định sau bước |
|---|---|---|---|
| 1 | A.1: ε, T offline trên log sẵn có | ~0 | Kill 1 của A |
| 2 | B.1: phân xử chunk cũ vs medoid mới tại switch | ~0.3 | Kill 1 của B (dừng hẳn dòng "giữ plan" nếu âm) |
| 3 | A.2–A.3: branching 300 state | ~1.5 | Kill 2 của A; điều kiện mở C |
| 4 | B.2–B.3 hoặc C, tuỳ kết quả | ~1–1.5 | — |

Các ước lượng chi phí là suy từ số rollout, chưa đo thời gian thực; cần xác nhận bằng một lần chạy thử 20 state trước khi cam kết.
