# Proposal: Consensus-Coherent Chunk Selection (CCS): chọn action chunk thay vì chọn horizon

- **Người thực hiện:** Hiếu · **Ngày:** 2026-11-21 · **Repo:** `aac-duac` (server 3)
- **Trạng thái:**
  - VLABench: đã chạy xong (2.460 episode ghép cặp mỗi arm), kể cả 3 mở rộng ACS / EGC / XDC (mục 6.3).
  - LIBERO-10 (Track A): đã chạy xong (1.500 episode ghép cặp mỗi arm).
- **Một câu:** π0.5 đã sinh sẵn 20 action chunk ở mỗi bước (như AAC làm), nhưng AAC chỉ dùng chúng để chọn số bước chạy rồi chạy chunk số 0. Nếu chạy **chunk nằm giữa đám đông (medoid)** với h cố định:
  - Trên VLABench, success tăng từ 25.9% lên **36.8%** (+10.9 pp so với fixed horizon tốt nhất, p < 10⁻²⁰).
  - Trên LIBERO-10, mức tăng chỉ +0.7 đến +1.9 pp và **không có ý nghĩa thống kê**.
  - Không cần train gì.

---

## 0. Tóm tắt

**Vấn đề.**
- Trên ba benchmark của O-LONG (LIBERO-10 có nhiễu, RoboCasa365, VLABench), AAC và DUAC (hai phương pháp chọn horizon theo độ bất định) đều **không thắng fixed horizon tốt nhất**.
- Phân tích log cho thấy hai lý do:
  - Việc chọn horizon gần như không có dư địa: nếu biết trước kết quả rồi chọn fixed tốt nhất cho từng task, mức trần chỉ cao hơn 1–4 pp.
  - Tín hiệu bất định không mang thông tin theo state: random-h với cùng phân phối h cho kết quả bằng DUAC.

**Đề xuất.**
- Giữ h cố định. Chuyển câu hỏi từ "chạy bao nhiêu bước" sang "**chạy chunk nào trong N chunk**".
- CCS chọn chunk bằng ba tiêu chí, đều không cần train:
  1. đồng thuận về đóng/mở gripper,
  2. gần các chunk khác nhất (medoid),
  3. liền mạch với phần chưa chạy của chunk trước.

**Kết quả trên VLABench** (π0.5 `pi05-primitive-10task`, 5 track × 10 task × 50 episode, ghép cặp):

| Arm | Success | So với fixed tốt nhất (fixed_8) |
|---|---|---|
| **medoid_h8** | **36.8%** | **+10.9 pp** [+8.8, +13.1] |
| ccs_h8 | 34.4% | +8.5 pp [+6.4, +10.5] |
| sample0_h8: cùng 20 mẫu, chạy mẫu 0 (đối chứng) | 25.5% | −0.4 pp (không khác) |
| fixed_8 / fixed_10 / AAC / DUAC | 25.9 / 25.9 / 25.4 / 24.3% | — |

- Lợi ích có ở cả 5 track và ở 43/50 ô (track × task).
- Kết quả không đổi khi bỏ các episode lỗi mô phỏng.
- Cơ chế đo được: bước nhảy của lệnh điều khiển ở chỗ nối chunk giảm **2.4 lần** (3.31 → 1.38 cm), gia tốc giảm một nửa, ít bật/tắt gripper hơn. Intention score (chọn đúng vật hay không) gần như không đổi.
- Kết luận: lợi ích đến từ **thực thi nhất quán**, không phải từ hiểu lệnh tốt hơn.

**Kết quả trên LIBERO-10** (Track A, 1.500 episode mỗi arm):
- medoid_h8 có điểm ước lượng cao nhất trong 9 arm: 71.2% trên 2 suite nhiễu gộp, so với 70.5% của Fixed-10 và 69.4% của sample 0.
- Nhưng mức chênh **nằm trong nhiễu**: so với Fixed-10 là +0.7 pp [−1.6, +3.0]; so với sample 0 là +1.8 pp [−0.6, +4.2].
- Không đạt tiêu chí manifest. medoid ở h = 10 không giúp gì.
- Bước nhảy ở chỗ nối chunk chỉ giảm 8%, so với 58% trên VLABench.
- Như vậy lợi ích **phụ thuộc vào benchmark**. Giả thuyết hiện tại: lợi ích lớn khi policy gửi pose đích tuyệt đối (VLABench) và nhỏ khi gửi delta qua bộ điều khiển OSC vốn đã làm mượt (LIBERO). Giả thuyết này chưa được kiểm riêng.

**Điều chưa biết.**
- Vì sao lợi ích khác nhau giữa hai benchmark (mục 9, việc #1).
- Medoid thắng CCS đầy đủ trên VLABench. Chưa có benchmark thứ hai chạy CCS đầy đủ để xác nhận.

---

## 1. Bối cảnh và vấn đề

### 1.1 Action chunking và adaptive horizon

- VLA dạng flow-matching như π0.5 dự đoán một **chunk** gồm H action (H = 10 cho checkpoint LIBERO và VLABench).
- Robot chạy h ≤ H action rồi mới gọi lại policy:
  - h lớn: ít gọi model, chuyển động nhất quán, nhưng phản ứng chậm.
  - h nhỏ: phản ứng nhanh, nhưng dễ **nhảy mode** giữa các chunk, gây giật.
- **AAC** (Liang et al., CVPR 2026):
  - Lấy N = 20 chunk từ cùng một quan sát, tính entropy theo từng bước.
  - Chọn h tại "điểm khuỷu" của entropy trung bình tiền tố, với cận dưới chuyển động ξ.
  - **Robot chạy chunk số 0.**
- **DUAC** (Ben-Ami et al., NeurIPS 2026 workshop): đo độ bất định từ divergence của velocity field trên một chunk, cùng luật chọn h, cùng ξ.

### 1.2 Kết quả O-LONG: adaptive horizon không thắng fixed tốt nhất

| Benchmark | Fixed tốt nhất | AAC | DUAC | Ghi chú |
|---|---|---|---|---|
| LIBERO-10, 2 suite nhiễu (n = 1.000) | Fixed-10 70.5% | 67.5% | 68.5% | [REPORT_trackA.md](REPORT_trackA.md) |
| RoboCasa365 composite (n = 100) | Fixed-50 25% | 15% | 20% | `results/robocasa/main/REPORT.md` |
| VLABench, 5 track (n = 2.460) | fixed_8 / fixed_10 25.9% | 25.4% | 24.3% | `results/vlabench/full/REPORT.md` |

Các mức tăng trong paper AAC và DUAC (+0.5 đến +2.5 pp) được tính so với **h mặc định hoặc Fixed-10**. Khi so với fixed tốt nhất, mức tăng biến mất. Trên VLABench, AAC hơn vanilla (h = 5) đúng +2.7 pp, khớp với kiểu mức tăng trong paper, nhưng vẫn thua fixed tốt nhất 0.5 pp.

### 1.3 Chẩn đoán: vì sao chọn horizon không hiệu quả

Nguồn: [track_a/DIAGNOSIS.md](track_a/DIAGNOSIS.md), REPORT VLABench, ablation ξ và random-h.

1. **Dư địa nhỏ.** Chọn fixed tốt nhất riêng cho từng task sau khi biết kết quả chỉ hơn fixed tốt nhất chung 1–4 pp. Với 50 episode mỗi task, CI khoảng ±4 pp.
2. **h_unc là nhiễu.**
   - Profile bất định có hình chữ U cố định, nên quy tắc "bước nhảy lớn nhất" gần như luôn chọn h = 9 hoặc h = 2.
   - Tự tương quan giữa hai decision liên tiếp chỉ khoảng 0.1.
   - random-h với cùng phân phối h cho kết quả bằng DUAC (−0.6 pp [−2.7, +1.5]).
3. **ξ quyết định h ở khoảng 74% decision**, và đẩy h dài ra đúng lúc gắp vật.
4. **h ngắn có hại.** Bỏ ξ khiến AAC chạy h̄ ≈ 2.9 và thua 9 pp, vì replan dày làm robot đổi mode liên tục.

Hệ quả: nếu muốn tăng success lúc inference mà không train, phải tác động vào **nội dung action được chạy**, không phải vào **số lượng** action.

---

## 2. Câu hỏi nghiên cứu và giả thuyết

**RQ.** Với cùng N mẫu mà AAC đã sinh, chọn **mẫu nào** để chạy (h cố định) có tăng success so với (a) chạy một mẫu bất kỳ và (b) fixed horizon tốt nhất không?

| Giả thuyết | Nội dung | Cách kiểm |
|---|---|---|
| H1 | Chọn chunk theo đồng thuận > chạy mẫu bất kỳ, cùng h, cùng N mẫu | `medoid_h8`, `ccs_h8` so với `sample0_h8` |
| H2 | Chọn chunk > fixed horizon tốt nhất | So với fixed_8 và fixed_10 |
| H3 | Lợi ích đến từ việc giảm nhảy mode ở chỗ nối chunk, không từ hiểu lệnh | Bước nhảy ở chỗ nối, gia tốc, số lần bật/tắt gripper, intention score |
| H4 | Lợi ích lặp lại trên benchmark khác (LIBERO-10, action dạng delta) | Track A select stage. **Kết quả: không được ủng hộ ở mức có ý nghĩa** (mục 6.2) |

---

## 3. Phương pháp

### 3.1 Ký hiệu

- Ở mỗi decision, server trả về N chunk a⁽¹⁾…a⁽ᴺ⁾ ∈ ℝ^{H×7}, cùng quan sát, khác noise ban đầu của flow. Ở đây N = 20, H = 10.
- h là số action sẽ chạy (cố định). Mọi khoảng cách chỉ tính trên **h action đầu**, vì đó là phần thực sự được chạy.
- Chunk trước là c_prev, đã chạy h_prev action đầu. Phần chưa chạy của nó, c_prev[h_prev:], trùng thời gian với a[0 : H − h_prev] của chunk mới.

**Khoảng cách giữa hai chunk** trên t = 1..h:

  d(a, b) = mean_t ( ‖p_t(a) − p_t(b)‖₂ + w_rot · θ_t(a, b) )

trong đó p là vị trí end-effector (m), θ là góc quay giữa hai hướng (rad), w_rot = 0.05 m/rad (1 rad tương đương 5 cm).

Hai không gian action:

| Không gian | Benchmark | Cách đưa về quỹ đạo | Gripper "mở" |
|---|---|---|---|
| `absolute` | VLABench: [vị trí đích, euler xyz đích, gripper] | Dùng trực tiếp; góc là geodesic giữa hai hướng euler, an toàn khi góc quấn ±π | g ≥ 0.1 |
| `libero_delta` | LIBERO OSC: [Δpos, Δaxis-angle, gripper], đơn vị robosuite (1.0 = 0.05 m / 0.5 rad) | Cộng dồn từ pose hiện tại: p_t = 0.05·Σ Δpos, r_t = 0.5·Σ Δrot; tail của chunk trước cũng cộng dồn từ cùng gốc | g < 0 |

### 3.2 Ba tiêu chí

1. **Bỏ phiếu gripper.**
   - Lấy g\*_t = trạng thái gripper chiếm đa số ở bước t.
   - Giữ tập C = {j : Hamming(g⁽ʲ⁾[:h], g\*) ≤ 1}. Nếu C rỗng, giữ các mẫu có Hamming nhỏ nhất.
   - Mục đích: loại các phương án lẻ như "đóng gripper sớm 3 bước". Trên VLABench, 17.2% decision có ít nhất một mẫu bị loại.
2. **Đồng thuận (medoid).** D_j = (1/(|C|−1)) · Σ_{k∈C, k≠j} d(a⁽ʲ⁾, a⁽ᵏ⁾). D nhỏ nghĩa là mẫu nằm ở vùng mật độ cao của phân phối action.
3. **Liền mạch.**
   - B_j = d(a⁽ʲ⁾[:L], c_prev[h_prev : h_prev + L]), với L = min(h, H − h_prev). Ở decision đầu tiên thì B = 0.
   - Mục đích: tránh chuyển sang một mode khác ở chỗ nối chunk.

**Chọn:** j\* = argmin_{j∈C} [ z(D_j) + λ·z(B_j) ], với z là chuẩn hoá (trừ trung bình, chia độ lệch chuẩn) trong C. Khi hoà điểm, chọn chỉ số nhỏ nhất.

### 3.3 Thuật toán

```
mỗi decision:
    A = policy.sample(obs, N=20)                     # 1 lần encode prefix ảnh + lệnh, 20 noise
    j* = select_chunk(A, h, c_prev, h_prev, cfg)     # ~5 ms trên CPU
    chạy A[j*][:h]; c_prev, h_prev = A[j*], h
```

### 3.4 Các biến thể (cùng code, khác cấu hình)

| Arm | Bỏ phiếu gripper | Đồng thuận | Liền mạch (λ) | Vai trò |
|---|---|---|---|---|
| `ccs_hK` | có (tol = 1) | có | 1 | Phương pháp đầy đủ |
| `medoid_hK` | không | có | 0 | Ablation: chỉ đồng thuận |
| `egc_hK`, `xdc_hK` | — | — | — | Hai mở rộng nhắm vào việc đổi kế hoạch giữa các chunk (mục 6.3) |
| `sample0_hK` | — | — | — | **Đối chứng:** chạy mẫu 0, tức chunk mà AAC chạy |

### 3.5 Chi phí

| | Gọi model (ms, đo sạch, VLABench) | Ghi chú |
|---|---|---|
| vanilla (1 mẫu) | 64.9 | |
| DUAC server (cũng phục vụ Fixed-8/10) | 71.5 | |
| AAC server (20 mẫu, chung KV cache của prefix) | 86.0 | Cũng là server của CCS |
| Chọn chunk (CPU) | khoảng 4.8 mỗi decision | O(N²·h) phép tính khoảng cách |

- Với h = 8, CCS tốn khoảng 11.3 ms policy cho mỗi action, so với 8.9 ms của Fixed-8 trên DUAC server.
- Nếu Fixed-8 chạy trên vanilla server thì tốn khoảng 8.1 ms mỗi action. Như vậy CCS chậm hơn khoảng 25–40%.
- Không cần train, không đổi checkpoint.

### 3.6 Triển khai

| File | Nội dung |
|---|---|
| `flowbased_aac/flowbased_aac/ccs.py` | `select_chunk`, `VARIANTS`, self-test (`python -m flowbased_aac.ccs`) |
| `flowbased_aac/flowbased_aac/vlabench/eval_vlabench.py` | `--select {sample0,ccs,medoid,egc,xdc} --fixed-horizon K` (server mode aac) |
| `flowbased_aac/flowbased_aac/eval_libero.py`, `flowbased_aac/duac.py` | Như trên cho LIBERO / LIBERO-PRO (`eval-liberopro --select ...`) |
| `tools/run_vlabench_full.sh` | Arm `<select>_h<K>`, ví dụ `ARMS="ccs_h8 sample0_h8 medoid_h8"` |
| `tools/schedule_track_a.py` | `--stage select --arms medoid_h8,sample0_h8,medoid_h10` |
| `analysis/vlabench_ccs_report.py` | Report VLABench → `REPORT_CCS.md` |
| `analysis/track_a_select_report.py` | Report Track A nhiều mặt → `outputs/A/analysis/SELECT_REPORT.md` |

---

## 4. Vì sao kỳ vọng hiệu quả

1. **Các mẫu thật sự khác nhau.** Đo offline trên 4.868 decision của AAC trên VLABench:

   | Đại lượng | Trung vị (p90) |
   |---|---|
   | Khoảng cách xa nhất giữa 2 mẫu ở cuối chunk | 6.9 cm (14.6 cm) |
   | Quãng đường cả chunk đi được | 7.7 cm |
   | Tỉ lệ decision có ≥ 20% mẫu bất đồng về gripper | 15.8% |

2. **Chạy mẫu 0 là chọn ngẫu nhiên.** Kiểm offline (G0): medoid khác mẫu 0 ở 94% decision. Khi khác, lệch trung vị 1.8 cm (p90 khoảng 10 cm), và ở 8.7% decision quyết định gripper khác hẳn.
3. **Chọn mẫu ngẫu nhiên ở mỗi lần replan là nguồn nhảy mode.**
   - Hai chunk liên tiếp có thể thuộc hai mode khác nhau (ví dụ vòng trái rồi vòng phải).
   - Medoid nằm ở vùng mật độ cao nhất, nên giữa các decision liên tiếp nó ổn định hơn một mẫu bất kỳ.
   - Đây là cùng nguyên lý với majority vote hoặc minimum Bayes risk decoding trong mô hình ngôn ngữ.
4. **Không cần tín hiệu "báo trước thất bại".** AAC/DUAC cần độ bất định dự báo được *khi nào* nên replan, và tín hiệu đó không đạt. CCS chỉ so các mẫu với nhau tại cùng một state, nên không phụ thuộc vào việc tín hiệu có dự báo được thất bại hay không.

---

## 5. Thiết kế thực nghiệm

### 5.1 Benchmark và arm

| | VLABench | LIBERO-10 (Track A) |
|---|---|---|
| Policy | π0.5 `VLABench/pi05-primitive-10task` (H = 10, action tuyệt đối) | π0.5 `pi05_libero` (H = 10, action delta) |
| Episode | T1/T2/T3/T4/T6 × 10 task × 50 (T2 insert_flower: 10) = 2.460 | clean, object, env × 10 task × 50 = 1.500 |
| Arm mới | `ccs_h8`, `medoid_h8`, `sample0_h8` | `medoid_h8`, `sample0_h8`, `medoid_h10` |
| Arm cũ ghép cặp | vanilla (h = 5), fixed_8, fixed_10, AAC, DUAC | Fixed-5/8/10, AAC, DUAC, Random-h |
| Server | 1 AAC server N = 20 | 1 AAC server N = 20, giới hạn VRAM (mục 5.4) |

- Mọi arm cùng episode config và seed, nên ghép cặp được theo (track/suite, task, episode).
- `sample0_h8` chạy cùng server, cùng lúc, cùng h với arm chọn chunk. Nhờ đó tách được tác dụng của việc chọn khỏi tác dụng của h và của server.

### 5.2 Các mặt so sánh

1. **Success:** theo track/suite và gộp, kèm Wilson CI.
2. **Kiểm định ghép cặp:**
   - McNemar exact.
   - Bootstrap phân tầng theo (track × task); 5.000 lần.
   - Holm cho hai nhóm so sánh: nhóm chính và nhóm phụ.
3. **Theo task:** lợi ích đến từ đâu; số ô (track × task) thắng/thua.
4. **Sub-goal và kiểu thất bại** (LIBERO): không đạt / một phần / đủ predicate nhưng không cùng lúc / thành công. Progress và intention score (VLABench).
5. **Tốc độ và chi phí:** số bước tới khi thành công, số decision mỗi episode, ms mỗi lần gọi model, thời gian policy mỗi episode.
6. **Chất lượng chuyển động:**
   - Bước nhảy của lệnh ở chỗ nối chunk so với thay đổi bên trong chunk.
   - Gia tốc hoặc độ giật.
   - Số lần bật/tắt gripper.
   - Tổng quãng đường lệnh.
7. **Selector đã làm gì:** tỉ lệ chọn khác mẫu 0, tỉ lệ bị loại bởi bỏ phiếu gripper.

### 5.3 Tiêu chí đặt trước

- **VLABench:** arm chọn chunk "WORK" khi CI 95% của chênh lệch so với **cả** `sample0_h8` **và** fixed tốt nhất nằm hoàn toàn trên 0, và p Holm < 0.05.
- **Track A:** dùng nguyên tiêu chí mục 5 đã khoá trong `manifest.json`:
  - Trên 2 suite nhiễu gộp, hơn mọi Fixed-5/8/10 ≥ 3 pp, p Holm < 0.05.
  - Vẫn hơn ≥ 1 pp khi bỏ task có lợi nhất.
  - Trên clean không kém fixed tốt nhất quá 2 pp.
- **Không chỉnh tham số** (N, h, w_rot, λ, tol) trên dữ liệu chạy chính.
  - h = 8 lấy từ fixed tốt nhất trên VLABench.
  - `medoid_h10` trên LIBERO được thêm trước khi có kết quả, vì Fixed-10 là fixed tốt nhất ở benchmark đó.

### 5.4 Ràng buộc tài nguyên

- 1 GPU RTX 5090 32 GB. Tổng VRAM giữ **≤ 27 GB**.
- Server AAC N = 20 tự nở cache lên khoảng 25 GB nếu không giới hạn. Vì vậy chạy với `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55`, server khoảng 18 GB.
- 12 client LIBERO render EGL thêm khoảng 6.6 GB, tổng khoảng 24.6 GB.
- Client khởi động lệch nhau 8 s, sau 120 s warm-up, để tránh timeout handshake khi server đang compile.

---

## 6. Kết quả

### 6.1 VLABench (xong 2026-11-21 02:32)

Report đầy đủ: [results/vlabench/full/REPORT_CCS.md](../results/vlabench/full/REPORT_CCS.md).

**Success theo track (%):**

| Track | ccs_h8 | medoid_h8 | sample0_h8 | vanilla | fixed_8 | fixed_10 | AAC | DUAC |
|---|---|---|---|---|---|---|---|---|
| T1 in-distribution | 54.4 | **55.0** | 40.4 | 42.2 | 38.0 | 42.2 | 41.8 | 38.4 |
| T2 cross-category | 28.7 | **33.0** | 22.8 | 20.7 | 26.5 | 21.5 | 21.7 | 21.7 |
| T3 common sense | 23.4 | **26.2** | 21.2 | 14.8 | 23.2 | 20.4 | 20.8 | 18.4 |
| T4 semantic instruction | **28.4** | 27.0 | 18.8 | 14.6 | 15.8 | 17.4 | 16.6 | 15.6 |
| T6 unseen texture | 36.6 | **42.4** | 24.2 | 21.2 | 26.2 | 27.6 | 25.8 | 27.4 |
| **Gộp** | 34.4 | **36.8** | 25.5 | 22.7 | 25.9 | 25.9 | 25.4 | 24.3 |

**So sánh ghép cặp (gộp, n = 2.460):**

| So sánh | Chênh (pp) | CI 95% | Chỉ A thắng / chỉ B thắng | p Holm |
|---|---|---|---|---|
| ccs_h8 vs sample0_h8 | +8.9 | [+6.7, +10.9] | 477 / 259 | < 0.001 |
| ccs_h8 vs fixed_8 | +8.5 | [+6.4, +10.5] | 460 / 252 | < 0.001 |
| medoid_h8 vs sample0_h8 | +11.3 | [+9.1, +13.3] | 522 / 245 | < 0.001 |
| medoid_h8 vs fixed_8 (bỏ episode lỗi) | +10.9 | [+8.8, +13.1] | — | < 10⁻²⁰ |
| ccs_h8 vs medoid_h8 | −2.4 | [−4.4, −0.4] | 292 / 351 | 0.044 |
| sample0_h8 vs fixed_8 | −0.4 | [−2.4, +1.7] | 341 / 351 | 0.73 |

**Độ vững:**
- Medoid hơn fixed_8 ở T1 +17.0, T2 +6.5, T3 +3.0 (CI chứa 0), T4 +11.2, T6 +16.2 pp.
- Medoid hơn sample 0 ở 43/50 ô (track × task), kém ở 5 ô.
- Bỏ 141 episode có lỗi mô phỏng ở bất kỳ arm nào thì các mức chênh thay đổi không quá 0.4 pp.

**Chất lượng chuyển động và hiểu lệnh** (kiểm H3):

| Arm | Bước nhảy ở chỗ nối (cm) | Thay đổi trong chunk (cm) | Nối / trong | Gia tốc (mm/step²) | Bật/tắt gripper | Intention |
|---|---|---|---|---|---|---|
| medoid_h8 | **1.38** | 0.71 | **1.95** | **4.24** | **2.11** | 0.615 |
| ccs_h8 | 1.30 | 0.71 | 1.82 | 4.21 | 2.10 | 0.615 |
| sample0_h8 | 3.31 | 0.79 | 4.18 | 9.58 | 2.59 | 0.603 |
| fixed_8 | 3.33 | 0.78 | 4.26 | 9.48 | 2.65 | 0.602 |
| AAC | 3.13 | 0.76 | 4.12 | 8.81 | 2.45 | 0.598 |

- Chọn chunk cắt bước nhảy ở chỗ nối chunk từ khoảng 4 lần xuống khoảng 2 lần mức thay đổi bên trong chunk.
- Intention gần như không đổi.
- H3 được ủng hộ: lợi ích đến từ việc thực thi nhất quán, không phải từ việc chọn đúng vật.

**Tốc độ:**
- Số decision mỗi episode như nhau (20.7 so với 21.3 của fixed_8, cùng h = 8).
- Số bước tới khi thành công tương đương (111 so với 110).

**Selector đã làm gì:**
- Chunk được chạy khác mẫu 0 ở 95% decision.
- Bỏ phiếu gripper loại ít nhất một mẫu ở 17.2% decision.
- Overlap 2 bước với chunk trước ở 95% decision.

### 6.2 LIBERO-10, Track A (xong 2026-11-21 13:24)

Report đầy đủ: [results/track_a/SELECT_REPORT.md](../results/track_a/SELECT_REPORT.md). Mỗi arm có 1.500 episode ghép cặp, không job nào lỗi.

**Success (%):**

| Arm | Clean | Object | Env | **2 suite nhiễu gộp** | Tất cả |
|---|---|---|---|---|---|
| **medoid_h8** | 94.6 | **69.8** | 72.6 | **71.2** | **79.0** |
| sample0_h8 | 94.8 | 67.0 | 71.8 | 69.4 | 77.9 |
| medoid_h10 | 94.8 | 67.8 | 70.0 | 68.9 | 77.5 |
| Fixed-10 | **95.4** | 68.2 | 72.8 | 70.5 | 78.8 |
| Fixed-8 | 94.6 | 65.0 | **73.6** | 69.3 | 77.7 |
| Fixed-5 | 93.8 | 67.6 | 69.4 | 68.5 | 76.9 |
| Random-h | 95.0 | 69.0 | 70.2 | 69.6 | 78.1 |
| DUAC | 94.2 | 67.8 | 69.2 | 68.5 | 77.1 |
| AAC | 94.0 | 67.0 | 68.0 | 67.5 | 76.3 |

**So sánh ghép cặp trên 2 suite nhiễu (n = 1.000):**

| So sánh | Chênh (pp) | CI 95% | Chỉ A thắng / chỉ B thắng | p Holm |
|---|---|---|---|---|
| medoid_h8 vs Fixed-10 | +0.7 | [−1.6, +3.0] | 75 / 68 | 1 |
| medoid_h8 vs Fixed-8 | +1.9 | [−0.3, +4.2] | 79 / 60 | 1 |
| medoid_h8 vs sample0_h8 | +1.8 | [−0.6, +4.2] | 85 / 67 | 1 |
| medoid_h8 vs AAC | **+3.7** | [+1.4, +6.0] | 92 / 55 | **0.043** |
| medoid_h10 vs Fixed-10 | −1.6 | [−3.9, +0.7] | 62 / 78 | 1 |

**Tiêu chí manifest Track A: không đạt.**

| Arm | Hơn Fixed-5/8/10 ít nhất (nhiễu) | Bỏ task có lợi nhất | Clean so với fixed tốt nhất | Kết luận |
|---|---|---|---|---|
| medoid_h8 | +0.7 (cần ≥ 3, p Holm < 0.05) | +0.4 (object/9) | −0.8 | Không đạt |
| medoid_h10 | −1.6 | −2.3 | −0.6 | Không đạt |

**Các mặt khác:**
- **Sub-goal và kiểu thất bại.** medoid_h8 có nhiều episode thành công nhất (1.185/1.500, so với 1.182 của Fixed-10). Nó cũng có ít thất bại "đạt một phần" nhất (188, so với 199–222 ở các arm khác) và ít "đủ predicate nhưng không cùng lúc" (7). Chênh lệch nhỏ.
- **Tốc độ.** Số bước tới khi thành công như nhau (253 so với 252, ghép cặp với Fixed-10). Số decision mỗi episode bằng Fixed-8 (39.3 so với 39.6).
- **Chất lượng chuyển động** (đơn vị robosuite):

  | Arm | Bước nhảy ở chỗ nối chunk | Nối / trong | Độ giật |
  |---|---|---|---|
  | medoid_h8 | 0.135 | 2.55 | 0.050 |
  | sample0_h8 | 0.147 | 2.78 | 0.053 |

  Bước nhảy giảm 8%, so với 58% trên VLABench.
- **Selector đã làm gì.** Chunk được chạy khác mẫu 0 ở 95% decision, giống VLABench.
- **Độ đa dạng của mẫu.** Đo trên 5.996 decision: khoảng cách xa nhất giữa 2 mẫu là 5.7 cm (VLABench 4.7 cm), độ phân tán 1.5 cm (VLABench 1.0 cm). Tỉ lệ ≥ 20% mẫu bất đồng về gripper là 16.6% (VLABench 15.7%). Như vậy các mẫu **không** kém đa dạng hơn VLABench.
- **Chi phí.** ms mỗi lần gọi model được ghi dưới mức tải khác nhau ở mỗi run (số client đồng thời khác nhau), nên không so được giữa các run. Đo sạch xem mục 3.5.

**Diễn giải.**
- H4 (lặp lại trên benchmark thứ hai) **không được ủng hộ** ở mức có ý nghĩa thống kê.
- medoid_h8 có điểm ước lượng cao nhất trong 9 arm, cả trên suite nhiễu gộp lẫn tổng. Nhưng mức tăng (+0.7 đến +1.9 pp) nằm trong nhiễu.
- medoid ở h = 10 không giúp gì.
- Lời giải thích hợp lý nhất (giả thuyết, chưa kiểm riêng) nằm ở **giao diện điều khiển**:
  - VLABench gửi **pose đích tuyệt đối** qua IK. Khi chunk mới đổi mode, đích nhảy ngay. Tỉ lệ bước nhảy ở chỗ nối so với bên trong chunk là khoảng 4.2.
  - LIBERO gửi **delta** qua bộ điều khiển OSC. Đổi mode chỉ làm đổi vận tốc lệnh, và bộ điều khiển làm mượt. Tỉ lệ đó vốn đã khoảng 2.8.
  - Vì vậy cơ chế "giảm nhảy mode" có ít việc để làm trên LIBERO.
- Ngoài ra, LIBERO có success cơ sở cao hơn nhiều (70–95% so với khoảng 25%), nên còn ít thất bại do thực thi để cứu.

### 6.3 Mở rộng nhắm vào việc đổi kế hoạch giữa các chunk (VLABench, 2026-11-22)

**Động cơ.** Gọi *độ lệch kế hoạch* là khoảng cách giữa 2 bước đầu của chunk mới và 2 bước chưa chạy của chunk trước, tại cùng thời điểm.

| Arm | Decision có độ lệch > 2 cm | AUROC dự đoán thất bại của episode | Success |
|---|---|---|---|
| sample0_h8 | 40% | 0.87 | 26% |
| medoid_h8 | 20% | 0.72 | 35% |
| ccs_h8 | 15% | 0.68 | 35% |

Độ lệch này dự đoán thất bại rất tốt. Medoid giảm nó một nửa. Câu hỏi đặt ra: xử lý phần còn lại có tăng thêm success không?

**Ba ý tưởng đã thử:**

| Ý tưởng | Cơ chế | Kết quả |
|---|---|---|
| **ACS** (Anchored Commit-or-Switch) | Thêm 4 mẫu sinh từ noise của chunk vừa chạy, dịch theo thời gian, làm ứng viên "tiếp tục kế hoạch". Giữ kế hoạch nếu đủ mẫu mới ủng hộ | **Giả định sai, dừng ở smoke test; code đã gỡ** (còn trong lịch sử git, commit `5199838`). Trên 126 decision, mẫu neo lệch kế hoạch cũ 0.85 cm, mẫu mới trung vị 0.80 cm; mẫu neo tốt hơn chỉ 52% số lần. Với π0.5, cùng noise không giữ được mode dưới quan sát mới (khớp hạn chế mà PAINT nêu: action expert trộn mạnh giữa các vị trí trong chunk) |
| **EGC** (Evidence-Gated Commitment) | Đếm số mẫu mới khớp với phần chưa chạy của kế hoạch cũ (bán kính = trung vị khoảng cách giữa các mẫu). Nếu ≥ 0.5 × mode đông nhất thì giữ, chạy medoid của nhóm đó; nếu không thì chạy medoid toàn cục | 37.2%, so với medoid **+0.4 pp [−1.5, +2.3]**, không khác. Giữ kế hoạch ở 31% decision |
| **XDC** (Cross-Decision Consensus) | Chọn mẫu ở trung tâm của cả 20 mẫu hiện tại và 20 mẫu của decision trước (trên đoạn chồng nhau), λ = 1 | 36.3%, so với medoid **−0.4 pp [−2.4, +1.5]**, không khác |

**Chẩn đoán vì sao không hơn được.** Offline trên 7.780 decision của medoid_h8:
- Ở các decision còn lệch > 2 cm, **54% không có mẫu nào trong 20 mẫu khớp kế hoạch cũ** (mẫu gần nhất vẫn lệch > 2 cm). Cả phân phối của policy đã dịch theo quan sát mới.
- Chỉ khoảng 21% trong số đó (tức khoảng 4% tổng decision) có sẵn một mẫu tiếp nối để chọn.

**Kết luận.**
- Medoid đã lấy gần hết phần lợi đến từ *nhiễu lấy mẫu*.
- Phần đổi kế hoạch còn lại chủ yếu do *policy đổi ý theo quan sát*, nằm ngoài tầm của mọi phương pháp chỉ chọn trong N mẫu của cùng một quan sát.
- Bước tiếp theo, nếu muốn vượt medoid, phải tác động vào cách policy phản ứng với quan sát (mục 9).
- Cả EGC và XDC đều giảm thêm bước nhảy ở chỗ nối chunk (1.30 và 1.27 cm, so với 1.38 cm của medoid) mà success không tăng. Điều này cho thấy, vượt quá mức của medoid, độ mượt không còn là nút thắt.

---

## 7. So sánh với các phương pháp liên quan

| | AAC | DUAC | BID | **CCS / medoid** |
|---|---|---|---|---|
| Quyết định | h | h | chunk nào (h cố định, thường nhỏ) | chunk nào (h cố định) |
| Tín hiệu | Entropy của N mẫu | Divergence của velocity field (1 mẫu) | Backward coherence + forward contrast (cần thêm một policy "yếu") | Đồng thuận giữa N mẫu + liền mạch |
| Cần train | Không | Không | Không, nhưng cần policy yếu | Không |
| Mẫu được chạy | Mẫu 0 | Mẫu duy nhất | Mẫu tốt nhất theo điểm | Medoid |
| Chi phí mỗi lần gọi | N mẫu | 1 mẫu + JVP | N mẫu × 2 policy | N mẫu |
| VLABench (gộp) | 25.4% | 24.3% | chưa đo | **36.8%** |
| LIBERO-10, 2 suite nhiễu gộp | 67.5% | 68.5% | chưa đo | 71.2% (Fixed-10: 70.5%) |

**Về novelty:**
- Best-of-N theo đồng thuận và BID-style coherence **không phải ý tưởng mới**.
- Phần đóng góp có thể bảo vệ được là:
  1. So sánh trực tiếp "chọn h" với "chọn chunk" trên **cùng N mẫu, cùng chi phí, cùng checkpoint**, có đối chứng sample 0. Ở đây chọn h cho 0 pp, chọn chunk cho +11 pp.
  2. Cơ chế được đo: giảm bước nhảy ở chỗ nối chunk, intention không đổi.
  3. Kiểm trên nhiều benchmark với kiểm định ghép cặp và tiêu chí đặt trước.
- Các công trình best-of-N cho VLA gần đây (ví dụ VLA-ATTC, KeyStone) chưa được đọc toàn văn.

**Các công trình liên quan đã đọc (2026-11-22, qua arXiv HTML):**

| Công trình | Quyết định | Cần train? | Có xét tính nhất quán qua thời gian? | Khác với đề xuất này |
|---|---|---|---|---|
| CTP, Conditional Trajectory Peaks (arXiv 2610.06104) | Mode nào trong K mode dự đoán; truyền belief qua các lần replan, có cổng nhả mode | **Có**: head riêng gắn lên π0.5 | Có (belief và cổng nhả) | Gọi tên đúng vấn đề "cross-chunk mode stitching". EGC là bản không cần train của ý tưởng giữ-hay-đổi, nhưng không hơn medoid trên VLABench |
| PAINT (arXiv 2606.19774) | Noise ban đầu cho chunk mới, bằng cách đảo ngược phần đã chạy | Không (khoảng 3 lần số bước flow) | Có, cho một chunk | Không có mode hay nhiều mẫu. Kết quả ACS ở đây cho thấy noise dịch theo thời gian không giữ được mode trên π0.5 |
| WarmPrior (arXiv 2605.13959) | Phân phối noise lấy chunk trước làm tâm | **Có** | Ngầm | Cùng mục tiêu chống mode switching, nhưng cần train |
| RTC (Black et al., NeurIPS 2025) | Inpainting phần chunk đã khoá, khi chạy bất đồng bộ | Không | Có | Bổ sung, có thể kết hợp |
| BID (ICLR 2025) | Mẫu tốt nhất theo backward coherence + forward contrast | Không, cần thêm checkpoint yếu | Có (một chunk trước) | CCS/XDC gần với backward coherence; không hơn medoid ở đây |
| MG-Select (ICLR 2026) | Best-of-N theo KL khi che bớt điều kiện | Không (bản mạnh hơn cần fine-tune) | Không | Mỗi decision độc lập; tốn thêm forward pass |
| RoboMonkey (arXiv 2506.17811) | Best-of-N với verifier VLM 7B; bỏ phiếu gripper để tạo đề xuất | **Có** | Không | Có bỏ phiếu, nhưng không dùng để chọn |
| VGAS (arXiv 2602.07399) | Best-of-N với critic Q đã train | **Có** | Không | Cần dữ liệu và train |
| ElevateHead (arXiv 2610.34031) | Chọn mode một lần rồi giữ suốt episode | **Có** | Có (giữ cứng) | Không phải VLA |

**Vị trí novelty sau khi đọc:**
1. Chọn theo đồng thuận (medoid) trên N mẫu của một VLA đóng băng chưa thấy ai báo cáo, nhưng là ý tưởng đơn giản.
2. Đóng góp có giá trị hơn là **phép so sánh có kiểm soát và cơ chế được đo**:
   - Trên cùng N mẫu, chọn h cho 0 pp, chọn chunk cho +11 pp.
   - Độ lệch kế hoạch dự đoán thất bại (AUROC 0.87).
   - Phần lệch còn lại là policy đổi ý theo quan sát.
   - Lợi ích phụ thuộc giao diện điều khiển (đích tuyệt đối so với delta OSC).
3. Ba mở rộng có tính mới hơn (ACS, EGC, XDC) **không vượt được medoid**. Đây là kết quả âm, cần báo cáo trung thực.

---

## 8. Hạn chế và rủi ro

1. **Medoid > CCS đầy đủ.**
   - Bỏ phiếu gripper và tính liền mạch không giúp được gì, còn hơi kéo kết quả xuống (−2.4 pp).
   - Có thể do liền mạch với overlap chỉ 2 bước bám theo chunk trước, kể cả khi chunk trước sai.
   - Đề xuất giữ **medoid** làm phương pháp chính. Kết luận này mới chỉ đo trên VLABench; Track A không chạy CCS đầy đủ nên chưa xác nhận được.
1b. **Lợi ích không chuyển sang LIBERO-10.**
   - Trên LIBERO, medoid chỉ hơn +0.7 đến +1.9 pp, không có ý nghĩa thống kê, và không đạt tiêu chí manifest.
   - Phương pháp vì vậy **chưa phải cải tiến tổng quát**. Hiện chỉ có bằng chứng mạnh trên một benchmark.
   - Cần hiểu điều kiện nào làm nó hiệu quả (giả thuyết về giao diện điều khiển, mục 6.2) trước khi claim rộng hơn.
2. **Chưa sửa được lỗi ngữ nghĩa.**
   - T3 (common sense) tăng ít nhất.
   - select_painting và select_unique_type_mahjong không tăng.
   - Đồng thuận chỉ khuếch đại phương án mà policy tin nhất. Nếu policy hiểu sai lệnh thì đa số cũng sai.
3. **Rủi ro khi gặp tình huống ngoài phân phối.** Các mẫu có thể tụ quanh cùng một lỗi. Khi đó medoid vẫn chọn trong vùng lỗi. Cần đo trên suite có nhiễu mạnh hơn.
4. **Chi phí.** Mỗi decision tốn khoảng +20% so với DUAC server và +40% so với vanilla (1 mẫu). Chưa đo với N nhỏ hơn (4, 8).
5. **Chỉ có mô phỏng.** Chưa thử trên robot thật, hoặc khi chạy bất đồng bộ (RTC), nơi chunk nối nhau theo cách khác.

---

## 9. Kế hoạch tiếp theo

| # | Việc | Mục đích | Chi phí ước tính |
|---|---|---|---|
| 0 | **Đổi đòn bẩy: làm ổn định cách policy phản ứng với quan sát.** Phần đổi kế hoạch còn lại là do policy đổi ý theo quan sát (mục 6.3). Ứng viên: đồng thuận qua K biến thể nhỏ của ảnh đầu vào (augment, mỗi biến thể N/K mẫu), rồi lấy medoid trên tập gộp | Vượt medoid; có tính mới hơn chọn mẫu đơn thuần | Mỗi lần gọi model đắt hơn khoảng 2–3 lần (K lần encode prefix) |
| 1 | Kiểm giả thuyết giao diện điều khiển: trên VLABench, làm mượt đích tuyệt đối (ví dụ nội suy hoặc giới hạn tốc độ đích ở chỗ nối chunk) cho sample0_h8. Nếu sample0 + làm mượt đạt gần medoid, cơ chế là "nhảy đích"; nếu không, medoid còn tác dụng khác | Giải thích vì sao VLABench khác LIBERO | Khoảng 2.5 giờ GPU |
| 1b | Chạy ccs_h8 đầy đủ trên LIBERO-10 | Xác nhận medoid > CCS trên benchmark thứ hai | Khoảng 2.5 giờ GPU |
| 2 | Quét N ∈ {4, 8, 20} cho medoid trên VLABench | Đường cong chi phí và lợi ích | Khoảng 2.5 giờ GPU mỗi giá trị N |
| 3 | Quét h ∈ {5, 8, 10} cho medoid | Xem chọn chunk có đổi h tối ưu không, vì ít nhảy mode có thể cho phép h nhỏ | Khoảng 2.5 giờ mỗi giá trị h |
| 4 | RoboCasa365 (chunk 50) | Chunk dài, task composite | Khoảng 1 ngày |
| 5 | Phân tích thất bại còn lại | Tách lỗi thao tác khỏi lỗi ngữ nghĩa, xem video | Offline |
| 6 | Kết hợp với làm mượt chỗ nối chunk (RTC, temporal ensembling) | Hai cơ chế bổ sung cho nhau | Khoảng 1 ngày |

---

## 10. Tái lập

```bash
cd /mnt/ssd-storage/hieudd/aac-duac
# self-test
cd flowbased_aac && ../.venv-vlabench/bin/python -m flowbased_aac.ccs && cd ..
# VLABench: 3 arm chọn chunk trên cùng AAC server (ghép cặp với các arm cũ trong outputs/vlabench/full)
ARMS="ccs_h8 sample0_h8 medoid_h8" TAG=_ccs CONC=10 setsid nohup bash tools/run_vlabench_full.sh \
  > outputs/vlabench/full/runner_ccs.txt 2>&1 &
.venv-vlabench/bin/python analysis/vlabench_ccs_report.py outputs/vlabench/full
# LIBERO-10 (Track A), VRAM <= 27 GB
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 setsid nohup .venv-libero/bin/python tools/schedule_track_a.py \
  --stage select --arms medoid_h8,sample0_h8,medoid_h10 --clients 12 > outputs/A/logs_sched/runner_select_capped.txt 2>&1 &
.venv-libero/bin/python analysis/track_a_select_report.py
```

---

## Tham khảo

- Liang et al., *Adaptive Action Chunking at Inference-time for Vision-Language-Action Models*, CVPR 2026 (arXiv:2604.04161).
- Ben-Ami et al., *Adaptive Action Horizon Selection via Flow-Field Uncertainty* (DUAC), RoboPAD @ NeurIPS 2026.
- Liu et al., *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling* (BID), ICLR 2025 (arXiv:2408.17355).
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, temporal ensembling), RSS 2023.
- Black et al., *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025.
- Tài liệu nội bộ: [REPORT_trackA.md](REPORT_trackA.md), [track_a/DIAGNOSIS.md](track_a/DIAGNOSIS.md), [VLABENCH_INTEGRATION.md](VLABENCH_INTEGRATION.md), [task3010.md](task3010.md).
