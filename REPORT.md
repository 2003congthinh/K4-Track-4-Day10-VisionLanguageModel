Họ tên: [[TODO: họ tên]]
`dataset_version`: d10-21e4070301

> Các số liệu lấy từ thí nghiệm (F1, CI95, FP/ảnh, VRAM, phút, hiệu F1 ghép cặp) được điền tự động từ `report/evidence/*.json` bằng `python lab/fill_report.py` (không gõ tay, để luôn khớp với bằng chứng). Các số khác (tham số từng khối, số ảnh/box, kiểm tra rò rỉ, kiểm tra thang toạ độ "oracle") là số mình tự đo bằng các cell trong `lab_day10.ipynb` (mục 1, 2, 4, 5), chạy được không cần GPU.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – (không huấn luyện) | – | {{e0.f1}} | {{e0.ci}} | {{e0.fp}} | {{e0.f1_pothole}} | {{e0.f1_crack}} | {{e0.f1_manhole}} |
| E1 | LoRA LLM | {{e1.params}} | {{e1.vram}} | {{e1.min}} | {{e1.f1}} | {{e1.ci}} | {{e1.fp}} | {{e1.f1_pothole}} | {{e1.f1_crack}} | {{e1.f1_manhole}} |
| E2 | + projector | {{e2.params}} | {{e2.vram}} | {{e2.min}} | {{e2.f1}} | {{e2.ci}} | {{e2.fp}} | {{e2.f1_pothole}} | {{e2.f1_crack}} | {{e2.f1_manhole}} |
| E3 | + LoRA ViT | {{e3.params}} | {{e3.vram}} | {{e3.min}} | {{e3.f1}} | {{e3.ci}} | {{e3.fp}} | {{e3.f1_pothole}} | {{e3.f1_crack}} | {{e3.f1_manhole}} |
| E4 | coord_scale = norm1000 (= setup gốc) | {{e4_coord_scale_norm1000.params}} | {{e4_coord_scale_norm1000.vram}} | {{e4_coord_scale_norm1000.min}} | {{e4_coord_scale_norm1000.f1}} | {{e4_coord_scale_norm1000.ci}} | {{e4_coord_scale_norm1000.fp}} | {{e4_coord_scale_norm1000.f1_pothole}} | {{e4_coord_scale_norm1000.f1_crack}} | {{e4_coord_scale_norm1000.f1_manhole}} |
| E4 | coord_scale = pixel | {{e4_coord_scale_pixel.params}} | {{e4_coord_scale_pixel.vram}} | {{e4_coord_scale_pixel.min}} | {{e4_coord_scale_pixel.f1}} | {{e4_coord_scale_pixel.ci}} | {{e4_coord_scale_pixel.fp}} | {{e4_coord_scale_pixel.f1_pothole}} | {{e4_coord_scale_pixel.f1_crack}} | {{e4_coord_scale_pixel.f1_manhole}} |

Bổ sung (cùng nguồn): `parse_valid_rate` E0/E1/E2/E3 = {{e0.valid}} / {{e1.valid}} / {{e2.valid}} / {{e3.valid}}; mean IoU của các cặp khớp = {{e0.iou}} / {{e1.iou}} / {{e2.iou}} / {{e3.iou}}. Bảng đầy đủ hơn (P, R, meanIoU) do notebook sinh ra ở `report/summary_table.md`.

**Ảnh minh hoạ** (sinh trong notebook, mục 7):

- `report/figures/qualitative_<setup tốt nhất>.png` — 8 ảnh test: hàng trên 4 ảnh dự đoán tốt, hàng dưới 4 ảnh sai nhiều nhất. Xanh lá dày = box đúng (GT), xanh dương đứt = dự đoán khớp (TP), đỏ đứt = dự đoán sai (FP); tiêu đề ghi TP/FP/FN và lớp bị sót.
- `report/figures/qualitative_e0.png` — cùng kiểu cho E0 zero-shot.
- `report/figures/compare_runs.png` — 3 ảnh nhiều box nhất qua E0 → E1 → setup tốt nhất.
- `report/figures/recall_by_size_<setup>.png`, `report/figures/results_overview.png`, `report/figures/loss_curves.png`, `report/figures/e4_coord_scale.png`, `report/figures/data_samples.png`, `report/figures/data_stats.png`, `report/figures/leak_check.png`, `report/figures/label_roundtrip.png`.

[[TODO: sau khi chạy Colab, thay `<setup tốt nhất>` bằng tên thật (vd `qualitative_e2.png`) và chèn ảnh bằng `![](report/figures/...)`]]

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Mình đếm trên cây module thật của `Qwen3VLForConditionalGeneration` (dựng trên thiết bị `meta` nên không cần tải trọng số, hàm `param_report()` trong notebook mục 5):

| Khối | Tên module | Tham số |
|---|---|---|
| ViT (24 khối + patch_embed + pos_embed) | `model.visual.blocks`, ... | 306.242.560 |
| Projector chính | `model.visual.merger` | 27.271.680 |
| Projector DeepStack (3 merger, lấy đặc trưng ở lớp 5, 11, 17 của ViT) | `model.visual.deepstack_merger_list` | 81.833.472 |
| → toàn bộ `model.visual` | | 415.347.712 |
| LLM (36 lớp decoder, gồm embedding) | `model.language_model` | 4.022.468.096 |
| trong đó `embed_tokens` (151.936 × 2.560, **dùng chung** với `lm_head`) | | 388.956.160 |
| **Tổng** | | **4.437.815.808** |

Luồng dữ liệu: ảnh 640×360 → processor resize về 640×352 → 22×40 = 880 patch 16×16 → ViT → `merger` gộp 2×2 patch thành **220 token** 2.560 chiều đưa vào LLM; 3 merger DeepStack cộng thêm đặc trưng tầng giữa của ViT vào 3 lớp đầu của LLM. LLM sinh JSON.

Cập nhật trong từng thí nghiệm (LoRA r=16, α=32 trên các `nn.Linear`):

- **E1**: chỉ LoRA trên 252 module của LLM (36 lớp × `q,k,v,o,gate,up,down`) = **33.030.144** tham số (0,74% model). ViT, projector, embedding đóng băng; LLM gốc ở dạng 4-bit NF4.
- **E2**: E1 + **mở toàn bộ projector** (`merger` + `deepstack_merger_list`, fp32) = 33.030.144 + 109.105.152 = **142.135.296**.
- **E3**: E2 + LoRA trên 96 module của ViT (24 khối × `attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2`) thêm 6.291.456 → **148.426.752**. Thân ViT vẫn đóng băng, chỉ học phần LoRA.

Số `trainable_params` trong bằng chứng: E1 = {{e1.params}}, E2 = {{e2.params}}, E3 = {{e3.params}}, khớp với số mình tính trước. E2 − E1 đúng bằng 109.105.152 ≈ 109M như GUIDE.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate` của E0 = **{{e0.valid}}**, F1 = {{e0.f1}} (P = {{e0.p}}, R = {{e0.r}}), FP/ảnh = {{e0.fp}}.

Mình phân loại lỗi bằng hàm `diagnose()` (notebook, cell sau mục 3). Hàm này gắn nhãn cho từng ảnh: JSON hỏng, bọc ```json, chữ thừa, box bịa (IoU < 0,1 với mọi box đúng), sai lớp, box lệch (0,1 ≤ IoU < 0,5), toạ độ > 1000. Hai lỗi nhiều nhất:

1. [[TODO: nhãn lỗi #1 + số ảnh/tỉ lệ từ `error_report(WORK / "e0.jsonl")`]]
2. [[TODO: nhãn lỗi #2 + số ảnh/tỉ lệ]]

Ví dụ output thô (ảnh `[[TODO: image_id]]`):

```
[[TODO: dán 1 output thô E0, in ở cell "Phân tích output thô của E0"]]
```

Mình cũng kiểm tra xem model gốc có đang trả pixel thay vì 0–1000 không: chấm lại `work/e0.jsonl` với `--coord pixel` thì F1 = [[TODO: F1 của E0-as-pixel, file work/e0_as_pixel.json]], còn `coord_stats()` cho thấy [[TODO: tỉ lệ box có y > 360]]. [[TODO: kết luận: model gốc dùng đúng thang 0–1000 nhưng box lệch/bịa, hay trả pixel]]

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

F1 gần như **sụp về 0**. Mình kiểm chứng bằng ba cách:

1. **Oracle, không cần GPU** (`coord_mismatch_demo()`, notebook mục 2): đổi chính đáp án đúng của 311 ảnh test thành JSON rồi chấm bằng `scripts/score.py`:

   | Nhãn viết theo | Chấm với `--coord norm1000` | Chấm với `--coord pixel` |
   |---|---|---|
   | 0–1000 | F1 = **1,000** (mean IoU 0,988) | F1 = **0,000** |
   | pixel 640×360 | F1 = **0,000** | F1 = **1,000** (mean IoU 1,000) |

   Đáp án hoàn hảo mà chỉ lệch thang cũng chỉ còn 0 TP. Lý do: box pixel (x ≤ 640, y ≤ 360) khi bị đọc như thang 0–1000 sẽ bị **co lại 0,64 lần theo x và 0,36 lần theo y** về phía góc trên-trái. Ngược lại, box 0–1000 đọc như pixel bị phóng 1,56 lần / 2,78 lần (nhiều box rơi ra ngoài khung). Box ở đây rất nhỏ (diện tích trung vị: pothole 0,38%, manhole 0,46%, crack 1,42% khung hình), nên chỉ cần dịch vài chục pixel là IoU < 0,5, tức là không còn cặp nào khớp.
2. **Trên output thật của model**: chấm lại dự đoán của setup tốt nhất với `--coord pixel`: F1 từ [[TODO: F1 đúng thang]] xuống [[TODO: F1 trong work/<best>_as_pixel.json]].
3. **E4 (coord_scale)**: huấn luyện lại với nhãn + prompt theo pixel và chấm đúng `--coord pixel` thì F1 = {{e4_coord_scale_pixel.f1}} (so với {{e4_coord_scale_norm1000.f1}} khi dùng 0–1000). Như vậy thang nào cũng học được, miễn là **nhãn, prompt và cách chấm cùng một thang**. Xem câu 10.

Vì sao Qwen3-VL dùng 0–1000: toạ độ tương đối **không phụ thuộc độ phân giải**. Processor luôn resize ảnh (ở đây 640×360 → 640×352 để chia hết cho 32), và mỗi ảnh khác cỡ sẽ có dải pixel khác nhau. Còn 0–1000 là quy ước cố định mà model đã thấy khi pretrain grounding.

Một chi tiết mình tự phát hiện: nhãn gốc của dataset **khớp đúng lưới pixel** (quy ngược về pixel sai số 0,001 px), nên làm tròn sang 0–1000 mất một chút độ chính xác (sai số tối đa 0,32 px, mean IoU của oracle 0,988 thay vì 1,000). Mất mát này không đổi F1@0,5 nhưng có ảnh hưởng nếu chấm ở IoU cao hơn.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Mình đặt `labels = -100` cho toàn bộ phần prompt (`encode_example`). Trước đó có kiểm tra bằng `assert` rằng prompt (đã thêm generation prompt) là **tiền tố đúng** của chuỗi đầy đủ. Giải mã riêng các token có `labels != -100` thì ra đúng JSON đáp án + `<|im_end|>`.

Đo trên 200 ảnh train: mỗi mẫu có **405 token prompt** (trong đó **220 token ảnh** `<|image_pad|>` và khoảng 185 token chữ của template + câu lệnh), còn câu trả lời chỉ có trung vị **59 token**, tức khoảng **15%** chuỗi. Nếu tính loss trên tất cả:

- **85% tín hiệu loss rơi vào token không cần học.** Prompt chữ giống hệt nhau ở mọi mẫu, nên model chỉ học thuộc câu lệnh. 220 token ảnh đều là cùng một id `<|image_pad|>` lặp lại, nên đoán "token tiếp theo vẫn là image_pad" là việc vô nghĩa, không dạy gì về vị trí hư hại. Loss trung bình bị pha loãng khoảng 6–7 lần, gradient cho phần JSON (phần duy nhất có ích) yếu hẳn với cùng lr.
- **Loss mất ý nghĩa để theo dõi:** phần dễ (prompt lặp lại) nhanh chóng về gần 0 và kéo loss tổng xuống, trong khi phần grounding chưa học được.
- **Model học sai hành vi:** nó được thưởng khi sinh lại câu lệnh/token ảnh, nên dễ lặp prompt hoặc sinh token đặc biệt thay vì JSON, làm `parse_valid_rate` giảm.

Lúc suy luận prompt luôn được cho sẵn, nên chỉ phân phối của **câu trả lời có điều kiện theo ảnh + prompt** mới cần tối ưu.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Khi lượng tử hoá 4-bit, bitsandbytes thay mỗi `nn.Linear` bằng `Linear4bit`. Trọng số được nén thành `Params4bit` (2 giá trị NF4 trong một byte uint8, kèm hệ số absmax theo block 64). Đây không phải tensor số thực nên **không có gradient** và optimizer không cập nhật trực tiếp được. NF4 chỉ có 16 mức giá trị, nên dù có giải nén → cập nhật → nén lại thì các bước cập nhật nhỏ (lr 2e-4) cũng bị làm tròn mất. Với tầng 4-bit chỉ có thể học **thêm LoRA bên cạnh**, không thể "huấn luyện đầy đủ". Projector nằm trong `model.visual`, nên nếu lượng tử hoá cả ViT thì `merger` cũng thành 4-bit và E2/E3 không còn đúng nghĩa.

Cách mình xử lý: `BitsAndBytesConfig(..., llm_int8_skip_modules=["visual", "lm_head"])`. Mọi module có tên chứa `visual` (ViT + `merger` + `deepstack_merger_list`) và `lm_head` được giữ fp16, chỉ 252 Linear của LLM bị nén NF4. Ở E2/E3 mình đổi riêng 109.105.152 tham số projector sang **float32** và bật `requires_grad`.

Vì sao float32:

- Huấn luyện dùng `autocast(fp16)` + `GradScaler`. `GradScaler.unscale_()` **từ chối gradient fp16** ("Attempting to unscale FP16 gradients"), nên tham số được tối ưu phải là fp32. Đây cũng là lý do peft tự đưa trọng số LoRA lên fp32.
- fp16 chỉ có 10 bit mantissa. Khi lr giảm theo cosine về 0, các bước cập nhật nhỏ bị làm tròn mất. Moment bậc hai của Adam (≈ g², với g ~ 1e-4 là ~1e-8) thấp hơn ngưỡng nhỏ nhất của fp16 nên bị underflow, chia cho số gần 0 thì bước cập nhật nổ.
- Trong forward, `autocast` vẫn tính matmul của merger ở fp16, nên dtype không xung đột với ViT fp16.

Cái giá: 109M tham số fp32 cần ≈ 0,44 GB trọng số + 0,44 GB gradient + 0,87 GB trạng thái Adam ≈ **1,75 GB** VRAM thêm so với E1 (đo được: E2 {{e2.vram}} GB so với E1 {{e1.vram}} GB). Khi lưu, mình lưu projector riêng (`projector.safetensors`), vì `save_pretrained` của peft chỉ lưu LoRA. Lúc suy luận, `load_for_inference()` nạp lại và **in độ lệch |Δ| so với projector gốc** để chắc chắn đã nạp đúng bản đã huấn luyện. Thiếu bước này thì E2/E3 trông y như E1.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Ảnh là các frame cắt từ video dashcam (xe chạy 20–50 km/h). Hai frame cách nhau vài giây gần như trùng nhau: cùng con đường, cùng ổ gà, box ở gần cùng vị trí. Nếu chia ngẫu nhiên, frame "anh em" của ảnh test nằm trong train, và model chỉ cần **nhớ cảnh** là đoán đúng. F1 khi đó đo khả năng học thuộc chứ không đo khả năng tổng quát sang đoạn đường mới.

Mình đo trực tiếp bằng `leak_check()` (notebook mục 1). Với mỗi ảnh test, mình tìm ảnh train gần nhất theo khoảng cách L2 trên thumbnail xám 32×18:

| | Split theo phiên (đề) | Split ngẫu nhiên (cùng 311 ảnh test, 3 seed) |
|---|---|---|
| Khoảng cách tới ảnh train gần nhất (trung vị) | 3,72 | 2,68 – 2,73 |
| % ảnh test có "bản gần trùng" trong train (d < 2,76) | 5,1% | **51,1% – 54,0%** |

Với split ngẫu nhiên, hơn một nửa ảnh test có bản gần trùng trong train. Ví dụ `vlcsnap-2025-02-19-15h02m56s842` (test) và `...15h03m10s511` (train) cách nhau 14 giây, d = 0,01, nhìn bằng mắt không phân biệt được (`report/figures/leak_check.png`). F1 trên split như vậy sẽ cao giả. Split của đề gom 18 phiên quay (`session` = ngày + giờ, hoặc nhóm 50 số với `vlcsnap-000xx`) thành 13 / 2 / 3 phiên cho train / val / test, và mình đã `assert` rằng không phiên nào nằm ở hai split.

| Split | Phiên | Ảnh | Box | pothole | crack | manhole |
|---|---|---|---|---|---|---|
| train | 13 | 1.422 | 3.515 | 1.074 | 1.750 | 691 |
| val | 2 | 276 | 639 | 66 | 470 | 103 |
| test | 3 | 311 | 583 | 121 | 299 | 163 |

Hệ quả: phân bố lớp lệch (val chỉ có 66 pothole). Ngoài ra, test chỉ đến từ 3 phiên nên các ảnh test **không độc lập** với nhau, xem câu 8.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

| | F1 | CI95 | FP/ảnh | Tham số học | VRAM đỉnh | Phút |
|---|---|---|---|---|---|---|
| E1 | {{e1.f1}} | {{e1.ci}} | {{e1.fp}} | 33,0M | {{e1.vram}} | {{e1.min}} |
| E2 | {{e2.f1}} | {{e2.ci}} | {{e2.fp}} | 142,1M | {{e2.vram}} | {{e2.min}} |
| E3 | {{e3.f1}} | {{e3.ci}} | {{e3.fp}} | 148,4M | {{e3.vram}} | {{e3.min}} |

Hiệu F1 với CI95 **bootstrap ghép cặp** (cùng ảnh được lấy mẫu lại cho cả hai setup, hàm `lab/stats.py`):

- E1 − E0 = {{paired:e1:e0}}
- E2 − E1 = {{paired:e2:e1}}
- E3 − E2 = {{paired:e3:e2}}
- E3 − E1 = {{paired:e3:e1}}

Chi phí: E2 tốn thêm {{e2-e1.vram}} GB VRAM (khoảng 1,75 GB là tham số projector fp32 + Adam) và {{e2-e1.min}} phút so với E1. E3 tốn thêm {{e3-e2.vram}} GB và {{e3-e2.min}} phút so với E2, chủ yếu vì gradient phải lan ngược qua toàn bộ 24 khối ViT để tới LoRA của ViT, nên phải giữ (hoặc tính lại khi checkpointing) activation của 880 patch/ảnh.

[[TODO: kết luận sau khi có số. Mẫu: "E1 đã tăng F1 từ … lên …, vượt hẳn CI. Mở projector (E2) thay đổi F1 …, CI ghép cặp [ …, … ] có/không chứa 0 → (không) có bằng chứng E2 hơn E1. LoRA ViT (E3) … Setup đáng đồng tiền nhất là … vì …"]]

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

Độ rộng CI95 (bootstrap 1.000 lần theo ảnh, do `score.py` tính): E0 {{e0.ciw}}, E1 {{e1.ciw}}, E2 {{e2.ciw}}, E3 {{e3.ciw}}. Nghĩa là F1 của một setup chỉ biết được trong khoảng ±[[TODO: nửa độ rộng, vd ±0,04]] quanh giá trị đo.

Giới hạn:

- **CI chồng nhau thì không kết luận được thứ hạng.** Nếu hiệu giữa hai setup nhỏ hơn độ rộng này, thì có thể chỉ do 311 ảnh cụ thể "may" cho setup này. Bootstrap ghép cặp ở câu 7 chặt hơn (cùng ảnh khó/dễ cho cả hai setup nên phương sai chung bị triệt tiêu), nhưng một hiệu chỉ đáng tin khi CI ghép cặp **không chứa 0**.
- **F1 từng lớp còn kém chắc hơn:** test chỉ có 121 pothole, 163 manhole, 299 crack. Chênh vài TP ở pothole đã làm F1 lớp đó dao động vài điểm.
- **CI này còn lạc quan:** bootstrap giả định 311 ảnh độc lập, nhưng chúng đến từ **chỉ 3 phiên quay**, và frame liền nhau tương quan mạnh (câu 6). Số "mẫu độc lập" thực tế ít hơn 311 nhiều, nên CI thật còn rộng hơn. Một setup có thể hơn trên 3 đoạn đường này mà không hơn trên đường khác.
- Mỗi setup chỉ chạy **1 seed**, nên CI không bao gồm phương sai do huấn luyện (khởi tạo LoRA, thứ tự batch).

Vì vậy mình chỉ coi là chắc chắn những kết luận có hiệu lớn hơn hẳn CI (ví dụ E1 so với E0). Các khác biệt nhỏ giữa E1/E2/E3 mình chỉ ghi là "xu hướng".

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

F1 từng lớp của setup tốt nhất: [[TODO: pothole …, crack …, manhole … (lấy từ dòng setup tốt nhất ở bảng)]]. Lớp kém nhất là **[[TODO]]**.

Các yếu tố mình đo được trên dữ liệu:

- **Kích thước:** diện tích box trung vị pothole 0,38%, manhole 0,46%, crack 1,42% khung hình (`report/figures/data_stats.png`). Box pothole/manhole chỉ khoảng 30×20 pixel. Trong khi đó mỗi token ảnh của Qwen3-VL ứng với một ô 32×32 pixel, nên một ổ gà nhỏ chỉ chiếm khoảng một token, rất khó định vị đủ chính xác để IoU ≥ 0,5. `recall_by_size()` cho thấy recall ở nhóm < 0,3% khung hình là [[TODO]] so với [[TODO]] ở nhóm > 3%.
- **Ngữ nghĩa / độ mơ hồ của nhãn:** crack mảnh, dài, phân nhánh, và "vết nứt kết thúc ở đâu" là quyết định chủ quan của người gán nhãn. Box của crack thường dài và hẹp, chỉ lệch một đầu là IoU tụt dưới 0,5. Nhiều vết nứt sát nhau có lúc được gán một box lớn, lúc nhiều box nhỏ. Manhole có hình dạng rõ (tròn/vuông, kim loại) nên dễ nhận dạng hơn, nhưng dễ nhầm với miếng vá đường.
- **Số mẫu:** train có 1.750 crack, 1.074 pothole, 691 manhole. Manhole ít nhất nhưng dễ về ngữ nghĩa.
- **Loại FP** (`fp_breakdown()`): [[TODO: tỉ lệ FP "bịa trên nền" / "lệch vị trí" / "sai lớp" theo lớp]].

Ảnh minh hoạ: `report/figures/qualitative_<best>.png` (8 ảnh đúng/sai có chú thích TP/FP/FN) và `report/figures/recall_by_size_<best>.png`.

[[TODO: 2–3 câu nhận xét cụ thể từ ảnh, vd "ảnh … sót 3 pothole nhỏ ở xa; ảnh … vẽ một box crack bao cả mặt đường thay vì từng vết"]]

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

Mình chọn biến **`coord_scale`**: `norm1000` và `pixel`. Đây đúng là giả thuyết của câu 3: nếu model đã quen thang 0–1000 từ pretrain thì ép nó học pixel có thiệt không? Mình giữ nguyên mọi thứ khác của setup tốt nhất ([[TODO: E1/E2/E3]]): cùng r, lr, số bước, seed, dữ liệu. Chỉ đổi **cả** nhãn huấn luyện, prompt ("absolute integer pixel coordinates of the 640x360 image") **và** cách chấm (`--coord pixel`). Giá trị `norm1000` chính là setup gốc, nên mình chấm lại đúng file dự đoán của nó dưới tên E4, không huấn luyện lại.

| coord_scale | F1 | CI95 | F1 pothole / crack / manhole | parse_valid |
|---|---|---|---|---|
| norm1000 | {{e4_coord_scale_norm1000.f1}} | {{e4_coord_scale_norm1000.ci}} | {{e4_coord_scale_norm1000.f1_pothole}} / {{e4_coord_scale_norm1000.f1_crack}} / {{e4_coord_scale_norm1000.f1_manhole}} | {{e4_coord_scale_norm1000.valid}} |
| pixel | {{e4_coord_scale_pixel.f1}} | {{e4_coord_scale_pixel.ci}} | {{e4_coord_scale_pixel.f1_pothole}} / {{e4_coord_scale_pixel.f1_crack}} / {{e4_coord_scale_pixel.f1_manhole}} | {{e4_coord_scale_pixel.valid}} |

Hiệu pixel − norm1000 = {{paired:e4_coord_scale_pixel:e4_coord_scale_norm1000}}.

[[TODO: kết luận. Nếu CI ghép cặp chứa 0: "với 300 bước, model học được cả thang pixel; hiệu nằm trong CI nên mình không kết luận thang nào tốt hơn, chỉ chắc chắn rằng thang phải nhất quán giữa nhãn–prompt–chấm (câu 3)". Nếu pixel kém rõ: "thang 0–1000 khớp với pretrain của Qwen3-VL nên học nhanh hơn; ở thang pixel model còn phải tự học quy đổi theo 640×360 trong khi lưới nó thấy là 640×352"]]

Độ tin cậy: đây là **một lần chạy, một seed**, trên 311 ảnh từ 3 phiên. Kết luận chỉ đáng tin nếu CI ghép cặp không chứa 0, và kể cả khi đó cũng chỉ đúng cho 300 bước huấn luyện này. Thang pixel có thể chỉ cần học lâu hơn.

## Nhìn lại

Các lỗi dữ liệu mình gặp mà công cụ của đề chưa bắt:

- **Nhãn suy biến:** `vlcsnap-00058` (train) có một box crack cao **0** pixel (`h = 0.0`). `prepare_data.py` chỉ kiểm tra giá trị nằm trong [0,1] nên cho qua. Nếu đưa vào nhãn, model học sinh box `y2 == y1`, và `parse_output` sẽ coi **cả output của ảnh đó** là JSON hỏng (một box suy biến làm hỏng mọi box khác). Mình bỏ các nhãn w=0/h=0, và trong `yolo_to_box` nới 1 đơn vị cho box bị làm tròn thành suy biến.
- **Box tràn khung:** 63/4.736 box YOLO tràn ra ngoài ảnh, tối đa 1,56% chiều rộng (~10 px). Mình kẹp nhãn huấn luyện về [0, 1000], nhưng `score.py` dựng box đúng **không kẹp**. Vì vậy với các box này, dự đoán hoàn hảo cũng có IoU < 1 (không đổi F1@0,5).
- **Mất độ chính xác do làm tròn 0–1000:** nhãn gốc khớp lưới pixel, nên oracle 0–1000 chỉ đạt mean IoU 0,988, còn oracle pixel đạt 1,000.
- **Độ dài đáp án:** 142/1.422 ảnh train có đáp án đúng dài hơn 128 token (dài nhất 365). Với `max_completion_length=128` của E5, model trả đúng cũng bị cắt thành JSON hỏng (reward 0), tạo áp lực "trả ít box". Mình lọc dữ liệu GRPO, chỉ giữ ảnh có đáp án ≤ 100 token (còn 1.130/1.422 ảnh). Với suy luận, đáp án test dài nhất 195 token nên `max_new_tokens=256` là đủ.
- **CI của trọng tài** giả định ảnh test độc lập, trong khi chúng là frame liên tiếp của 3 phiên (câu 8).
- Nhãn có chữ hoa (`"Pothole"`) bị `score.py` tính vào lớp `other`, tức là FP + FN. Prompt của mình ghi rõ tên lớp viết thường để tránh lỗi này.

[[TODO: E5 (nếu chạy): reward trung bình đầu/cuối, tỉ lệ trả [] và số box/ảnh so với SFT (`hacking_stats`), F1 E5 = {{e5.f1}} {{e5.ci}} so với setup gốc: thay eX trong {{paired:e5:eX}} bằng setup gốc rồi chạy lại fill_report → có reward hacking không]]

Điều mình chưa hiểu hết: DeepStack cộng đặc trưng của ViT vào 3 lớp đầu của LLM. Mình chưa đo riêng được việc mở `deepstack_merger_list` (82M) hay `merger` (27M) đóng góp phần lớn vào khác biệt E2 so với E1. Muốn biết thì cần thêm một ablation chỉ mở `merger`. [[TODO: bổ sung điều bạn thấy bất ngờ khi chạy thật, vd tốc độ, VRAM, lỗi Colab]]
