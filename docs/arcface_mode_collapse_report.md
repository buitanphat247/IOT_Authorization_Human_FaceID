# Báo Cáo Phân Tích Kỹ Thuật: Sự Cố Mode Collapse Trong Quá Trình Fine-tune ArcFace

**Dự án:** Nhận diện khuôn mặt & Chống giả mạo (Face Recognition & Anti-Spoofing)
**Tài liệu:** Phân tích nguyên nhân và giải pháp khắc phục hiện tượng Mode Collapse
**Phiên bản được phân tích:** Mô hình v3 (Bị lỗi) vs Mô hình v4 (Khắc phục)

---

## 1. Hiện tượng "Mode Collapse" (Feature Collapse) là gì?

Trong bài toán định danh khuôn mặt (Face Recognition) sử dụng cấu trúc ArcFace, mục tiêu của mạng Neural Network là ánh xạ (mapping) các khuôn mặt thành một vector đặc trưng 512 chiều (embedding). Trong không gian siêu cầu (hypersphere) này, lý tưởng nhất là:
- Khuôn mặt của **cùng một người** (Intra-class) hội tụ về cùng một cụm điểm.
- Khuôn mặt của **những người khác nhau** (Inter-class) bị đẩy ra xa nhau (ví dụ: người A ở "Bắc Cực", người B ở "Nam Cực").

Tuy nhiên, khi quá trình huấn luyện bị mất ổn định nghiêm trọng, mô hình sẽ rơi vào trạng thái cực đoan mang tên **Mode Collapse** (hay Feature Collapse). Thay vì cố gắng giải quyết bài toán khó là phân tầng hàng vạn khuôn mặt, mạng Neural Network sẽ tìm một "đường tắt toán học" (mathematical shortcut): nó **gom tất cả mọi khuôn mặt của tất cả mọi người (10,572 lớp) về chung một cụm vĩ độ/tọa độ duy nhất** trên quả cầu.

### Biểu hiện thực tế trên hệ thống:
- Tất cả mọi người đưa mặt vào hệ thống (dù là sếp, nhân viên hay người lạ) đều ra những vector có tọa độ `[0.02, -0.41, 0.99, ...]` giống hệt nhau.
- Độ tương đồng Cosine (Cosine Similarity) giữa **bất kỳ ai với bất kỳ ai** luôn lớn hơn `0.80`, thậm chí tiệm cận `0.99`.
- **Hệ quả khi tích hợp MiniFASNet:** Khi kết hợp cùng mô hình chống giả mạo (MiniFASNet), MiniFASNet hoạt động đúng và báo "Đây là người thật". Tuy nhiên, bước tiếp theo ArcFace lại phán đoán "Người thật này giống 90% với tất cả mọi người trong công ty" → Gây ra hiện tượng False Acceptance (chấp nhận sai) trầm trọng, sụp đổ bài toán định danh.

---

## 2. Nguyên nhân gốc rễ gây ra Mode Collapse trong bản v3

Code huấn luyện cũ (v3) chứa 4 lỗi thiết lập siêu tham số tồi tệ đã "kích nổ" hiện tượng này:

### Nguyên nhân 2.1: Tốc độ học (Learning Rate) quá khổng lồ
Bản v3 sử dụng cơ chế `Linear Scaling Rule` (vốn phổ biến khi train ImageNet) để tự động hóa Learning Rate (LR) dựa trên Batch Size.
- Công thức cũ: `HEAD_LR = 0.1 * (BATCH_SIZE / 256)`
- Với sức mạnh của GPU RTX 3090, chúng ta đã tối ưu `BATCH_SIZE` lên tới con số khổng lồ: `1792`.
- Kết quả: Máy tự động tính ra `HEAD_LR = 0.1 * (1792 / 256) = 0.70` (Mốc 0.70 là vô cùng khủng khiếp đối với một lớp phân loại ArcFace dùng phép chiếu Cosine). Việc Learning Rate điên cuồng phá vỡ mọi cấu trúc, ép mô hình "sập" về 1 điểm.

### Nguyên nhân 2.2: "Đòn bẩy" Margin quá gắt (Aggressive Margin)
Mục tiêu của ArcFace là tạo ra khoảng cách an toàn (margin `m`) giữa các danh tính.
- Mức margin `m = 0.50` rất hiệu quả khi mô hình **đã trưởng thành** và đã biết cách phân lập các khuôn mặt.
- Tuy nhiên, khi mới bắt đầu fine-tune (não bộ còn là tờ giấy trắng), các vector đang nhảy múa loạn xạ. Việc đột ngột ép một lực `m = 0.50` ngay từ `Epoch 1` tạo ra các Vector Gradient (đạo hàm) cực lớn, bóp méo không gian tối ưu hóa.

### Nguyên nhân 2.3: Scale (s) quá lớn ngay từ đầu
Thuộc tính `s = 64.0` trong v3 là bội số nhân của hàm Loss. Nhân với 64.0 làm các điểm gradient tăng theo cấp số nhân, khiến Optimizer SGD nhảy ra khỏi vùng giới hạn của không gian 512 chiều.

### Nguyên nhân 2.4: Lỗi Warmup Learning Rate tính chồng chéo (Bug Logic)
```python
# Lỗi thuật toán ở cấu hình v3:
pg['lr'] = pg['lr'] * warmup_factor 
```
Lỗi này khiến cho LR vốn đã cao lại bị lấy chính nó nhân đè lên ở mỗi epoch, làm sự ổn định hoàn toàn bằng Không.

### Nguyên nhân 2.5: Early Stopping dựa trên Val Loss thay vì Val Accuracy (Bug phát hiện trong quá trình train v4)
Trong bản v4 ban đầu, cơ chế Early Stopping (dừng sớm) được thiết kế dựa trên `Val Loss` (Độ lỗi trên tập kiểm tra). Tuy nhiên, khi kết hợp với **Curriculum Margin** (Giáo án tăng dần), Margin tăng từ `0.20` lên `0.50` qua 15 Epoch, kéo theo Val Loss **luôn tăng theo** (do bài thi khó hơn → điểm phạt cao hơn). Điều này khiến cho:
- Epoch 1: Val Loss = `16.24` → **NEW BEST** (Kỷ lục được lưu).
- Epoch 2-10: Val Loss = `16.44 → 19.41` → Luôn cao hơn kỷ lục → `No improvement x1, x2, ... x9`.
- **Epoch 11: `No improvement x10/10` → Early Stopping kích hoạt!**

Hậu quả: Model tốt nhất được lưu lại chỉ là model ở Epoch 1 (gần như chưa học gì, Acc chỉ `0.06%`), trong khi model thực sự đã đạt Acc `5.5%` ở Epoch 10 nhưng bị bỏ qua.

### Lưu ý: Tại sao Loss khởi điểm là 16.x thay vì 9.x?
Với 10,572 lớp (classes), Loss ngẫu nhiên theo lý thuyết Cross-Entropy thuần túy là `ln(10572) ≈ 9.27`. Tuy nhiên, ArcFace sử dụng hệ số **Scale `s = 30.0`** để nhân tất cả logits lên 30 lần trước khi đưa vào hàm Softmax. Khi logits bị khuếch đại 30 lần, hàm Softmax trở nên cực kì nhạy cảm — sai lệch một chút cũng bị phạt nặng gấp bội → Loss khởi điểm bị đội lên `16.x`. **Đây là hành vi bình thường của ArcFace có Scale, không phải do load nhầm model v3.**

---

## 3. Giải pháp khắc phục triệt để trong bản cập nhật (v4)

Để đưa mô hình trở lại quỹ đạo nhận diện chính xác từng dị biệt trên khuôn mặt con người, **bản v4** được thiết kế lại theo tiêu chuẩn công nghiệp với các giải pháp:

### 3.1. Thiết lập LR tuyệt đối và an toàn
Khóa trực tiếp thuộc tính Learning Rate thay vì tính toán nội suy:
- `HEAD_LR = 0.01` (Dành cho lớp ArcFace hoàn toàn mới, vừa đủ để học).
- `BACKBONE_LR = 0.0001` (Do ResNet50 đã được hội tụ từ ImageNet, chỉ cần bước học cực nhỏ để tránh quên đi những khả năng nhận diện cạnh, góc có sẵn).

### 3.2. Áp dụng giáo án tăng dần (Curriculum Margin)
Mô hình sẽ không bị "ép" ngay từ đầu.
- Ở Epoch 1, `m = 0.20`: Rất nhẹ nhàng, chỉ yêu cầu mô hình từ từ gom các khuôn mặt cùng nhóm.
- Từ Epoch 1 đến 15, `m` sẽ tăng dần rải đều lên tới mức `0.50`. Nhờ đó cấu trúc quả cầu không bị phá vỡ.
- Đồng thời, scale được ép xuống `s = 30.0` để giữ các vector hoạt động trong quỹ đạo êm ái hơn, tránh mất kiểm soát.

### 3.3. Siết chặt Gradient Clipping
Thêm một vòng an toàn (safety net) cho mạng học sâu:
- `torch.nn.utils.clip_grad_norm_` bị bóp từ `5.0` xuống `1.0`. Nếu trong quá trình tính toán xảy ra cực trị đột biến, nó sẽ tự động được gọt dũa lại, hệ thống không bao giờ bị "sốc thuốc" do Gradient nổ (Exploding Gradients).

### 3.4. Hệ thống Giám sát Phân tán "Camera" (Embedding Spread Monitor)
Thay vì nhắm mắt tin vào con số Accuracy (vốn là 0.0% trong giai đoạn đầu), cấu hình v4 bổ sung một thuật toán liên tục khảo sát ngẫu nhiên 256 khuôn mặt ở mỗi 50 steps:
- Đo lường khoảng cách giữa toàn bộ 256 khuôn mặt (`pairwise_cosine_sim`).
- Tính toán độ lệch chuẩn (`std`).
- Nếu mô hình phân bổ các khuôn mặt đẹp đẽ ra diện rộng, `std` sẽ luôn nằm ở ngưỡng `[0.06 ~ 0.15]`.
- Nếu mô hình bất chợt có bệnh gom về 1 điểm, `std` sẽ tụt xuống dưới `0.05`. Hệ thống sẽ ngay lập tức dừng chạy, kích hoạt còi báo động **"COLLAPSE DETECTED"** và lưu lại bằng chứng, tiết kiệm hàng giờ vô ích cho kỹ sư.

### 3.5. Sửa lỗi Early Stopping: Chuyển từ Val Loss sang Val Accuracy
Do Curriculum Margin khiến Val Loss luôn tăng theo từng Epoch (do bài thi ngày một khó hơn), nên tiêu chí đánh giá model tốt nhất phải dựa trên **Val Accuracy** (Độ chính xác thực tế) thay vì Val Loss.
```python
# SAI (v4 ban đầu): Early Stopping dựa trên Val Loss — luôn bị "No improvement"
if avg_vl < best_val_loss:
    best_val_loss = avg_vl  # Loss luôn tăng → không bao giờ cải thiện

# ĐÚNG (v4 đã sửa): Early Stopping dựa trên Val Accuracy
if avg_va > best_val_acc:
    best_val_acc = avg_va  # Accuracy luôn tăng → lưu model đúng thời điểm
```
Nhờ sửa đổi này, hệ thống sẽ lưu lại model có khả năng nhận diện chính xác nhất thay vì model ở Epoch 1 (gần như chưa học gì).

---

## 4. Bảng so sánh tổng hợp v3 vs v4

| Thông số | v3 (Bị lỗi) | v4 (Khắc phục) |
|----------|-------------|----------------|
| Head LR | `0.7` (scale theo batch) | `0.01` (cố định) |
| Backbone LR | `0.007` (scale theo batch) | `0.0001` (cố định) |
| ArcFace Margin | `m=0.50` ngay từ đầu | `m=0.20→0.50` tăng dần 15 epoch |
| ArcFace Scale | `s=64` | `s=30` |
| Gradient Clip | `max_norm=5.0` | `max_norm=1.0` |
| Warmup LR | Nhân chồng (bug) | Tuyến tính tuyệt đối |
| Embedding Monitor | Không có | Spread Monitor (std < 0.05 = COLLAPSE) |
| Early Stopping | Dựa trên Val Loss (sai) | Dựa trên Val Accuracy (đúng) |

---

### Tổng Kết
Hiện tượng Mode Collapse trong v3 không phải là lỗi dữ liệu xấu hay môi trường yếu, mà hoàn toàn xuất phát từ việc cấu hình Hyper-Parameters không phù hợp ở những vòng lặp đầu tiên. Bản v4 đã khắc phục được 5 lỗi cấu hình nghiêm trọng (LR, Margin, Scale, Warmup, Early Stopping) và bổ sung hệ thống giám sát Embedding Spread Monitor theo tiêu chuẩn khắt khe, giúp ArcFace và MiniFASNet hoạt động phối hợp với nhau nhịp nhàng nhất.

---

## 5. Nền Tảng Lý Thuyết: Tại Sao ArcFace Dễ Bị Collapse Hơn Softmax Thường?

### 5.1. Bài toán Face Recognition không phải Classification đơn thuần

Trong Face Recognition hiện đại, mục tiêu **không chỉ là phân loại đúng** (classification) mà là **học không gian embedding (metric learning)**:
- **Lúc train:** Dùng ArcFace margin-softmax loss với hàng ngàn class ID (10,572 người)
- **Lúc deploy/inference:** Bỏ hết classifier, chỉ giữ lại backbone → Lấy **embedding 512-D** → So sánh bằng **Cosine Similarity** giữa các khuôn mặt

Đây là lý do tại sao khi model bị Collapse (tất cả embedding giống nhau), thì dù chạy trên hệ thống production nó sẽ trả về Cosine > 0.80 cho mọi cặp người.

### 5.2. ArcFace vs FaceNet (Triplet Loss) vs CosFace

| Phương pháp | Cách dạy model | Margin ở đâu | Ưu điểm | Nhược điểm |
|-------------|----------------|-------------|---------|-----------|
| **FaceNet (Triplet Loss)** | So bộ ba ảnh: Anchor-Positive-Negative | Khoảng cách Euclidean | Trực quan, metric learning thuần | Mining triplet khó, train tốn công ở quy mô lớn |
| **CosFace** | Normalized softmax + trừ margin trên cosine: `s(cos(θ) - m)` | Miền Cosine | Dễ triển khai, ổn định | Margin nằm ở cosine space, ít "hình học tự nhiên" |
| **ArcFace** ← (Hệ thống đang dùng) | Normalized softmax + cộng margin vào góc: `s·cos(θ + m)` | Miền Angular (Góc) | Ý nghĩa hình học đẹp nhất, hiệu năng rất mạnh | Nhạy cảm với s, m → **dễ Collapse nếu cấu hình sai** |

### 5.3. Tại sao ArcFace nhạy cảm hơn?

Công thức ArcFace biến logit của class đúng (target) từ:
```
logit_target = s · cos(θ_y)
```
thành:
```
logit_target = s · cos(θ_y + m)
```

Vì `cos` là hàm giảm trong `[0, π]`, nên `cos(θ + m) < cos(θ)`. Điều này **ép mạnh** logit target xuống thấp hơn, buộc model phải kéo embedding gần center (W_y) hơn nữa. 

**Khi s quá lớn + m quá lớn + LR quá cao** → Gradient trở nên cực lớn → Model không đủ khả năng hội tụ → Tìm "đường tắt" bằng cách gom tất cả về 1 điểm trên hypersphere → **MODE COLLAPSE**.

### 5.4. Feature Normalization và Hypersphere

Trong code v4, cả embedding `x` lẫn class weights `W` đều được L2-normalize:
```python
cosine = F.linear(F.normalize(input), F.normalize(self.weight))
```
Sau khi normalize:
- Tất cả embedding nằm trên **bề mặt hypersphere** (mặt cầu 512 chiều)
- Việc phân biệt class **chỉ phụ thuộc vào góc** giữa embedding và center
- Scale `s` nhân vào để giữ tín hiệu gradient đủ mạnh cho Softmax

Đây là lý do scale `s` rất quan trọng: quá nhỏ → gradient yếu, học chậm; quá lớn → gradient nổ, collapse.

---

## 6. Kiểm Tra Đối Chiếu: Code v4 Đã Fix Hết Chưa?

| # | Vấn đề | Nguyên nhân gốc | Fix trong v4 | Trạng thái |
|---|--------|----------------|-------------|-----------|
| 1 | LR quá cao (0.7) | Linear scaling rule nhân batch | HEAD_LR cố định = 0.01 | ✅ ĐÃ FIX |
| 2 | Margin gắt ngay từ đầu (m=0.50) | Không có curriculum | Curriculum: 0.20 → 0.50 qua 15 epoch | ✅ ĐÃ FIX |
| 3 | Scale quá lớn (s=64) | Khuếch đại gradient | s = 30.0 | ✅ ĐÃ FIX |
| 4 | Warmup LR nhân chồng | Bug `pg['lr'] *= factor` | Set tuyệt đối: `pg['lr'] = base * factor` | ✅ ĐÃ FIX |
| 5 | Không có collapse detector | Thiếu monitoring | Embedding Spread Monitor (std < 0.05) | ✅ ĐÃ FIX |
| 6 | Gradient bùng nổ | clip_grad_norm = 5.0 | clip_grad_norm = 1.0 | ✅ ĐÃ FIX |
| 7 | Early Stopping dùng Val Loss | Curriculum Margin làm Loss tăng liên tục | Đổi sang Val Accuracy | ✅ ĐÃ FIX |
| 8 | Biến `best_val_loss` còn sót ở dòng in cuối | Đổi tên biến chưa triệt để | Sửa thành `best_val_acc` | ✅ ĐÃ FIX |

---

## 7. Pipeline Tích Hợp ArcFace + MiniFASNet

Khi hệ thống face recognition hoạt động, pipeline xử lý một khuôn mặt như sau:

```
Camera → Detect Face (RetinaFace/SCRFD)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
MiniFASNet          ArcFace v4
(Anti-Spoofing)     (Embedding)
    ↓                   ↓
Thật/Giả?          Vector 512-D
    ↓                   ↓
  [Giả → TỪ CHỐI]   Cosine Similarity
                     với Database
                        ↓
                  Ai? (Threshold ≥ 0.4)
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
           Trùng khớp         Không khớp
           → XÁC NHẬN        → NGƯỜI LẠ
```

**Khi ArcFace bị Collapse:** Bước `Cosine Similarity` luôn trả về > 0.80 cho mọi người → Mọi người đều "trùng khớp" → Hệ thống vô dụng.

**Khi ArcFace v4 hoạt động đúng:** Cosine giữa người A và người B sẽ chỉ khoảng 0.1~0.3 (xa nhau), chỉ khi đúng người A so với ảnh của chính người A thì Cosine mới > 0.5~0.7 → Hệ thống hoạt động chính xác.

---

## 8. Checklist Tích Hợp Sau Khi Train Xong

- [ ] Cell 7 trên Zeppelin báo `HOAN TAT!` với Best Val Acc đủ cao
- [ ] Cell 8 đóng gói `arcface_best_model_v4.onnx` + `anti_spoofing_v2_q.onnx` thành ZIP
- [ ] Tải ZIP từ Cloudflare R2 về máy local
- [ ] Copy `arcface_best_model_v4.onnx` vào `e:\Workspace\detect\models\`
- [ ] Cập nhật `core/config.py`: đổi `ARCFACE_PATH` sang file mới
- [ ] Khởi động lại hệ thống
- [ ] Test: Đưa 2 người khác nhau vào → Cosine < 0.3 (PASS)
- [ ] Test: Đưa cùng 1 người vào → Cosine > 0.5 (PASS)
- [ ] Test: Dùng ảnh in/điện thoại → MiniFASNet chặn (PASS)
