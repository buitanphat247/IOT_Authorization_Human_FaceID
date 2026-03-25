# Báo Cáo Phân Tích Hiện Trạng Đánh Giá Model ArcFace v4 📉

**Ngày đánh giá:** Xem lịch sử chạy Benchmark LFW trên VS Code gốc  
**Kiểu đánh giá:** Trích xuất đặc trưng CSDL LFW (Labeled Faces in the Wild) & Phân lớp Tách biệt 512D Cosine Distance

---

## 1. Dữ Liệu Benchmark Thực Nghiệm Ghi Nhận
Căn cứ các thông số tính toán vòng for duyệt chéo qua 13,000 ảnh LFW trên môi trường Local:
* **Tổng Cặp Ảnh Cùng Một Người (Genuine Pairs):** `238,335 pairs`
* **Tổng Cặp Ảnh Khác Người (Imposter Pairs):** `8,803,217 pairs`
* **Tỉ Lệ Sai Cân Bằng (EER - Equal Error Rate):** 🚨 `39.788%`
* **Ngưỡng Độ Tương Đồng Cắt Đỉnh (Threshold):** `0.359`

---

## 2. Diễn Giải Khoa Học Về Biểu Đồ Và Thông Số
Nhìn vào kết quả đồ thị Histogram (màu Đỏ vs Xanh) được văng xuất ra từ công cụ Notebook, ta nhận định thẳng thắn đây là một hiện tượng **THẢM HỌA**, bởi một AI nhận diện mặt phân biệt người nét căng sẽ luôn chia chùm phân bổ thành Dải Núi Đôi rời rạc: Vách núi Khác Người (Imposter) ở mốc Similarity ~ `0.0` và Đỉnh đồi Cùng Người (Genuine) tập trung quanh ~ `0.6 - 0.9`.

Trái lại trên cục thiết kế nhận diện Facial v4 này, cả 2 tổ hộp (238 ngàn Cặp đúng tính năng và 8.8 Trj Cặp lệch khung) phân thân lấp quấn và chập chung thành một quả núi đỉnh duy nhất là `mốc 0.36`.
Tức là nó phát quang luôn coi bức ảnh A và B bất kì **đều lờ mờ giống nhau 36%** bất kể là giới tính hay khác dân tộc.

Đặc biệt, **EER chạm mức ~40%** (Sác suất nhận người quen thành kẻ gian ngót 40%), việc cắm mô hình này vào các thuật toán Sorting Backend như Cổng Camera sẽ vô hiệu do ngã ngũ tính chất nhận diện hên xui như bốc thăm (Random Guessing). 

---

## 3. Chẩn Đoán Lỗi Thâm Căn Nguyên: "Mode Collapse" (Sập Phân Cụm)
Dựa theo phân chia và đối soát tổng đồ (SYSTEM OVERVIEW), file mạng Nơ ron ONNX v4 này là sản phẩm tàn dư của một quá trình bị rách kiến trúc **Quá Khớp / Sập Cụm Tính Năng (Mode Collapse)** trong lúc tự Fine-Tuning ngày nọ.

* **Cực Trị Learning Rate Lỗi Vượt Thác Cạn:** Thuật toán dò Gradient dò đạo hàm trôi dạc vào chạng hóc của Loss Surface khiến Vector hóa phân tán đứt gãy, dẫn việc tất cả các kết quả trích suất khuôn mặt dồn về 1 tổ hợp trung bình đại Trà giống khuôn rập.
* **Cực Cân Hàm Margin Loss Rút Sai:** Trong thuật Toán huấn luyện ArcFace Loss quen mất không cài hàm Warm-up hay Scaling dẫn tới sự trễ lấp học lẹ đè chập hệ trọng số.

---

## 4. Hành Động Giải Pháp Khắc Phục (Vận Hành Tối Ưu)
Dữ liệu biểu đồ là bằng chứng thép tố cáo Model v4 Không Thể Tiếp Tục Cắm Để Nhận Diện (Duy trí sẽ gây đứt gãy False Acceptance - Dẫn khách vô bậy bạ).

Vậy hệ thống cần phải thay đổi cục diện Tối Ưu Tầng Đáy chứ không thể nâng Ngưỡng Cắt (Margin Rules):
1. **Khôi Phục Mô Hình Nguyên Tử Quốc Dân (Rollback Baseline Model):** Lập tức dừng việc cố nhồi nhét cấu hình nhận `...model_v4.onnx`. Hãy thay thế hệ sinh thái bằng việc trỏ đường dẫn load lại bộ Model tiền Nhiệm Đào Tạo Chuyên Sâu như `w600k_r50.onnx` hoặc `buffalo_l` (Loại có Baseline EER < 1% xịn sò). 
2. **Cấu Hình Tái Fine-Tune (Nếu nhất quyết muốn Train lên v5 mới):**
    * Phải quản lý cực sâu lịch trình học Cosine Annealing Warm Restarts (Tránh xả kẹt Rate lúc mấu chốt mạng ban rễ).
    * Áp dụng Penalty Scale (s = 64) và Margin (m = 0.5) cực cẩn thận trên Loss Hộp Phân Chùm ArcFace.
3. **Thanh Lý Kịch Bản Chấp Niệm:** Ngừng việc dùng hàm Python tăng Threshold bằng lệnh Code vì Vector Core bị chết đặc - Phải chữa gốc bằng File Mô hình khác ngay lập tức.
