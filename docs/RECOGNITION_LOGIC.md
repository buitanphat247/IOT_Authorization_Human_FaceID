# Cơ Chế Nhận Diện & Quyết Định "Unknown" (Face Recognition Logic)

Tài liệu này giải thích chi tiết cơ chế hoạt động đằng sau hệ thống nhận diện khuôn mặt v5.12, đặc biệt tập trung vào cách hệ thống đưa ra quyết định **Chấp nhận (Accepted)**, **Từ chối (Rejected/Unknown)**, và **Giả mạo (Spoofing)**.

---

## 1. Tổng Quan Luồng Xử Lý Đa Khung Hình (Multi-Frame Voting)

Hệ thống không đánh giá dựa trên một bức ảnh duy nhất để tăng tối đa độ chính xác và tính bảo mật. Khi bạn nhấn "Quét Nhận Diện", hệ thống sẽ thu thập liên tiếp **5 khung hình (Frames)** trong một khoảng thời gian cực ngắn (~150ms) và thực hiện đánh giá đồng thời:

1. **Phát hiện khuôn mặt:** Dùng Hybrid Detector (SCRFD + MediaPipe) để tìm vị trí.
2. **Đánh giá chất lượng (Face Quality):** Đo độ mờ, độ sáng, tư thế góc nghiêng (6DoF Pose).
3. **Chống giả mạo (Anti-Spoofing):** Dùng MiniFASNet kiểm tra xem đây là người thật hay là hình chụp/thiết bị điện tử.
4. **Trích xuất đặc trưng (Embedding):** Đưa qua ArcFace Model để tạo vector 512 chiều.
5. **Tìm kiếm (FAISS Match):** So sánh vector với cơ sở dữ liệu để tìm ra người giống nhất và **Độ trùng khớp (Cosine Similarity)**.

---

## 2. Hệ Thống Ngưỡng Động (Dynamic Thresholds)

Thuật toán đo lường mức độ khớp nhau bằng giá trị **Cosine Similarity** (Độ Trùng Khớp). Giá trị này chạy từ `0.0` (không giống chút nào) đến `1.0` (giống y hệt hoàn toàn).

Hệ thống của chúng ta áp dụng cơ chế đánh giá **Ngưỡng Động (Dynamic Threshold)**, nghĩa là ngưỡng yêu cầu khó hay dễ sẽ tuỳ thuộc vào chất lượng bức ảnh thực tế của người dùng:

*   **Ngưỡng Cao (High Quality Threshold = 0.38):**
    Áp dụng khi khuôn mặt đủ sáng, rõ nét, không bị nhòe. Nếu **Điểm Trùng Khớp >= 0.38**, hệ thống sẽ `ACCEPTED`.

*   **Ngưỡng Thấp (Low Quality Threshold = 0.34 + Yêu Cầu Chớp Mắt):**
    Áp dụng khi môi trường thiếu sáng, camera bị mờ, khiến điểm số tối đa khó vượt qua 0.38. Lúc này hệ thống **hạ tiêu chuẩn xuống 0.34**, nhưng bù lại **YÊU CẦU** trong 5 khung hình người dùng phải có hành động **chớp mắt (Blink)** để chứng minh liveness bổ sung.

*   **Ngưỡng Từ Chối Bắt Buộc (Reject Threshold = 0.25):**
    Dưới mốc **0.25**, hệ thống chắc chắn 100% đây là "người lạ" hoặc ảnh quá tệ sinh ra nhiễu cao. Kết quả trả về sẽ luôn là **Unknown**.

---

## 3. Tại Sao Lại Sinh Ra Kết Quả "Unknown" (Không Xác Định)?

Khác với các hệ thống phổ thông nhận diện bừa một người giống nhất vào mọi lúc. Hệ thống giới hạn vùng an toàn rất khắt khe để tránh việc **Nhận diện Sai (False Accept)**:

Kết quả sẽ bị trả về trạng thái **Unknown / Thuộc tính chưa xác minh** trong các trường hợp sau:

1. **Người lạ lọt vào camera:** 
   Điểm nhận diện (Cosine Similarity) đối với Database quá thấp (dưới mức `0.38`). Đặc biệt nếu điểm < `0.25`, FAISS Engine lập tức loại bỏ.
   
2. **Bị Mờ, Nhòe nhưng Không Chớp Mắt:** 
   Giả sử môi trường mờ, điểm số đạt mức `0.36`. Mức này thấp hơn Threshold `0.38`, nhưng thuộc diện "Ngưỡng Hạ Thấp (`0.34`)". Dẫu vậy, nếu trong suốt 5 frame bạn **KHÔNG CHỚP MẮT**, hệ thống sẽ nghĩ điểm `0.36` có thể là do thẻ giấy rọi trước camera, nên nó quyết định **Unknown**.
   
3. **Cohort Normalization (Chống Mode Collapse):** 
   Một hiện tượng hiếm xảy ra do AI là một khuôn mặt quá chung chung (ánh sáng chói lóa mất hết nét) có thể sinh ra điểm giống với tất cả mọi người. Hệ thống phát hiện hiện tượng này bằng cách đo Z-Score (so sánh sự phân biệt giữa đám đông). Nếu nó thấy ảnh này "giống quá nhiều người", nó lập tức đánh tụt điểm để đưa về **Unknown**.

---

## 4. Cơ Chế Loại Bỏ Tấn Công Giả Mạo (Spoofing)

Nếu bạn đưa màn hình điện thoại hoặc ảnh in chứa hình của `buitanphat` ra trước Camera:

*   **Không dính lỗi Unknown:** Điểm Cosine Similarity của ảnh giấy so với Database chắc chắn sẽ CỰC KỲ CAO (Vd: `0.80+`) và dễ dàng vượt qua ngưỡng `0.38`. Tại sao? Vì bức ảnh trên điện thoại đúng thật sự là mặt người đó.
*   **Bị đánh trượt ở GATE LIVENESS:** Model MiniFASNet trong luồng đa khung hình phát hiện các đường vân pixel (moiré effect) của màn hình, hay viền khung tranh ảnh. Nó gán nhãn khung hình đó là **Spoof**. 
*   **Final Quyết Định:** Dù Similarity là **0.80+**, tổng phiên (session) bị cắm cờ `SPOOF`, nó đè lên kết quả ACCEPTED, ép đầu ra chuyển sang nhãn hiệu **"Ảnh giả mạo — 📵 REJECTED"**.

---

## 5. Tổng Kết Luồng Đánh Giá (Decision Tree)

1. `Similarity < 0.25` ➔ ❌ Trực tiếp `Unknown`.
2. `Phát hiện Giả mạo > 40% số khung` ➔ 📵 `SPOOF` (Ảnh giả mạo).
3. `Chất lượng cao` + `Similarity >= 0.38` ➔ ✅ `ACCEPTED`.
4. `Chất lượng thấp` + `Có chớp mắt` + `Similarity >= 0.34` ➔ ✅ `ACCEPTED`.
5. `Chất lượng thấp` + `Không chớp mắt` + `Similarity < 0.38` ➔ ❌ `Unknown` (Yêu cầu chớp mắt).
6. Tên Match bị đánh dấu "quá phổ thông" (Z-Score) ➔ Điểm bị gọt phăng ➔ ❌ `Unknown`.

---

## 6. Sơ Đồ Trình Tự Giao Tiếp (Sequence Diagram)

Sơ đồ dưới đây mô tả quá trình từ lúc người dùng ấn nút "Quét Nhận Diện" cho tới khi nhận được kết quả cuối cùng.

```mermaid
sequenceDiagram
    autonumber
    actor User as Người dùng
    participant UI as Giao diện (Web/Socket)
    participant SVC as FaceService (Backend)
    participant DET as HybridDetector
    participant QUAL as QualityAssessor
    participant FAS as MiniFASNet (Anti-Spoof)
    participant ARC as ArcFace (Embedding)
    participant DB as FAISS Engine (Database)

    User->>UI: Bấm "Quét Nhận Diện"
    UI->>UI: Chụp liên tiếp 5 khung hình (Frames)
    UI->>SVC: Gửi POST /api/recognize/multi (5 Frames)
    
    note right of SVC: Bắt đầu Xử lý Đa luồng (Thread-Pool)
    loop Duyệt qua từng Frame (x5)
        SVC->>DET: Trích xuất Face Bounding Box & Landmarks
        DET-->>SVC: Vị trí Khuôn mặt (hoặc Bỏ qua nếu ko thấy)
        
        SVC->>QUAL: Phân tích 6DoF Pose, Độ mờ, EAR (Độ mở mắt)
        QUAL-->>SVC: Điểm chất lượng (q_score) & Trạng thái mắt
        
        SVC->>FAS: Phân tích vân màn hình pixel / mặt nạ in
        FAS-->>SVC: Điểm Real / Fake
        
        SVC->>ARC: Trích xuất Vector 512 chiều
        ARC-->>SVC: Face Embedding
        
        SVC->>DB: Truy vấn Cosine Similarity
        DB-->>SVC: Tên Match & Điểm Raw Score
    end

    SVC->>SVC: Tổng hợp Multi-Frame Voting (Top 70% tốt nhất)
    
    alt Giả mạo (Spoof) >= 40% frames
        SVC-->>UI: Trả về trạng thái SPOOF (📵 Ảnh giả mạo)
    else Tồn tại khuôn mặt hợp lệ
        SVC->>SVC: Kích hoạt Ngưỡng Động (Dynamic Threshold)
        
        alt Score < 0.25 HOẶC Z-Score quá thấp
            SVC-->>UI: Trả về REJECTED / UNKNOWN (❌)
        else Score >= Threshold Động (Yêu cầu chớp mắt nếu mờ)
            SVC->>DB: Ghi log Attendance (Điểm danh thành công)
            SVC-->>UI: Trả về ACCEPTED (✅) kèm Điểm số chi tiết
        else Không đạt Threshold
            SVC-->>UI: Trả về REJECTED / UNKNOWN (❌)
        end
    end
    
    UI-->>User: Hiển thị Bảng kết quả + Thông số chi tiết cấu hình
```

---

## 7. Lưu Đồ Quyết Định Hệ Thống (Decision Flowchart)

```mermaid
graph TD
    Start(("Bắt Đầu")) --> CheckSpoof{"Tỉ lệ khung hình<br>Fake >= 40%?"}
    CheckSpoof -- Có --> Spoof["📵 BÁO ĐỘNG GỈA MẠO<br>Trạng thái: SPOOF"]
    CheckSpoof -- Không --> RawScore{"Raw Score < 0.25?"}
    
    RawScore -- Đúng --> Under25["❌ Trạng thái: UNKNOWN<br>Bị Cấm Hoàn Toàn"]
    RawScore -- Sai (>= 0.25) --> QualityCheck{"Chất lượng Ảnh<br>Trung Bình (q_score)?"}
    
    QualityCheck -- Cao (Rõ, Sáng) --> T38["Áp dụng Threshold: 0.38"]
    QualityCheck -- Thấp (Mờ, Tối) --> CheckBlink{"Có chớp mắt<br>trong 5 khung hình?"}
    
    CheckBlink -- Có --> T34["Áp dụng Threshold: 0.34<br>Yêu cầu Chớp Mắt Pass"]
    CheckBlink -- Không --> T38B["Ép áp dụng Threshold: 0.38<br>Vì không chứng minh liveness"]
    
    T38 --> FinalEval{"Score >= Threshold?"}
    T34 --> FinalEval
    T38B --> FinalEval
    
    FinalEval -- Đạt --> Cohort{"Z-Score Kiểm Tra<br>Mode Collapse"}
    FinalEval -- Không Đạt --> FailMatch["❌ Trạng thái: UNKNOWN<br>Không Đạt Ngưỡng"]
    
    Cohort -- Quá phổ biến --> FailMatch
    Cohort -- Độc nhất minh bạch --> Accept["✅ Trạng thái: ACCEPTED<br>Lưu lịch sử thành công"]
    
    classDef reject fill:#3f1a1a,stroke:#e63946,stroke-width:2px,color:#fff;
    classDef accept fill:#132a13,stroke:#2a9d8f,stroke-width:2px,color:#fff;
    classDef warning fill:#4a3b10,stroke:#e9c46a,stroke-width:2px,color:#fff;
    
    class Spoof,Under25,FailMatch reject;
    class Accept accept;
    class CheckBlink,Cohort warning;
```
