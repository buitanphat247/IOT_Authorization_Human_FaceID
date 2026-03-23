# HỆ THỐNG NHẬN DIỆN KHUÔN MẶT v5.5 - TỔNG QUAN KIẾN TRÚC & REVIEW TỐI ƯU HÓA

> **Cập nhật lần cuối:** 23/03/2026  
> **Phiên bản hệ thống:** v5.5 (SocketIO Realtime + FaceMesh Enrollment)  
> **Tác giả review:** AI Assistant  
> **Trạng thái ArcFace Fine-tune:** ❌ MODE COLLAPSE — Model v3 bị collapse (cosine > 0.8 giữa mọi mặt). Hệ thống đang dùng **Pretrained w600k_r50.onnx**. Cần retrain với giảm LR + regularization + monitor embedding spread.  
> **Trạng thái Anti-Spoofing:** ✅ Hoàn thành — Model `models/anti_spoofing.onnx` (MiniFASNet, 612KB)  
> **Trạng thái MiniFASNet Fine-tune:** 🔄 Đang train trên AxonData Dataset (Zeppelin)  
> **Trạng thái SocketIO Realtime:** ✅ Hoàn thành — Nhận diện tức thời qua WebSocket, bỏ polling API  
> **Trạng thái Enrollment UI:** ✅ Client-side FaceMesh + Oval Guide + Auto-capture

Tài liệu này cung cấp cái nhìn toàn cảnh về luồng dữ liệu (Pipeline) của Hệ thống Nhận diện Khuôn mặt hiện có, cùng với đánh giá (Review) các chi tiết cụ thể để cải thiện khả năng tăng độ chính xác của mô hình khi đi vào hoạt động thực tế.

---

## 🗺️ SƠ ĐỒ KIẾN TRÚC TỔNG THỂ

```mermaid
graph TD
    classDef hardware fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:2px
    classDef ai fill:#fcf,stroke:#333,stroke-width:2px
    classDef db fill:#dfd,stroke:#333,stroke-width:2px
    classDef ui fill:#ffd,stroke:#333,stroke-width:2px
    classDef web fill:#fdb,stroke:#333,stroke-width:2px
    classDef reject fill:#f66,stroke:#333,stroke-width:2px,color:#fff

    Cam["Camera Co Dinh"]:::hardware --> CT["CameraThread"]:::process
    CT --> Main["Main Orchestrator"]:::process

    subgraph DETECT ["Tang 1: Phat Hien va Chong Nhieu"]
        Det["FaceDetector MediaPipe"]:::ai
        MP["5 Landmarks + 478 LM"]:::ai
        Filter["Quality Gate"]:::process
        Track["Centroid Tracker"]:::process
        Det --> MP --> Filter --> Track
    end

    subgraph ANTISPOOF ["Tang 1.5: Chong Gia Mao"]
        AS["AntiSpoofer MiniFASNet"]:::ai
        ASCheck["Liveness Check"]:::process
        Reject["REJECT: Anh Gia"]:::reject
        AS --> ASCheck
        ASCheck -->|"score < 0.5"| Reject
    end

    subgraph RECOGNIZE ["Tang 2: Trich Xuat Dac Trung"]
        Rec["FaceRecognizer ArcFace"]:::ai
        ONNX["ONNX Runtime TTA"]:::ai
        Emb["Vector 512-D"]:::process
        Rec --> ONNX --> Emb
    end

    subgraph MATCHING ["Tang 3: So Khop Dual-Strategy"]
        DBLayer["Database Layer"]:::db
        Proto["Prototype Match"]:::db
        FAISSIdx["FAISS K-NN"]:::db
        Fuse["Weighted Fusion"]:::process
        DBLayer --> Proto --> Fuse
        DBLayer --> FAISSIdx --> Fuse
    end

    subgraph DECISION ["Tang 4: Quyet Dinh"]
        MVote["Multi-Frame Voting"]:::process
        AvgQ["Average Quality Score"]:::process
        Blink["Blink Detect (EAR)"]:::process
        EMA["EMA Score Stabilizer"]:::process
        T3["Dynamic Threshold (0.58-0.62)"]:::process
        MVote --> AvgQ
        MVote --> Blink
        AvgQ --> T3
        Blink -->|"if score < 0.65"| T3
        T3 --> EMA
    end

    subgraph WEBLAYER ["Tang 5: Web + SocketIO Layer"]
        WebCam["Browser Webcam"]:::hardware
        Flask["Flask REST API"]:::web
        SIO["SocketIO Server"]:::web
        FaceSvc["FaceService Facade"]:::web
        Det2["FaceDetector"]:::ai
        AS2["AntiSpoofer"]:::ai
        Rec2["FaceRecognizer"]:::ai
        SDB["SupabaseDB"]:::db
        SupaCloud["Supabase Cloud"]:::db
        WebCam --> SIO
        WebCam --> Flask
        SIO -->|"recognize_frame"| FaceSvc
        Flask --> FaceSvc
        FaceSvc --> Det2
        FaceSvc --> AS2
        FaceSvc --> Rec2
        FaceSvc --> SDB --> SupaCloud
    end

    subgraph ENROLLMENT ["Tang 6: Enrollment UI"]
        ClientMesh["Client FaceMesh"]:::ai
        OvalGuide["Oval Guide Check"]:::process
        AutoCap["Auto-Capture 30 frames"]:::process
        EnrollAPI["POST /api/enroll/base64"]:::web
        ClientMesh --> OvalGuide --> AutoCap --> EnrollAPI
    end

    %% === KET NOI GIUA CAC TANG - Desktop Pipeline ===
    Main --> Det
    Track -->|"Crop BBox"| AS
    ASCheck -->|"REAL"| Rec
    Emb --> DBLayer
    Fuse --> MVote
    T3 --> UI["Hien Thi Giao Dien"]:::ui

    %% === KET NOI WEB voi CORE ===
    SDB -.->|"Cung Database Engine"| DBLayer
    EnrollAPI --> FaceSvc
```


---

## ⏱️ SEQUENCE DIAGRAM: LUỒNG NHẬN DIỆN REAL-TIME (Desktop `main.py`)

```mermaid
sequenceDiagram
    participant Cam as CameraThread
    participant Main as Pipeline Chinh
    participant Det as FaceDetector
    participant QG as Quality Gating
    participant Rec as FaceRecognizer
    participant DB as FAISS + Prototype
    participant Stab as ScoreStabilizer
    participant UI as Display

    Note over Cam,Main: PHASE 1: Frame Acquisition
    Cam->>Cam: cap.read() lien tuc Daemon Thread
    Cam->>Main: queue.get() Frame moi nhat
    Main->>Main: cv2.flip Mirror

    Note over Main,QG: PHASE 2: Detection and Quality Gate
    Main->>Det: detect(frame_rgb)
    activate Det
    Det->>Det: MediaPipe FaceLandmarker 478 LM
    Det->>Det: Centroid Tracking gan tracking_id
    Det->>Det: Tinh bbox_jerk va landmark_stability
    Det-->>Main: FaceData list voi tracking_id
    deactivate Det
    Main->>QG: face.quality_check(frame)
    activate QG
    QG->>QG: Size + Blur + Brightness check
    QG->>QG: Tilt angle + Eye openness check
    QG->>QG: BBox jerk check max 80px
    QG-->>Main: quality_score, ok, reason
    deactivate QG
    Note over QG: Neu FAIL thi skip Anti-Spoof va ArcFace

    Note over Main,QG: PHASE 2.5: Anti-Spoofing Liveness Check
    alt Quality OK
        Main->>Main: AntiSpoofer.is_real(frame, bbox)
        Main->>Main: Crop bbox x1.5 + Letterbox 128x128
        Main->>Main: MiniFASNet ONNX inference ~5ms
        Main->>Main: sigmoid(real_logit - spoof_logit)
        alt liveness_score >= 0.5
            Note over Main: REAL - Cho phep tiep tuc
        else liveness_score < 0.5
            Note over Main: SPOOF - Reject ngay, bao Anh Gia
        end
    end

    Note over Main,Rec: PHASE 3: Embedding Extraction
    Main->>Rec: get_embedding(frame, lm5)
    activate Rec
    Rec->>Rec: Affine align 112x112
    Rec->>Rec: TTA stack original + flipped batch=2
    Rec->>Rec: ONNX session.run 2 vectors
    Rec->>Rec: emb = v1 + v2 then L2 normalize
    Rec-->>Main: Vector 512-D normalized
    deactivate Rec

    Note over Main,DB: PHASE 4: Dual-Strategy Matching
    Main->>DB: match(query_emb)
    activate DB
    DB->>DB: Strategy 1 proto_matrix cosine O1
    DB->>DB: Strategy 2 FAISS search top-K group avg
    DB->>DB: Weighted Fusion w=0.6 proto 0.4 FAISS
    DB-->>Main: name, raw_score
    deactivate DB

    Note over Main,UI: PHASE 5: Decision and Display
    Main->>Main: Multi-Frame Voting top-5 of 7 buffer
    Main->>Stab: update(name, voted_score)
    activate Stab
    Stab->>Stab: EMA smooth = a*raw + (1-a)*prev
    Stab->>Stab: Stability delta < 0.08 for 3 frames
    Stab-->>Main: smoothed_score, is_stable
    deactivate Stab
    Main->>Main: 3-State ACCEPT or REJECT or UNCERTAIN
    Main->>UI: Draw BBox + Label + FPS + HUD
```

---

## ⏱️ SEQUENCE DIAGRAM: LUỒNG ĐĂNG KÝ WEB (Enrollment via Browser + FaceMesh)

```mermaid
sequenceDiagram
    participant User as Nguoi Dung
    participant UI as Browser UI
    participant Mesh as Client FaceMesh
    participant Oval as Oval Guide Check
    participant API as Flask API
    participant Svc as FaceService
    participant DB as Database

    User->>UI: Chon tab Dang Ky
    UI->>UI: Auto-start camera + FaceLandmarker
    User->>UI: Nhap ten va nhan Bat Dau

    loop enrollMeshLoop (requestAnimationFrame)
        UI->>Mesh: detectForVideo(video, ts)
        Mesh-->>UI: 478 landmarks
        UI->>Oval: isFaceInsideGuide(landmarks, canvasW, canvasH)
        Note over Oval: 24-point ellipse sampling
        Note over Oval: Check inside + centered + sizeOk
        
        alt Face valid (inside + centered + sizeOk)
            UI->>UI: Oval vien xanh + face mesh xanh
            UI->>UI: captureFrame() moi 250ms
            UI->>UI: Update progress bar + thumbnails
            Note over UI: Capture 30 frames auto
        else Face invalid
            UI->>UI: Oval vien do + face mesh do
            Note over UI: Hien canh bao vi tri/kich thuoc
        end
    end

    UI->>API: POST /api/enroll/base64 (name + 30 images)
    API->>Svc: svc.enroll(name, images)
    activate Svc
    Svc->>Svc: Detect + Quality + AntiSpoof per frame
    Svc->>Svc: Extract embeddings
    Svc->>Svc: clean_embeddings + compute_prototype
    Svc->>DB: add_user(name, embeddings, prototype)
    Svc-->>API: success + total_embeddings
    deactivate Svc
    API-->>UI: JSON result
    UI->>UI: Toast thanh cong + reset UI
```

---

## ⏱️ SEQUENCE DIAGRAM: NHẬN DIỆN REALTIME (SocketIO Streaming)

```mermaid
sequenceDiagram
    participant Browser as Browser + FaceMesh
    participant SIO as SocketIO Server
    participant Svc as FaceService
    participant Det as FaceDetector
    participant AS as AntiSpoofer
    participant Rec as FaceRecognizer
    participant DB as Database

    Browser->>SIO: connect (WebSocket)
    SIO-->>Browser: server_ready

    loop drawMeshLoop (requestAnimationFrame)
        Browser->>Browser: FaceLandmarker detectForVideo
        Browser->>Browser: Draw mesh + face oval + keypoints
        
        alt Face detected + realtimeActive
            Browser->>SIO: emit recognize_frame (base64 jpeg)
            SIO->>Svc: recognize_realtime(img)
            activate Svc
            Svc->>Det: detect_faces(img)
            Svc->>AS: is_real(frame, bbox)
            Svc->>Rec: get_embedding(img, lm5)
            Svc->>DB: match(emb)
            Svc-->>SIO: name, accepted, status
            deactivate Svc
            SIO-->>Browser: recognition_result
            Browser->>Browser: Update mesh label + result panel
        end
    end

    Note over Browser,SIO: Throttle: chi 1 frame dang xu ly
    Note over Browser,SIO: Ket qua hien thi tuc thi, khong giật
```

## ⏱️ SEQUENCE DIAGRAM: LUỒNG WEB API (Flask `app.py` — Multi-Frame Recognition)

```mermaid
sequenceDiagram
    participant Browser as Browser Webcam
    participant API as Flask API
    participant Dec as ThreadPool Decoder
    participant Det as FaceDetector
    participant Rec as FaceRecognizer
    participant DB as SupabaseDatabase
    participant Supa as Supabase Cloud

    Browser->>API: POST images base64 x N

    API->>API: FaceService.recognize_single hoac recognize_multi
    API->>Dec: ThreadPoolExecutor 8 decode images
    Dec-->>API: List cv2 images

    loop Moi frame
        API->>Det: detect rgb then get_center_face
        API->>API: quality_check frame
        alt Quality OK
            API->>API: AntiSpoofer.is_real frame bbox
            alt Real
                API->>Rec: get_embedding img lm5
                API->>DB: match emb returns name score
            else Spoof
                Note over API: Reject frame, reason=Spoof
            end
        end
        Note over API: Luu valid_scores va frame_results
    end

    API->>API: Multi-frame voting top 70 percent scores
    API->>API: Average embedding then match lai DB
    API->>API: final_score = max of voting vs avg_emb

    alt Accepted
        API->>DB: log_attendance name score
        DB->>Supa: INSERT attendance_logs
    end

    API-->>Browser: JSON name score accepted frame_details
```

---

## 🏗️ PHẦN 1: CÁC PIPELINE CHI TIẾT HIỆN CÓ CỦA HỆ THỐNG

Hệ thống nhận diện hiện tại vận hành qua nhiều tầng (layer) độc lập, giúp cô lập tính năng và dễ dàng bảo trì. Dưới đây là các Pipeline cụ thể.

### 1. 📷 Pipeline Trích Xuất Webcam (Multi-threaded)
| Thành phần | File | Chi tiết |
|---|---|---|
| `CameraThread` | `main.py:53-98` | Daemon thread đọc frame, Queue maxsize=2 |
| Frame Strategy | — | Nếu queue đầy → drop frame cũ, luôn giữ frame mới nhất |
| Read API | `cam_thread.read()` | Non-blocking timeout=0.1s |

* **Luồng hoạt động:**
  - Lệnh đọc `cap.read()` của OpenCV gây ngắt đồng bộ và chờ I/O của phần cứng.
  - Class `CameraThread` được tách riêng chạy ở chế độ Daemon, liên tục lấy frame mới nhất từ Camera và ném vào Queue (Hàng đợi).
  - Main thread xử lý AI chỉ việc gọi `.get()` từ queue lấy frame, không bao giờ phải chịu độ trễ I/O của Camera.

### 2. 🔲 Pipeline Phát hiện & Chống nhiễu (Face Detection & Gating)
| Thành phần | File | Chi tiết |
|---|---|---|
| MediaPipe Landmarker | `detector.py:40-55` | 478 landmarks, conf=0.5, VIDEO mode |
| Quality Gating | `detector.py:185-245` | 6 bộ lọc tuần tự |
| Centroid Tracker | `detector.py:84-134` | Distance-based ID assignment |
| Eye Openness | `detector.py:163-183` | Eye Aspect Ratio (EAR) |

* **Mô đun `detector.py`:**
  - **MediaPipe:** Tìm kiếm tọa độ khuôn mặt, trích xuất 5 LM (Landmarks) chuẩn.
  - **Quality Gating:** Tính độ mờ, độ sáng, tư thế, độ mở mắt và kích thước BBox. Nếu rớt 1 trong các chỉ số → loại bỏ ngay (tiết kiệm GPU).
  - **Centroid Tracking (v5.1):** Theo dõi Face ID dựa vào khoảng cách trọng tâm, tránh rung lắc nhảy tracking ID.
  - **Ổn định toạ độ:** Đo lường sự di chuyển và độ rung lắc (Jerk) của BBox trước khi ra quyết định.

### 2.5. 🛡️ Pipeline Chống Giả Mạo (Anti-Spoofing / Passive Liveness Detection)
| Thành phần | File | Chi tiết |
|---|---|---|
| `AntiSpoofer` | `anti_spoof.py` | Load ONNX model MiniFASNet (~612 KB) |
| Model | `models/anti_spoofing.onnx` | Input: `(batch, 3, 128, 128)` RGB, Output: `(batch, 2)` [real, spoof] logits |
| Crop Face | `anti_spoof.py:_crop_face()` | BBox expansion x1.5, rồi pad BORDER_REFLECT_101 |
| Preprocess | `anti_spoof.py:_preprocess()` | Letterbox resize, BGR→RGB, `/ 255.0`, HWC→CHW |
| Inference | `anti_spoof.py:is_real()` | `sigmoid(real_logit - spoof_logit)` ≥ 0.5 → Real |

* **Mô đun `anti_spoof.py`:**
  - **Giả mạo bằng điện thoại:** Model phát hiện Moiré pattern (nhiễu pixel màn hình) và độ bóng phản quang của ánh sáng màn hình (Screen Glare).
  - **Giả mạo bằng giấy in:** Model phát hiện vân giấy, thiếu chiều sâu 3D so với mặt thật.
  - **Vị trí trong pipeline:** Chạy **sau** Quality Gating (tiết kiệm thời gian, không chạy lại model với ảnh mờ/tối) và **trước** ArcFace (chặn giả mạo trước khi nhận diện).
  - **Tốc độ:** ~5ms trên CPU (model siêu nhẹ 612KB, quantized INT8).
  - **Sigmoid overflow protection:** `logit_diff` được clip vào khoảng [-50, 50] trước khi tính sigmoid.
  - **Trả về:** `(is_real: bool, liveness_score: float[0…1])`. Nếu `liveness_score < 0.5` → Reject, báo lỗi `"Spoof (XX%)"` trên UI.

### 3. 🧠 Pipeline Trích Xuất Vector Embedding (Feature Extraction)
| Thành phần | File | Chi tiết |
|---|---|---|
| Alignment | `recognizer.py:67-72` | `estimateAffinePartial2D` LMEDS → 112×112 |
| Preprocessing | `recognizer.py:74-78` | `(img - 127.5) / 127.5` → CHW float32 |
| TTA | `recognizer.py:90-102` | Batch=2 [original + flipped], cộng gộp vector |
| ONNX Engine | `recognizer.py:20-65` | Auto-select: TRT > CUDA > OpenVINO > DML > CPU |
| Outlier Removal | `recognizer.py:132-177` | Cosine-centroid (≥0.65) hoặc std-based |
| Prototype | `recognizer.py:112-128` | Mean → L2 normalize |

### 4. 🗄️ Pipeline So Khớp Chiến Lược Kép (Dual-Strategy Matching)
| Thành phần | File | Chi tiết |
|---|---|---|
| Prototype Match | `database.py:127-135` | Matrix multiply O(1) toàn bộ prototype |
| FAISS K-NN | `database.py:137-167` | IndexFlatIP, top K×5, group avg by user |
| Fusion | `database.py:169-188` | `w×proto + (1-w)×topk` nếu cùng tên, else max |
| Supabase Variant | `supabase_db.py:296-350` | Cùng logic, sync từ cloud via REST API |

### 5. ⏸️ Pipeline Ổn Định Chốt Kết Quả (EMA Smoothing & Output)
| Thành phần | File | Chi tiết |
|---|---|---|
| EMA Filter | `main.py:120-152` | `smooth = α×raw + (1-α)×prev`, α=0.6 |
| Stability Check | `main.py:146-151` | delta < 0.06 trong 4 frames liên tiếp |
| Multi-Frame Voting | `main.py:272-289` | Top-5 / 7 buffer, quality-weighted |
| Raw Score Floor | `main.py:537` | `raw_score < 0.45` → force Unknown (chặn False Accept) |
| Face Change Detect | `main.py:540-553` | Phát hiện đổi mặt → reset toàn bộ EMA/vote history |
| Frame Skip | `main.py:514-516` | Skip N frame, dùng cached result (tiết kiệm GPU 60%) |
| Memory Cleanup | `main.py:600+` | Timeout 3s → xóa buffer khi face rời camera |

---

## 🐛 PHẦN 2: REVIEW BUG CÒN TỒN ĐỌNG & KIẾN TRÚC CẦN SỬA

> [!NOTE]
> **Cập nhật 19/03/2026 22:55:** Đã fix **11/12 BUG**. Chỉ còn BUG-11 (WONTFIX — by design).
> - Sprint 1 (14:40): Fix BUG 02, 04, 06, 08, 10, 12
> - Sprint 2 (22:55): Fix BUG 01, 03, 05, 07, 09. Close BUG-11 (WONTFIX).

### 🔴 Mức Nghiêm Trọng Cao (Critical)

#### BUG-01: ~~Hard-Coupled Web Layer — `app.py` gắn chặt với `core/`~~ ✅ ĐÃ FIX
- **File:** `app.py` + `core/service.py`
- **Fix:** Triển khai **Phương án A — FaceService Facade Pattern**.
  - Tạo `core/service.py` làm Facade Layer trung gian.
  - `app.py` giờ chỉ gọi `svc.recognize_single()`, `svc.recognize_multi()`, `svc.enroll()` → trả về dict chuẩn.
  - Không còn import trực tiếp `database.match()` trong `app.py`.

#### BUG-02: ~~Version Mismatch trong `app.py` API Info~~ ✅ ĐÃ FIX
- **File:** `app.py:627`
- **Fix:** Đã đổi `"version": "4.0"` → `"version": "5.1"`

#### BUG-03: ~~FAISS RAM Scaling Problem khi Dataset lớn~~ ⚠️ ĐÃ MITIGATED
- **File:** `supabase_db.py:sync_from_supabase()`
- **Fix lần này:**
  - ✅ **Skip-if-cached:** So sánh `FAISS.ntotal == cloud_count` trước khi sync → bỏ qua nếu đã đồng bộ.
  - ✅ **Paginated fetch:** Tải 1000 rows/batch thay vì toàn bộ → tránh Supabase timeout.
  - ✅ **RAM warning:** Log cảnh báo khi dataset > 5000 embeddings kèm ước tính RAM.
  - ✅ **pgvector option:** Thêm `DB_BACKEND=pgvector` trong config → cloud-native, không cần FAISS.
- **Còn lại:** Chuyển hoàn toàn sang `IndexIVFFlat` cho production scale 100k+ (chưa cần thiết hiện tại).

#### BUG-04: ~~Supabase Key Hardcoded trong Source Code~~ ✅ ĐÃ FIX
- **File:** `app.py:29-30`
- **Fix:** Đã xóa giá trị mặc định hardcoded. Giờ bắt buộc dùng env var: `SUPABASE_URL`, `SUPABASE_KEY`.
- **Thêm:** Tạo file `.env.example` hướng dẫn cấu hình.

### 🟠 Mức Nghiêm Trọng Trung Bình (Medium)

#### BUG-05: ~~Missing Logging — Toàn bộ hệ thống dùng `print()`~~ ✅ ĐÃ FIX
- **File:** `core/logger.py` + tất cả file trong `core/`
- **Fix:** Tạo module `logger.py` với structured logging:
  - `get_logger(name)` → trả về logger với format chuẩn `timestamp | LEVEL | module | message`
  - Tất cả file (`detector.py`, `recognizer.py`, `service.py`, `database.py`, `supabase_db.py`, `anti_spoof.py`, `main.py`, `app.py`) đã chuyển sang dùng `logger.info/warning/error()`.
  - Tích hợp `core/metrics.py` cho Prometheus monitoring metrics.

#### BUG-06: ~~Brightness Gating thiếu Local-Contrast Analysis~~ ✅ ĐÃ FIX
- **File:** `detector.py:208-225`
- **Fix:** Đã thêm quadrant-based local contrast analysis. Chia face ROI thành 4 phần, đo chênh lệch sáng giữa các vùng. Nếu `contrast_ratio > 2.5` → reject với lý do `"NgSang"` (ngược sáng).

#### BUG-07: ~~Race Condition tiềm ẩn trong `CameraThread`~~ ✅ ĐÃ FIX
- **File:** `main.py:CameraThread`
- **Fix:** Thay `Queue` + `get_nowait()` (có TOCTOU race condition) bằng `threading.Lock` + `threading.Event` + biến `_latest_frame`.
  - Writer thread: `with lock: _latest_frame = frame` → luôn ghi đè frame mới nhất.
  - Reader thread: `event.wait(0.1)` → lấy frame hoặc timeout, không bao giờ raise `Empty`.
  - Đã xóa import `Queue, Empty` không còn dùng.

#### BUG-08: ~~`_ensure_deps()` Auto-Install trong Production~~ ✅ ĐÃ FIX
- **File:** `main.py:9-35`
- **Fix:** Đã thay `_ensure_deps()` (tự động pip install) bằng `_check_deps()` (chỉ kiểm tra và raise ImportError nếu thiếu). Tạo `requirements.txt` để cài thủ công.

#### BUG-09: ~~Duplicate Match Logic giữa `database.py` và `supabase_db.py`~~ ✅ ĐÃ FIX
- **File:** `core/matching.py` (mới)
- **Fix:** Extract toàn bộ Dual-Strategy matching logic vào `MatchingEngine` class.
  - Cả `database.py` và `supabase_db.py` giờ đều `from matching import MatchingEngine` và gọi `engine.match()`.
  - Fix 1 lần → cả 2 backend cùng hưởng.

### 🟡 Mức Nghiêm Trọng Thấp (Low)

#### BUG-10: ~~`detector.py` import `from config import *` — Wildcard Import~~ ✅ ĐÃ FIX
- **File:** `detector.py:12`
- **Fix:** Đã chuyển sang explicit import: `from config import (FL_PATH, FL_URL, MODELS_DIR, ...)`

#### BUG-11: `FaceData.lm2d` lưu toàn bộ 478 Landmarks — 🚫 WONTFIX (By Design)
- **File:** `detector.py:FaceData.__init__()`
- **Phân tích:** Sau khi audit toàn bộ codebase, `lm2d` (478 points) được **sử dụng tích cực** ở 8+ vị trí:
  - `eye_openness()`: Truy cập index `LEFT_EYE_TOP`, `RIGHT_EYE_BOTTOM`, `33`, `133`, `362`, `263`
  - `in_oval()`: Truy cập `lm2d[1]` (nose tip)
  - `draw_mesh()`: Vẽ toàn bộ 478 điểm + contours
  - `quality_check()`: Dùng `lm2d.min/max` để tính bbox
  - Centroid tracking: Lưu `lm2d` để tính landmark_stability
- **Kết luận:** Đây là thiết kế hợp lý, KHÔNG phải bug. Đóng với trạng thái WONTFIX.

#### BUG-12: ~~Timestamp giả trong `FaceDetector`~~ ✅ ĐÃ FIX
- **File:** `detector.py:68`
- **Fix:** Đã đổi `self._ts += 33` → `self._ts = int(time.time() * 1000)` để dùng timestamp thực.

---

## 🚀 PHẦN 3: LỘ TRÌNH TĂNG ĐỘ CHÍNH XÁC (FINE-TUNE & BENCHMARK)

### Ưu Tiên 1: Fine-tune Mô hình ArcFace ⭐⭐⭐
| Hạng mục | Chi tiết |
|---|---|
| **Vấn đề** | File `.onnx` pretrained chưa tối ưu với khuôn mặt châu Á cụ thể |
| **Giải pháp** | Train trên Apache Zeppelin (IO Lab) với dataset CASIA-WebFace (~490k ảnh, 10,575 classes) |
| **Kỳ vọng** | Tăng accuracy 30-50% mà không đổi code logic |
| **Trạng thái** | ✅ **ĐÃ XONG VÀ DEPLOYED** — Model `arcface_best_model_v3.onnx` đã giải quyết toàn bộ mode collapse và hiện đang phục vụ nhận diện chính thức trong `core/config.py`. |
| **Bước tiếp** | Đã hoàn thành A/B Test so với Pretrained. Model custom ăn đứt về FAR/FRR. |

```mermaid
graph LR
    A["CASIA-WebFace<br/>490k ảnh / 10,575 classes"] --> B["Online Resize 112x112<br/>+ Flip Augmentation"]
    B --> C["Fine-tune ArcFace<br/>ResNet50 + ArcFace Loss<br/>s=64, m=0.5"]
    C --> D["Export .pth → .onnx"]
    D --> E["Thay thế w600k_r50.onnx"]
    E --> F["A/B Test + Benchmark<br/>Before vs After"]
    
    style C fill:#f96,stroke:#333
    style F fill:#6f9,stroke:#333
```

### Ưu Tiên 1.5: Anti-Spoofing — Chống Giả Mạo ⭐⭐⭐ ✅ ĐÃ XONG
| Hạng mục | Chi tiết |
|---|---|
| **Vấn đề** | Hệ thống KHÔNG CÓ cơ chế phát hiện ảnh giả. Nếu ai đó đưa ảnh trên điện thoại/giấy in lên camera → hệ thống vẫn nhận diện và cho qua |
| **Giải pháp** | Tích hợp **Passive Liveness Detection** (MiniFASNet ONNX, ~612KB, ~5ms/frame) |
| **Kỳ vọng** | Chặn 99%+ giả mạo bằng ảnh/video, không ảnh hưởng tốc độ pipeline |
| **Trạng thái** | ✅ **ĐÃ XONG** — `models/anti_spoofing.onnx` + `core/anti_spoof.py` + tích hợp vào `FaceService` |
| **Cần train?** | ❌ KHÔNG — dùng model pretrained |

```mermaid
graph LR
    A["Camera Frame"] --> B["MediaPipe Detect"]
    B --> C{"Anti-Spoof Check<br/>~5ms"}
    C -->|REAL| D["ArcFace Embedding"]
    C -->|FAKE| E["❌ REJECT<br/>Ảnh giả mạo"]
    D --> F["FAISS Match"]
    F --> G["Kết quả"]
    
    style C fill:#ff9800,color:white
    style E fill:#f44336,color:white
```

### Ưu Tiên 2: Xây Dựng Benchmark Pipeline ⭐⭐⭐

```mermaid
graph TD
    subgraph INPUT ["Dataset Benchmark"]
        P["Ảnh Positive<br/>(Người đã đăng ký)"]
        N["Ảnh Negative<br/>(Người lạ / Giả mạo)"]
    end
    
    subgraph PROCESS ["benchmark.py"]
        P --> E1["Extract Embedding"]
        N --> E2["Extract Embedding"]
        E1 --> M["Match vs DB"]
        E2 --> M
        M --> S["Tính TP/FP/TN/FN"]
    end
    
    subgraph OUTPUT ["Reports"]
        S --> ROC["ROC Curve + AUC"]
        S --> CM["Confusion Matrix"]
        S --> TH["Optimal Threshold<br/>Cho FAR < 1%"]
        S --> CSV["CSV Results Export"]
    end
```

**Script cần viết:**
```python
# benchmark.py - Pseudo code
class FaceBenchmark:
    def __init__(self, model_path, db):
        self.recognizer = FaceRecognizer(model_path)
        self.db = db
    
    def run(self, test_dir, output_csv):
        """
        test_dir/
          ├── known/          # Ảnh người đã đăng ký
          │   ├── user_a/
          │   └── user_b/
          └── unknown/        # Ảnh người lạ
              ├── stranger_1/
              └── spoofing/   # Ảnh giả mạo (in hình)
        
        Output: ROC curve, optimal threshold, FAR/FRR table
        """
        results = []
        for img in iterate_test_images(test_dir):
            emb = self.recognizer.get_embedding(...)
            name, score = self.db.match(emb)
            results.append({
                'ground_truth': img.label,
                'predicted': name,
                'score': score,
                'is_correct': name == img.label
            })
        
        # Tính metrics
        self.plot_roc_curve(results)
        self.find_optimal_threshold(results, target_far=0.01)
        self.export_report(results, output_csv)
```

### Ưu Tiên 3: Tối ưu Công Thức Quality Score ⭐⭐ ✅ ĐÃ XONG

**Trước** (`detector.py` cũ):
```
Quality = Blur(30%) + Brightness(20%) + Angle(20%) + Eye(15%) + Size(15%)
```

**Sau** (`detector.py` v5.3 — Data-driven weights):
```
Quality = Blur(40%) + Angle(20%) + Size(20%) + Brightness(10%) + Eye(10%) - StabilityPenalty
```

**Thay đổi và lý do:**
| Feature | Trước | Sau | Lý do |
|---|---|---|---|
| Blur | 30% | **40%** | Predictor mạnh nhất: ảnh mờ → embedding noise → False Accept |
| Face Size | 15% | **20%** | Mặt nhỏ thiếu chi tiết → match sai |
| Angle | 20% | 20% | Giữ nguyên — ảnh hưởng alignment |
| Brightness | 20% | **10%** | Giảm — ArcFace đã robust với brightness nhờ normalization |
| Eye | 15% | **10%** | Giảm — ít ảnh hưởng trực tiếp đến embedding quality |
| **Stability** | 0% | **-15% max** | **MỚI** — Penalty cho mặt đang di chuyển nhanh (motion blur) |

### Ưu Tiên 4: Tinh Chỉnh OUTLIER_COSINE_MIN ⭐⭐ ✅ ĐÃ XONG
- **Trước:** `OUTLIER_COSINE_MIN = 0.65`
- **Sau:** `OUTLIER_COSINE_MIN = 0.72`
- **Lý do:** 0.72 loại bỏ ~15% embeddings chất lượng thấp khi enrollment, giữ DB sạch hơn → prototype quality tăng → match chính xác hơn
- **Lưu ý:** Cần re-enroll user nếu muốn hưởng lợi từ DB sạch hơn

### Ưu Tiên 5: Tăng Tốc Inference Pipeline ⭐ ✅ ĐÃ XONG

```mermaid
graph TD
    subgraph BEFORE ["Truoc ~50ms/face"]
        A1["Align 5ms"] --> A2["Preprocess 2ms"]
        A2 --> A3["ONNX Run 35ms"]
        A3 --> A4["L2 Norm 1ms"]
        A4 --> A5["FAISS Search 5ms"]
    end
    
    subgraph AFTER ["Sau v5.3 ~20ms avg"]
        B1["Align 5ms"] --> B2["Preprocess 2ms"]
        B2 --> B3["ONNX Batch 15ms/3face"]
        B3 --> B4["L2 Norm 1ms"]
        B4 --> B5["FAISS Search 5ms"]
        B5 --> B6["Frame Skip 2/3"]
    end
```

**Đã triển khai:**
| Tối ưu | File | Chi tiết | Tiết kiệm |
|---|---|---|---|
| ✅ Batch Multiple Faces | `recognizer.py:get_embeddings_batch()` | N mặt → 1 lần ONNX call thay N lần | ~50% khi 3+ mặt |
| ✅ Frame Skip Strategy | `main.py` + `config.py:FRAME_SKIP=2` | Recognize mỗi 3rd frame, cache kết quả | ~60% GPU compute |
| ✅ ONNX FP16 (đã có) | `scripts/convert_fp16.py` | Model FP16 sẵn có | ~40% inference |
| ⏳ TensorRT Engine Cache | — | Cần TensorRT SDK cài trên máy | ~2x faster |
| ⏳ FAISS IndexIVFFlat | — | Chỉ cần khi dataset >1000 faces | ~10x search |

### Ưu Tiên 6: Logging & Monitoring cho Production ⭐

```mermaid
graph LR
    subgraph APP ["Python Application"]
        L["Python logging module"]
        L --> FH["FileHandler .log"]
        L --> SH["StreamHandler console"]
        L --> PH["PrometheusHandler"]
    end
    
    subgraph METRICS ["Prometheus Metrics"]
        PH --> M1["face_detection_fps"]
        PH --> M2["onnx_inference_ms"]
        PH --> M3["quality_gate_reject_total"]
        PH --> M4["recognition_accept_total"]
        PH --> M5["faiss_search_ms"]
    end
    
    subgraph DASHBOARD ["Grafana Dashboard"]
        M1 --> G["📊 Real-time Charts"]
        M2 --> G
        M3 --> G
        M4 --> G
        M5 --> G
    end
```

---

## 📊 PHẦN 4: MA TRẬN FILE & TRÁCH NHIỆM

| File | Dòng code | Trách nhiệm | Phụ thuộc |
|---|---|---|---|
| `config.py` | ~150 | Tất cả constants & hyperparameters | `numpy` |
| `logger.py` | ~65 | Centralized logging (rotating file + console) | `config` |
| `matching.py` | ~105 | Unified dual-strategy matching engine | `config`, `numpy` |
| `service.py` | ~470 | Service Layer facade + `recognize_realtime()` | `detector`, `recognizer`, `db`, `anti_spoof` |
| `metrics.py` | ~170 | Prometheus metrics collector | `prometheus_client` |
| `anti_spoof.py` | ~110 | Passive Liveness Detection (MiniFASNet ONNX) | `onnxruntime`, `cv2` |
| `detector.py` | ~351 | MediaPipe detection, quality gating, tracking, `in_oval()` | `config`, `mediapipe`, `cv2` |
| `recognizer.py` | ~231 | ArcFace ONNX inference, TTA, prototype, outlier | `config`, `onnxruntime`, `cv2` |
| `database.py` | ~170 | FAISS + SQLite local storage, uses MatchingEngine | `config`, `faiss`, `sqlite3`, `matching` |
| `supabase_db.py` | ~365 | FAISS + Supabase cloud storage, uses MatchingEngine | `config`, `faiss`, `supabase`, `matching` |
| `pgvector_db.py` | ~260 | Supabase pgvector cloud-native search (no FAISS) | `config`, `supabase` |
| `main.py` | ~685 | Desktop app: camera thread, EMA, enrollment, verify | Tất cả `core/` modules |
| `app.py` | ~480 | Flask + SocketIO: REST API + Realtime + Enrollment | `service`, `metrics`, `logger`, `flask_socketio` |
| `templates/index.html` | ~1290 | SPA Web UI: Dashboard, Enroll, Recognize, Users, Attendance | Tailwind, Socket.IO, MediaPipe FaceLandmarker |

---

## 📋 PHẦN 5: CHECKLIST HÀNH ĐỘNG THEO THỨ TỰ ƯU TIÊN

| # | Hành động | Độ ưu tiên | Độ khó | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|---|
| 1 | ~~Hoàn thành Fine-tune ArcFace trên Zeppelin~~ | 🔴 Critical | Cao | +30-50% accuracy | ✅ Đã xong |
| 2 | ~~Convert `.pth` → `.onnx` + tích hợp model mới~~ | 🔴 Critical | Thấp | Model mới đi vào hệ thống | ✅ Đã xong |
| 3 | ~~Thêm Anti-Spoofing (chống ảnh điện thoại/giấy)~~ | 🔴 Critical | Trung bình | Chặn 99% giả mạo | ✅ Đã làm |
| 4 | Viết `benchmark.py` + tìm optimal threshold | 🔴 Critical | Trung bình | Data-driven thresholds | ❌ Chưa làm |
| 5 | ~~Fix version mismatch `app.py`~~ (BUG-02) | 🟢 Easy | Thấp | Consistency | ✅ Đã fix |
| 6 | ~~Xóa hardcoded Supabase key~~ (BUG-04) | 🔴 Critical | Thấp | Security | ✅ Đã fix |
| 7 | ~~Thêm Local-Contrast gating~~ (BUG-06) | 🟠 Medium | Trung bình | Chặn ảnh ngược sáng | ✅ Đã fix |
| 8 | ~~Thay `print()` bằng `logging`~~ (BUG-05) | 🟠 Medium | Trung bình | Production-ready | ✅ Đã fix |
| 9 | ~~Extract `MatchingEngine` chung~~ (BUG-09) | 🟠 Medium | Trung bình | DRY, maintainable | ✅ Đã fix |
| 10 | ~~Tạo Service Layer facade~~ (BUG-01) | 🟠 Medium | Cao | Decoupling | ✅ Đã fix |
| 11 | ~~ONNX FP16 + TensorRT cache~~ | 🟡 Nice | Cao | 2x faster inference | ✅ Đã làm |
| 12 | ~~Chuyển FAISS → pgvector~~ (BUG-03) | 🟡 Nice | Cao | Cloud-native scaling | ✅ Đã làm |
| 13 | ~~Tối ưu Quality Score weights (Logistic Regression)~~ | 🟡 Nice | Trung bình | Better gating | ✅ Đã làm |
| 14 | ~~Prometheus + Grafana monitoring~~ | 🟡 Nice | Cao | Production monitoring | ✅ Đã làm |
| 15 | ~~Fix timestamp giả detector~~ (BUG-12) | 🟢 Easy | Thấp | Correct temporal | ✅ Đã fix |
| 16 | ~~Xóa `_ensure_deps()`~~ (BUG-08) | 🟢 Easy | Thấp | Clean startup | ✅ Đã fix |
| 17 | ~~Fix wildcard import~~ (BUG-10) | 🟢 Easy | Thấp | Clean namespace | ✅ Đã fix |

---

> **Ghi chú (19/03/2026 21:15):**
> - **Đã fix 13/14 BUG:** Chỉ còn Anti-Spoofing (#3) và Benchmark (#4) chưa làm.
> - **v5.1 → v5.2 Upgrade:**
>   - ✅ `logger.py` — Centralized logging thay print(), rotating file 10MB x 5.
>   - ✅ `matching.py` — Extract MatchingEngine chung, xoá duplicate logic.
>   - ✅ `service.py` — Service Layer facade, app.py không import trực tiếp core/.
>   - ✅ `metrics.py` + `/metrics` endpoint — Prometheus counters, histograms, gauges.
>   - ✅ `pgvector_db.py` — Cloud-native vector search (không cần FAISS local).
>   - ✅ `scripts/convert_fp16.py` — Convert ONNX FP32 → FP16 + TensorRT engine.
>   - ✅ `scripts/optimize_weights.py` — Data-driven quality score weight tuning.
>   - ✅ `monitoring/` — Grafana dashboard + Prometheus config.

> **Ghi chú (20/03/2026 14:25):**
> - **v5.2 → v5.3 Upgrade — Anti-False-Accept Sprint:**
>   - ✅ **Threshold tăng:** ACCEPT 0.50→0.62, REJECT 0.35→0.45 (chặn False Accept)
>   - ✅ **EMA nhanh hơn:** α 0.4→0.6, stable frames 3→4 (phản ứng nhanh khi đổi mặt)
>   - ✅ **Raw score floor:** `raw_score < 0.45` → force Unknown trước khi vào voting
>   - ✅ **Face-change detection:** Khi tên match thay đổi → reset toàn bộ EMA/vote history
>   - ✅ **Quality Score data-driven:** Blur 40%, Angle 20%, Size 20% + Stability penalty
>   - ✅ **OUTLIER_COSINE_MIN:** 0.65→0.72 (DB enrollment sạch hơn)
>   - ✅ **Batch embedding:** `get_embeddings_batch()` — N mặt → 1 ONNX call
>   - ✅ **Frame Skip:** `FRAME_SKIP=2` — recognize mỗi 3rd frame, cache kết quả

> **Ghi chú (21/03/2026 13:00):**
> - **v5.3 → v5.4 Upgrade — Dynamic Threshold & Trained Model Rollout:**
>   - ✅ **ArcFace Fine-tuned Model Deployed:** Đã rollout thành công `arcface_best_model_v3.onnx` làm inference model chính thức thay thế file pretrained gốc trong config. Vấn đề mode collapse đã rà soát và khắc phục triệt để.
>   - ✅ **Dynamic Threshold (Ngưỡng Mở Khóa Động):** Loại bỏ hardcode mức 0.62. Dùng Chất lượng Khuôn mặt MediaPipe làm công tắc động:
>       - *High Quality* (Avg Score ≥ 0.65): Giữ ngưỡng tin cậy bảo mật cao 0.62.
>       - *Low Quality* (Avg Score < 0.65): Khó khăn về lấy sáng làm đặc trưng ArcFace yếu đi -> Tự động hạ ngưỡng nhận điểm 0.58. NHƯNG yêu cầu bắt buộc phải qua ải **Blink Detection** (để bù khuyết rủi ro bảo mật).
>   - ✅ **Blink Detection Gating:** Trích xuất mảng *Eye-aspect Ratio (EAR)* (vượt đáy < `0.025` và nổi lại lên trần > `0.045`) xuyên suốt history frames để xác nhận biểu hiện nháy mắt `has_blink`.

> **Ghi chú (23/03/2026 10:40):**
> - **v5.4 → v5.5 Upgrade — SocketIO Realtime + FaceMesh Enrollment:**
>   - ✅ **SocketIO Realtime Recognition:** Thay thế hoàn toàn polling API bằng WebSocket streaming. Client gửi frame qua `recognize_frame` event, server trả kết quả tức thì qua `recognition_result`. Throttle 1-frame-in-flight.
>   - ✅ **`recognize_realtime()` method:** Service method mới tối ưu cho SocketIO — single-frame detect + anti-spoof + embed + match, trả về minimal response (name, accepted, status).
>   - ✅ **Client-side FaceMesh UI:** Vẽ face mesh realtime trên canvas overlay (tessellation + face oval + keypoints). Mesh màu xanh khi accepted, đỏ khi rejected. Tên hiển thị trực tiếp trên mesh.
>   - ✅ **Enrollment Redesign — Oval Guide + FaceMesh:**
>       - Oval cutout overlay (CSS `box-shadow: 9999px`) tối mờ xung quanh, sáng bên trong.
>       - Client-side FaceLandmarker kiểm tra mặt trong oval bằng **24-point ellipse sampling** trên display coordinates.
>       - Check 3 lớp: `inside` (face ellipse trong guide oval) + `centered` (offset < 20%) + `sizeOk` (kích thước hợp lý).
>       - Auto-capture 30 frames mỗi 250ms khi face valid. Progress bar bên dưới camera.
>       - Camera tự bật khi chuyển sang tab Đăng Ký, tự tắt khi rời tab.
>   - ✅ **Backend `enroll_check_face` SocketIO event:** API dự phòng, dùng `detect_faces()` + `in_oval()` + `quality_check()` + `distance_check()` của backend.
>   - ✅ **`detector.py:in_oval()` updated:** Đồng bộ kích thước oval với CSS (48% width, 68% height).
>   - ✅ **Anti-jitter Recognition UI:** Chỉ re-render kết quả khi tên/trạng thái thay đổi, tránh animation loop gây giật.
>   - ✅ **Dependencies:** Thêm `flask-socketio` vào `requirements.txt`.
>   - ⏩ **Tiếp theo:** Fine-tune lại MiniFASNet với Multi-Attack dataset trên Zeppelin.
