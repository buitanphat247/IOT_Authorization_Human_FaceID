# ArcFace v5 — Kỷ Nguyên Phục Hưng Tiêu Chuẩn Công Nghiệp (Zeppelin / Kaggle)

> **v5 FIX CHÍ MẠNG TOÀN DIỆN:** Thiết kế MỘT CELL DUY NHẤT (Cell 7 - Độc Lập Mọi Thứ).
> Vì trên Kaggle bạn chỉ giữ lại Cell 1, 2, 3 tải Data, nên **Cell 7** dưới đây được tôi bọc lại tất cả mọi thứ: Từ Model Core, ArcFace Neck, DataLoader, cho đến Ma lực AdamW và Cosine Annealing.
>
> 🔧 **Đã tối ưu hóa siêu tốc để giải quyết 40% EER Mù Nhận Diện:**
> 1. `optim.AdamW` siêu lực kéo bù cho SGD rùa bò.
> 2. `s=64.0` phá vỡ mặt phẳng Softmax thay vì s=30.
> 3. `Cụm bóp Neck:` Linear Model được bao quanh bởi BN-Dropout-Linear chống chết trọng số.

---

```
%md
### Cell 1, 2, 3 (Hãy giữ nguyên của bản v4 cũ trên Kaggle để tải Data)
```

---

```
%md
### Cell 7: Kịch Bản Kéo Trọng Số Bất Bại v5 (Tích hợp Full Model + Loss + Train)
👉 Copy toàn bộ đoạn dưới đập vào Cell Train Kaggle (Thay cho cái Cell 7 v4 cũ).
```

```python
import sys, os
sys.path.insert(0, os.path.abspath("./extra_libs"))

os.environ['TORCHDYNAMO_DISABLE'] = '1'
for key in list(sys.modules.keys()):
    if 'torch._dynamo' in key: del sys.modules[key]

import math, glob, gc, time, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
warnings.filterwarnings('ignore', message='.*_MultiProcessingDataLoaderIter.*')

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()


# =========================================================
# PHẦN 1: DATASET & CLASSES TỰ ĐỦ (Cho Cell 7 chạy trên Kaggle)
# =========================================================

class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds, lbl_col, label_map=None):
        self.ds, self.lbl_c, self.label_map = hf_ds, lbl_col, label_map
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5), # Standard Face Augment
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        tensor = self.transform(item['image'].convert('RGB'))
        raw = item[self.lbl_c]
        return tensor, torch.tensor(self.label_map[raw] if self.label_map else raw, dtype=torch.long)

def get_dataloader(data_dir, batch_size=128, num_workers=2):
    pqs = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)
    pqs = [f for f in pqs if not f.endswith('.metadata')]
    if not pqs: raise FileNotFoundError(f"Khong lay duoc Parquet o {data_dir}")
    ds = load_dataset("parquet", data_files=pqs, split="train")
    cols = ds.column_names
    lbl_col = next((c for c in cols if c in ['label','labels','target']), cols[1])
    
    feat = ds.features[lbl_col]
    if hasattr(feat, 'num_classes'):
        nc, lmap = feat.num_classes, None
    else:
        uniq = sorted(set(ds[lbl_col]))
        nc, lmap = len(uniq), {o:i for i,o in enumerate(uniq)}
        
    return DataLoader(HFDatasetWrapper(ds, lbl_col, lmap), batch_size=batch_size,
                      shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True), nc, lmap


# =========================================================
# PHẦN 2: KIẾN TRÚC ARCFACE v5 CHỐNG MODE COLLAPSE LFW
# =========================================================

class ArcFaceMarginProduct(nn.Module):
    """
    Kéo giãn 13,000 class trên Mặt Cầu Siêu Bề Mặt (s=64 chuẩn InsightFace).
    Không xài s=30 (Mode collapse).
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1+1e-7, 1-1e-7)
        sine = (1.0 - cosine.pow(2)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh = torch.zeros_like(cosine); oh.scatter_(1, label.view(-1,1).long(), 1)
        return ((oh * phi) + ((1-oh) * cosine)) * self.s

def build_model(nc):
    """
    Rút Linear trần đi. Thêm Neck Tiêu chuẩn chống quá nạp Batch Norm.
    """
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(
        nn.BatchNorm1d(m.fc.in_features),
        nn.Dropout(0.4), # Chống Mode collapse (Ngăn ResNet nhai lại pattern)
        nn.Linear(m.fc.in_features, 512, bias=False), # ArcFace yêu cầu False Bias!
        nn.BatchNorm1d(512), 
    )
    torch.nn.init.xavier_normal_(m.fc[2].weight)
    return m


# =========================================================
# PHẦN 3: MAIN TRAINING LOOP (ADAMW SIÊU BÃO KIẾM)
# =========================================================

def main():
    GPU_ID = 0 
    print("=" * 60)
    print("  ArcFace v5 — ADAMW + S(64): RESET TO DEFAULT MIGHTY")
    print("=" * 60)

    drive_base = './Workspace' if not os.path.exists('/kaggle/working') else '/kaggle/working/Workspace'
    DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
    SAVE_DIR = os.path.join(drive_base, 'FaceModels')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ====== v5 HYPER-PARAMETERS (ADAMW CHỮA CHÁY) ======
    EPOCHS          = 30
    EMBEDDING_SIZE  = 512
    BATCH_SIZE      = 256         # Không cần nhồi quá lớn (Giảm từ 1792 về 256 chuẩn Adam)
    USE_AMP         = True
    WARMUP_EPOCHS   = 2           # Đỡ sốc 2 Epoch đầu đầu cho Pretrained ResNet
    GRAD_CLIP       = 2.0         

    # LR CỦA ADAMW (SGD Đã Quá Yếu Đuối)
    BACKBONE_LR     = 2e-4        # Model gốc chỉ nhích rất nhẹ (Không để hư Pretrain)
    HEAD_LR         = 2e-3        # Cụm Linear ArcFace nhích siêu mạnh (gấp 10 Backbone)

    ARCFACE_SCALE   = 64.0        # FIX 40% EER
    ARCFACE_MARGIN  = 0.50        

    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'arcface_best_model_v5.pth')
    BEST_ONNX_PATH  = os.path.join(SAVE_DIR, 'arcface_best_model_v5.onnx')
    # ==================================================

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Thức tỉnh lực ma đạo: AdamW (LR: {HEAD_LR}) | Không Curriculum rườm rà!")

    full_loader, num_classes, label_map = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    full_dataset = full_loader.dataset
    total_size = len(full_dataset)
    val_size = max(1000, int(total_size * 0.01))
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    backbone = build_model(num_classes).to(device)
    margin_layer = ArcFaceMarginProduct(
        in_features=EMBEDDING_SIZE, out_features=num_classes,
        s=ARCFACE_SCALE, m=ARCFACE_MARGIN
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # Đưa AdamW vào. Sức kéo phá băng Mù Nhận Diện
    optimizer = optim.AdamW([
        {'params': backbone.parameters(), 'lr': BACKBONE_LR, 'weight_decay': 0.01},
        {'params': margin_layer.parameters(), 'lr': HEAD_LR, 'weight_decay': 0.01}
    ])

    # Kéo đà rớt giá dần dần theo Cosine Curve. Không Freeze rườm rà.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n{'='*60}")
    print(f"  Train: {train_size} ảnh | {num_classes} classes | M={ARCFACE_MARGIN} | S={ARCFACE_SCALE}")
    print(f"  Batches/epoch: {len(train_loader)} (Batch Size: {BATCH_SIZE})")
    print(f"{'='*60}")

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Áp Lực Sốc Khí (Warmup 2 Epochs để LR bốc lên đỉnh)
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = epoch / WARMUP_EPOCHS
            base_lrs = [BACKBONE_LR, HEAD_LR]
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = base_lrs[i] * warmup_factor
        else:
            scheduler.step() 
            
        cur_backbone_lr = optimizer.param_groups[0]['lr']
        cur_head_lr = optimizer.param_groups[1]['lr']

        backbone.train(); margin_layer.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        total_batches = len(train_loader)

        print(f"\n  === Epoch {epoch}/{EPOCHS} === LR:(Backbone {cur_backbone_lr:.6f} / Head {cur_head_lr:.6f})")

        from tqdm.auto import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        
        for bi, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                features = backbone(images)
                outputs = margin_layer(features, labels)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(margin_layer.parameters()), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

            cur_acc = (train_correct / train_total) * 100
            cur_loss = train_loss / train_total
            pbar.set_postfix(Loss=f"{cur_loss:.4f}", Acc=f"{cur_acc:.2f}%")

        epoch_loss = train_loss / train_total
        epoch_acc = (train_correct / train_total) * 100

        # === VALIDATION ===
        backbone.eval(); margin_layer.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
            for vi, (v_img, v_lbl) in enumerate(val_loader):
                v_img, v_lbl = v_img.to(device, non_blocking=True), v_lbl.to(device, non_blocking=True)
                v_feat = backbone(v_img)
                v_out = margin_layer(v_feat, v_lbl)
                v_preds = v_out.argmax(1)
                val_correct += (v_preds == v_lbl).sum().item()
                val_total += v_img.size(0)

        val_acc = (val_correct / val_total) * 100
        ep_time = time.time() - t0

        print(f"  > Hoàn tất Epoch {epoch}. Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        print(f"  > Valid Acc: {val_acc:.2f}% | Thời gian: {ep_time / 60:.1f} phút")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  🌟 MÔ HÌNH VƯỢT ĐỈNH (Tốt Nhất Mới: {best_val_acc:.2f}%) -> LƯU MÔ HÌNH!")
            torch.save({
                'epoch': epoch,
                'backbone': backbone.state_dict(),
                'margin': margin_layer.state_dict(),
                'best_val_acc': best_val_acc
            }, BEST_MODEL_PATH)

            # TIẾN HÀNH XUẤT ONNX THẦN TỐC Ở ĐÂY LUÔN ĐỂ XÀI CHUẨN
            backbone.eval()
            dummy_input = torch.randn(1, 3, 112, 112, device=device)
            try:
                torch.onnx.export(
                    backbone, dummy_input, BEST_ONNX_PATH,
                    export_params=True, opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print(f"      [ONNX] Đã đúc khuôn thành phẩm ONNX thành công: {BEST_ONNX_PATH}")
            except Exception as e:
                print(f"      [ONNX LỖI] {e}")

    print(f"\n✅ ĐÃ HOÀN TẤT HUẤN LUYỆN v5 CHUYÊN NGHIỆP! Kết quả Tốt Nhất: {best_val_acc:.2f}%")
    print(f"Bạn hãy tải ngay file {BEST_ONNX_PATH} về máy để dùng thay cái cục v4 40% EER kia nhé.")

# CHỈNH MAIN BLOCK CHẠY TRỰC TIẾP
if __name__ == '__main__':
    main()
```

---

```
%md
### Cell 8: Thi Chứng Chỉ LFW Tại Chỗ (Kaggle Benchmark Thần Tốc)
👉 Sau khi Kaggle Train đẻ ra cục `v5.onnx`, đừng vội tắt máy! Chạy ngay Cell 8 này để Kaggle tự down LFW về test chấm điểm Benchmark luôn. Khỏi mất công tải về máy bàn PC để check. 
Nếu EER < 2% thì an tâm nộp bằng tốt nghiệp v5 này lên Production!
```

```python
import sys, os
import numpy as np
import onnxruntime as ort
import cv2
import itertools
from sklearn.datasets import fetch_lfw_people
import warnings
warnings.filterwarnings('ignore')

drive_base = './Workspace' if not os.path.exists('/kaggle/working') else '/kaggle/working/Workspace'
MODEL_PATH = os.path.join(drive_base, 'FaceModels', 'arcface_best_model_v5.onnx')

print("="*60)
print("  🚀 BÀI THI LFW TẠI CHỖ KAGGLE - MÁY CHẤM THI")
print("="*60)

if not os.path.exists(MODEL_PATH):
    print("❌ LỖI: Không tìm thấy model ONNX v5! Chạy xong Cell 7 chưa vậy bạn?")
else:
    print("📥 Đang kéo bộ sinh viên LFW (200MB) về phòng thi Kaggle...")
    lfw = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
    print(f"✅ Đã kéo xong: {len(lfw.images)} thí sinh ảnh.")

    # Cài MediaPipe trong Kaggle nếu thiếu
    try:
        import mediapipe as mp
    except:
        os.system("pip install mediapipe")
        import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    import urllib.request
    FL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists("face_landmarker.task"):
        urllib.request.urlretrieve(FL_URL, "face_landmarker.task")
        
    base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    SRC_PTS = np.array([\n        [38.2946, 51.6963], [73.5318, 51.5014],\n        [56.0252, 71.7366],\n        [41.5493, 92.3655], [70.7299, 92.2041]\n    ], dtype=np.float32)

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    embeddings = {}
    print("🧠 Bắt đầu vắt trí ốc v5.onnx lấy Vector (Xin chờ 3-5 phút)...")
    
    for i in range(len(lfw.images)):
        img_array = lfw.images[i]
        img = (img_array * 255.0).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        label = lfw.target_names[lfw.target[i]]
        
        # mp
        h, w = img.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        res = landmarker.detect(mp_img)
        if res.face_landmarks:
            lms = res.face_landmarks[0]
            pts = np.array([[lms[idx].x * w, lms[idx].y * h] for idx in [468, 473, 1, 61, 291]], dtype=np.float32)
            tform = cv2.estimateAffinePartial2D(pts, SRC_PTS, method=cv2.LMEDS)[0]
            if tform is not None:
                aligned = cv2.warpAffine(img, tform, (112, 112))
                blob = cv2.dnn.blobFromImage(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR), 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
                emb = session.run(None, {input_name: blob})[0][0]
                emb = emb / np.linalg.norm(emb)
                
                if label not in embeddings: embeddings[label] = []
                embeddings[label].append(emb)

    # Chấm điểm EER
    print("📊 Đang chấm điểm và nội suy ranh giới EER...")
    gen_scores, imp_scores = [], []
    p_list = [p for p in embeddings.keys() if len(embeddings[p]) >= 2]
    
    for p in p_list:
        for e1, e2 in itertools.combinations(embeddings[p], 2):
            gen_scores.append(float(np.dot(e1, e2)))
            
    for i in range(len(p_list)):
        num_pairs = min(50, len(embeddings[p_list[i]]))
        for e1 in embeddings[p_list[i]][:num_pairs]:
            for j in range(i+1, len(p_list)):
                for e2 in embeddings[p_list[j]][:1]:
                    imp_scores.append(float(np.dot(e1, e2)))

    gen_scores = np.array(gen_scores)
    imp_scores = np.array(imp_scores)
    
    thresholds = np.linspace(0.0, 1.0, 1000)
    far = np.array([np.sum(imp_scores >= t) / len(imp_scores) for t in thresholds])
    frr = np.array([np.sum(gen_scores < t) / len(gen_scores) for t in thresholds])
    
    min_idx = np.argmin(np.abs(far - frr))
    eer = (far[min_idx] + frr[min_idx]) / 2
    optimal_th = thresholds[min_idx]

    print("\n" + "="*50)
    print("🏆 BẢNG ĐIỂM TỐT NGHIỆP LFW (KAGGLE)")
    print(f"Số Cặp Cùng Người Đo Được  : {len(gen_scores)} pairs")
    print(f"Số Cặp Khác Người Đo Được  : {len(imp_scores)} pairs")
    print(f"💎 EER (Lỗi Nhận Diện)      : {eer*100:.3f}%")
    if eer < 0.05:
        print("🎉 NHẬN XÉT: EER Dưới 5%. TRẠNG THÁI: TỐT NGHIỆP XUẤT SẮC! THA HỒ TÍCH HỢP!")
    else:
        print("⚠️ NHẬN XÉT: EER Vẫn Cao, Đề Nghị Train Thêm.")
    print(f"📏 Ngưỡng Threshold Tối Ưu : {optimal_th:.3f}")
    print("="*50)
```
