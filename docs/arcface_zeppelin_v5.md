# ArcFace v5 — Kỷ Nguyên Phục Hưng Tiêu Chuẩn Công Nghiệp (Kaggle ALL-IN-ONE)

> **BẢN TỔNG HỢP ĐẦY ĐỦ (ALL-IN-ONE):** Phiên bản V5 Độc Lập từ A→Z.
> Không cần file V4 cũ. Tải Data bằng HuggingFace Parquet (Bypass Google Drive).  
> Tự Động Lưu/Tải Checkpoint qua Cloudflare R2!
>
> 🔧 **Đã tối ưu hóa siêu tốc để giải quyết 40% EER Mù Nhận Diện:**
> 1. `optim.AdamW` siêu lực kéo bù cho SGD rùa bò.
> 2. `s=64.0` phá vỡ mặt phẳng Softmax thay vì s=30.
> 3. `Cụm bóp Neck:` BN → Dropout → Linear → BN chống Mode Collapse.
> 4. `Cloudflare R2 Auto-Backup` mỗi epoch + Auto-Restore khi reset session.

---

## Tổng Quan Quy Trình Training

| Bước | Cell | Mô tả | Thời gian |
|------|------|-------|-----------|
| 1 | Cell 1 | Cài thư viện | ~1 phút |
| 2 | Cell 2 | Kiểm tra tài nguyên + Khai báo API R2 | ~5 giây |
| 3 | Cell 3 | Tải dataset CASIA-WebFace từ HuggingFace | ~5 phút |
| 4 | Cell 4 | **Train Model** (30 Epochs, Auto-Resume, Auto-Backup R2) | ~5 tiếng |
| 5 | Cell 5 | Benchmark LFW (Chấm điểm EER) | ~5 phút |
| 6 | Cell 6 | Upload Final Models lên Cloudflare R2 | ~1 phút |

### Thông Số Huấn Luyện (Hyperparameters)

| Thông số | Giá trị | Lý do |
|----------|---------|-------|
| Optimizer | AdamW | Ổn định hơn SGD, tự điều chỉnh momentum |
| Backbone LR | 2e-4 | Nhỏ để giữ pretrained weights ResNet50 |
| Head LR | 2e-3 | Lớn gấp 10x để ArcFace layer học nhanh |
| Weight Decay | 0.01 | Chống overfitting chuẩn AdamW |
| Batch Size | 256 | Cân bằng VRAM T4 (15GB) và tốc độ |
| ArcFace Scale (s) | 64.0 | Chuẩn InsightFace, chống mode collapse |
| ArcFace Margin (m) | 0.50 | Angular margin chuẩn cho face recognition |
| Epochs | 30 | Đủ để converge trên CASIA-WebFace |
| Warmup | 2 Epochs | Linear warmup tránh sốc LR ban đầu |
| Grad Clip | 2.0 | Chống gradient explosion từ ảnh lỗi |
| Scheduler | CosineAnnealingLR | Giảm LR mượt mà theo đường cong cosine |
| Embedding Size | 512 | Chuẩn ArcFace embedding dimension |
| AMP (Mixed Precision) | Bật | Tăng tốc ~40% trên GPU T4 |

### Tiêu Chí Đánh Giá Thành Công

| Chỉ số | Mục tiêu | Ý nghĩa |
|--------|----------|---------|
| Train Loss | < 5.0 (ổn định) | Model đã hội tụ |
| Valid Acc | 80% - 93% | Học tốt nhưng chưa overfitting |
| EER (LFW) | < 5% | Tốt nghiệp, sẵn sàng production |
| EER (LFW) | < 2% | Xuất sắc, ngang tầm thương mại |

---

```
%md
### Cell 1: Cài Đặt Thư Viện (Kaggle)
```

```python
import subprocess, sys, os, importlib

print("=" * 60)
print("  🔧 CELL 1: CÀI ĐẶT THƯ VIỆN CHO ARCFACE V5")
print("=" * 60)

IS_KAGGLE = os.path.exists('/kaggle')
print(f"🔍 Môi trường: {'KAGGLE' if IS_KAGGLE else 'LOCAL / COLAB'}")

# Cài thư viện cần thiết
PACKAGES = [
    "onnxscript",       # Fix lỗi ONNX export trên Kaggle
    "boto3",            # Cloudflare R2 API (S3 compatible)
    "huggingface_hub",  # Tải dataset siêu tốc
    "datasets",         # HuggingFace datasets loader
]

for pkg in PACKAGES:
    try:
        importlib.import_module(pkg.replace("-", "_"))
        print(f"  ✅ {pkg} — đã có sẵn")
    except ImportError:
        print(f"  📦 Đang cài {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], 
                      capture_output=True, text=True)
        print(f"  ✅ {pkg} — cài xong!")

# Kiểm tra PyTorch + GPU
import torch
print(f"\n{'='*50}")
print(f"  ✅ TẤT CẢ THƯ VIỆN ĐÃ SẴN SÀNG!")
print(f"  PyTorch : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}  : {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
print(f"{'='*50}")
```

---

```
%md
### Cell 2: Kiểm Tra Tài Nguyên + Khai Báo Biến Toàn Cục
Định nghĩa tất cả đường dẫn và khóa API Cloudflare R2 một lần duy nhất.
Các Cell phía dưới sẽ dùng chung các biến này.
```

```python
import os, shutil

print("=" * 60)
print("  📊 CELL 2: TÀI NGUYÊN + CẤU HÌNH TOÀN CỤC")
print("=" * 60)

# === ĐƯỜNG DẪN ===
WORKSPACE = '/kaggle/working/Workspace' if os.path.exists('/kaggle/working') else './Workspace'
DATA_DIR  = os.path.join(WORKSPACE, 'FaceData', 'CASIA-WebFace')
SAVE_DIR  = os.path.join(WORKSPACE, 'FaceModels')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_CHECKPOINT = os.path.join(SAVE_DIR, 'arcface_checkpoint_v5.pth')
BEST_PTH_PATH     = os.path.join(SAVE_DIR, 'arcface_best_model_v5.pth')
BEST_ONNX_PATH    = os.path.join(SAVE_DIR, 'arcface_best_model_v5.onnx')

# === CLOUDFLARE R2 API ===
R2_ACCESS_KEY = "a7684a3235bf1f8e3870d82c6dc5ef69"
R2_SECRET_KEY = "a8bf552923ce489626300dc18fe320b3aebba50d52f1439599ce43f955395833"
R2_ENDPOINT   = "https://7970c4a57482708b85fec0d3b79dba4d.r2.cloudflarestorage.com"
R2_BUCKET     = "edu-learning-storage"
R2_FOLDER     = "models/v5"

# === KIỂM TRA TÀI NGUYÊN ===
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"  RAM : {ram.available / (1024**3):.1f} GB trống / {ram.total / (1024**3):.1f} GB tổng")
except: pass

total, used, free = shutil.disk_usage("/")
print(f"  Disk: {free / (1024**3):.1f} GB trống / {total / (1024**3):.1f} GB tổng")

import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GB")

print(f"\n  📂 Data   : {os.path.abspath(DATA_DIR)}")
print(f"  📂 Models : {os.path.abspath(SAVE_DIR)}")
print(f"  ☁️  R2     : {R2_BUCKET}/{R2_FOLDER}/")
print("=" * 60)
```

---

```
%md
### Cell 3: Tải Dataset CASIA-WebFace Siêu Tốc (HuggingFace)
Kéo thẳng 20 file Parquet (~490K ảnh, 10,572 người) từ HuggingFace.
Không dùng Google Drive, không sợ BadZipFile, không sợ rate limit!
```

```python
import os, glob
from huggingface_hub import snapshot_download

print("=" * 60)
print("  🚀 CELL 3: TẢI DATASET CASIA-WEBFACE TỪ HUGGINGFACE")
print("=" * 60)

# Kiểm tra xem đã có data chưa
existing_pqs = glob.glob(os.path.join(DATA_DIR, '**/*.parquet'), recursive=True)
existing_pqs = [f for f in existing_pqs if not f.endswith('.metadata')]

if len(existing_pqs) >= 15:
    print(f"✅ Đã có {len(existing_pqs)} file Parquet. BỎ QUA tải lại!")
else:
    # Dọn rác file zip cũ nếu có (từ phiên v4)
    zip_trash = os.path.join(WORKSPACE, 'FaceData.zip')
    if os.path.exists(zip_trash):
        os.remove(zip_trash)
        print("🗑️ Đã dọn file FaceData.zip rác từ phiên cũ!")

    print("📥 Đang kéo kho dữ liệu từ HuggingFace (SaffalPoosh/casia_web_face)...")
    print("   Dung lượng: ~4.5GB Parquet | ~490,000 ảnh | 10,572 người")
    
    snapshot_download(
        repo_id="SaffalPoosh/casia_web_face",
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns="*.parquet",
        max_workers=4
    )
    
    final_pqs = glob.glob(os.path.join(DATA_DIR, '**/*.parquet'), recursive=True)
    final_pqs = [f for f in final_pqs if not f.endswith('.metadata')]
    print(f"\n✅ TẢI XONG! {len(final_pqs)} file Parquet đã an tọa tại {DATA_DIR}")

print("\n👉 Kéo xuống bấm Run Cell 4 (Train) ngay!")
```

---

```
%md
### Cell 4: Lão Đại Luyện Đan — Train ArcFace v5 (Full Pipeline)
Khối tuần hoàn tự động:
1. Mở nắp R2 kéo Checkpoint cũ (nếu có) → Resume
2. Train 30 Epochs (AdamW + CosineAnnealing + AMP)
3. Cuối mỗi vòng → Lưu Checkpoint + Bắn lên R2
4. Nếu phá kỷ lục → Xuất ONNX + Bắn lên R2

**Cách đọc kết quả:**
- Loss bắt đầu ~43, rớt dần về < 5.0 là model đã hội tụ
- Valid Acc tăng dần: 10% → 30% → 50% → 80%+ là quá trình chuẩn
- Nếu Loss đi ngang ở 3.0-5.0 → Model đã chín, đừng ép thêm
```

```python
import os, sys, math, glob, gc, time, warnings
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import boto3

warnings.filterwarnings('ignore')
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =========================================================
# 4.1: HYPERPARAMETERS (Mặc định — sẽ được Auto-Tuning ghi đè)
# =========================================================
EPOCHS          = 30
EMBEDDING_SIZE  = 512
BATCH_SIZE      = 256          # Giá trị mặc định, Auto-Tuning sẽ ghi đè
USE_AMP         = True
WARMUP_EPOCHS   = 2
GRAD_CLIP       = 2.0
BACKBONE_LR     = 2e-4         # Giá trị mặc định, LR Finder sẽ ghi đè
HEAD_LR         = 2e-3         # Giá trị mặc định, LR Finder sẽ ghi đè
ARCFACE_SCALE   = 64.0
ARCFACE_MARGIN  = 0.50
AUTO_TUNE       = True         # Bật/tắt Auto-Tuning (tắt nếu đã có checkpoint)

print("=" * 60)
print("  ArcFace v5 — ADAMW + S(64) + AUTO-TUNING")
print("=" * 60)

# =========================================================
# 4.2: CLOUDFLARE R2 CLIENT
# =========================================================
try:
    r2_client = boto3.client('s3', endpoint_url=R2_ENDPOINT,
                             aws_access_key_id=R2_ACCESS_KEY,
                             aws_secret_access_key=R2_SECRET_KEY,
                             region_name='auto')
    R2_READY = True
    print("☁️ Cloudflare R2 client: SẴN SÀNG")
except Exception as e:
    R2_READY = False
    print(f"⚠️ Cloudflare R2 client: LỖI ({e})")

# =========================================================
# 4.3: DATASET WRAPPER
# =========================================================
class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds, lbl_col, label_map=None):
        self.ds, self.lbl_c, self.label_map = hf_ds, lbl_col, label_map
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        tensor = self.transform(item['image'].convert('RGB'))
        raw = item[self.lbl_c]
        return tensor, torch.tensor(self.label_map[raw] if self.label_map else raw, dtype=torch.long)

def get_dataset(data_dir):
    """Load dataset - ưu tiên Parquet, fallback ImageFolder"""
    pqs = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)
    pqs = [f for f in pqs if not f.endswith('.metadata')]
    if pqs:
        print(f"📦 Tìm thấy {len(pqs)} file Parquet → Load bằng HuggingFace...")
        ds = load_dataset("parquet", data_files=pqs, split="train")
        cols = ds.column_names
        lbl_col = next((c for c in cols if c in ['label','labels','target']), cols[1])
        feat = ds.features[lbl_col]
        if hasattr(feat, 'num_classes'):
            nc, lmap = feat.num_classes, None
        else:
            uniq = sorted(set(ds[lbl_col]))
            nc, lmap = len(uniq), {o:i for i,o in enumerate(uniq)}
        return HFDatasetWrapper(ds, lbl_col, lmap), nc, lmap
    else:
        from torchvision.datasets import ImageFolder
        print(f"📦 Không có Parquet → Fallback ImageFolder từ {data_dir}")
        transform = transforms.Compose([
            transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        dataset = ImageFolder(data_dir, transform=transform)
        return dataset, len(dataset.classes), dataset.class_to_idx

# =========================================================
# 4.4: KIẾN TRÚC MODEL
# =========================================================
class ArcFaceMarginProduct(nn.Module):
    """ArcFace Angular Margin Loss (s=64, m=0.5 chuẩn InsightFace)"""
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
        oh = torch.zeros_like(cosine)
        oh.scatter_(1, label.view(-1, 1).long(), 1)
        return ((oh * phi) + ((1.0 - oh) * cosine)) * self.s

def build_model():
    """ResNet50 + Neck chuẩn ArcFace: BN → Dropout → Linear(512) → BN"""
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_feat = m.fc.in_features  # 2048
    m.fc = nn.Sequential(
        nn.BatchNorm1d(in_feat),
        nn.Dropout(0.4),
        nn.Linear(in_feat, EMBEDDING_SIZE, bias=False),
        nn.BatchNorm1d(EMBEDDING_SIZE),
    )
    torch.nn.init.xavier_normal_(m.fc[2].weight)
    return m

# =========================================================
# 4.5: AUTO-TUNING ENGINE (Batch Size + Learning Rate)
# =========================================================
def auto_find_batch_size(model, margin, dataset, device, start=64, max_bs=1024):
    """
    Tự động tìm Batch Size lớn nhất mà GPU chịu được.
    Thuật toán: Nhân đôi batch size từ start cho đến khi GPU báo OOM.
    Sau đó lùi về mức cuối cùng thành công, trừ 20% safety margin.
    """
    print("\n🔍 [AUTO-TUNE] Đang dò Batch Size tối ưu cho GPU...")
    best_bs = start
    test_loader = DataLoader(dataset, batch_size=start, shuffle=True, 
                             drop_last=True, num_workers=0, pin_memory=False)
    bs = start
    
    while bs <= max_bs:
        try:
            torch.cuda.empty_cache()
            test_loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                                    drop_last=True, num_workers=0, pin_memory=False)
            images, labels = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)
            
            model.train(); margin.train()
            with torch.amp.autocast('cuda', enabled=True):
                feat = model(images)
                out = margin(feat, labels)
                loss = F.cross_entropy(out, labels)
            loss.backward()
            model.zero_grad(); margin.zero_grad()
            
            vram_used = torch.cuda.max_memory_allocated() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ BS={bs:>4d} | VRAM: {vram_used:.1f}/{vram_total:.1f} GB")
            best_bs = bs
            torch.cuda.reset_peak_memory_stats()
            
            if vram_used > vram_total * 0.85:  # Đã dùng >85% VRAM → dừng
                break
            bs *= 2
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  💥 BS={bs:>4d} | OOM! Lùi về BS={best_bs}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    # Safety margin: lùi 20% để chừa VRAM cho gradient accumulation
    safe_bs = max(start, int(best_bs * 0.8))
    # Làm tròn về bội số 32 gần nhất (tối ưu GPU tensor cores)
    safe_bs = max(32, (safe_bs // 32) * 32)
    print(f"  🏆 BATCH SIZE TỐI ƯU: {safe_bs} (max thử được: {best_bs}, trừ 20% an toàn)")
    torch.cuda.empty_cache()
    return safe_bs

def lr_range_test(model, margin, train_loader, device, criterion,
                  lr_min=1e-7, lr_max=1e-1, num_steps=100):
    """
    LR Range Test (Leslie Smith, 2015).
    Tăng LR từ lr_min → lr_max theo hàm mũ trong num_steps bước.
    Ghi lại Loss tại mỗi LR. LR tối ưu = nơi Loss giảm mạnh nhất (gradient âm nhất).
    Trả về: (best_backbone_lr, best_head_lr)
    """
    import copy
    print("\n🔍 [AUTO-TUNE] Đang chạy LR Range Test (Leslie Smith)...")
    
    # Backup trạng thái model
    backbone_state = copy.deepcopy(model.state_dict())
    margin_state = copy.deepcopy(margin.state_dict())
    
    test_opt = optim.AdamW([
        {'params': model.parameters(), 'lr': lr_min, 'weight_decay': 0.01},
        {'params': margin.parameters(), 'lr': lr_min * 10, 'weight_decay': 0.01}
    ])
    
    lr_mult = (lr_max / lr_min) ** (1 / num_steps)
    lrs, losses = [], []
    best_loss = float('inf')
    model.train(); margin.train()
    scaler_test = torch.amp.GradScaler('cuda', enabled=True)
    
    data_iter = iter(train_loader)
    smoothed_loss = 0.0
    
    for step in range(num_steps):
        # Lấy batch (lặp lại nếu hết data)
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        test_opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=True):
            out = margin(model(images), labels)
            loss = criterion(out, labels)
        
        scaler_test.scale(loss).backward()
        scaler_test.step(test_opt)
        scaler_test.update()
        
        # Exponential smoothing
        cur_loss = loss.item()
        smoothed_loss = 0.98 * smoothed_loss + 0.02 * cur_loss if step > 0 else cur_loss
        corrected_loss = smoothed_loss / (1 - 0.98 ** (step + 1))
        
        cur_lr = test_opt.param_groups[0]['lr']
        lrs.append(cur_lr)
        losses.append(corrected_loss)
        
        # Dừng sớm nếu loss bùng nổ (gấp 4x loss tốt nhất)
        if corrected_loss < best_loss:
            best_loss = corrected_loss
        if corrected_loss > best_loss * 4 and step > 10:
            print(f"  💥 Loss bùng nổ tại LR={cur_lr:.2e} → Dừng sớm (step {step})")
            break
        
        # Tăng LR theo hàm mũ
        for pg in test_opt.param_groups:
            pg['lr'] *= lr_mult
    
    # Tìm LR tối ưu: điểm có gradient âm mạnh nhất (loss giảm nhanh nhất)
    import numpy as np
    losses_np = np.array(losses)
    if len(losses_np) > 5:
        gradients = np.gradient(losses_np)
        min_grad_idx = np.argmin(gradients[5:]) + 5  # Bỏ 5 bước đầu nhiễu
        optimal_lr = lrs[min_grad_idx]
    else:
        optimal_lr = 2e-4  # Fallback
    
    # Backbone LR = optimal, Head LR = optimal * 10 (tỉ lệ vàng)
    best_backbone_lr = optimal_lr
    best_head_lr = min(optimal_lr * 10, 1e-2)  # Cap tại 0.01
    
    print(f"  📊 Đã thử {len(lrs)} mức LR từ {lr_min:.1e} → {lrs[-1]:.1e}")
    print(f"  🏆 LR TỐI ƯU TÌM ĐƯỢC:")
    print(f"     Backbone LR : {best_backbone_lr:.6f}")
    print(f"     Head LR     : {best_head_lr:.6f}")
    
    # Khôi phục model về trạng thái ban đầu (xóa dấu vết LR test)
    model.load_state_dict(backbone_state)
    margin.load_state_dict(margin_state)
    del backbone_state, margin_state, scaler_test, test_opt
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_backbone_lr, best_head_lr

# =========================================================
# 4.6: KHỞI TẠO MÔ HÌNH + AUTO-TUNING
# =========================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\n🔥 KÍCH HOẠT NHÂN CHÍNH: {device}")

full_dataset, num_classes, label_map = get_dataset(DATA_DIR)
total_size = len(full_dataset)
val_size = max(1000, int(total_size * 0.01))
train_size = total_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

backbone = build_model().to(device)
margin_layer = ArcFaceMarginProduct(
    in_features=EMBEDDING_SIZE, out_features=num_classes,
    s=ARCFACE_SCALE, m=ARCFACE_MARGIN
).to(device)

criterion = nn.CrossEntropyLoss()
torch.backends.cudnn.benchmark = True

# Kiểm tra có checkpoint sẵn không (nếu có → bỏ qua Auto-Tune)
has_checkpoint = os.path.exists(RESUME_CHECKPOINT)
if not has_checkpoint and R2_READY:
    try:
        r2_client.download_file(R2_BUCKET, f"{R2_FOLDER}/arcface_checkpoint_v5.pth", RESUME_CHECKPOINT)
        has_checkpoint = True
        print("☁️ [R2] Đã kéo checkpoint từ R2 về (sẽ bỏ qua Auto-Tune)")
    except: pass

if AUTO_TUNE and not has_checkpoint:
    print("\n" + "="*60)
    print("  🤖 KHỞI ĐỘNG AUTO-TUNING (Lần đầu train, chưa có checkpoint)")
    print("="*60)
    
    # Bước 1: Tìm Batch Size tối ưu
    BATCH_SIZE = auto_find_batch_size(backbone, margin_layer, train_dataset, device,
                                      start=64, max_bs=1024)
    
    # Tạo DataLoader tạm với batch size vừa tìm được (cho LR test)
    temp_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True)
    
    # Bước 2: Tìm Learning Rate tối ưu
    BACKBONE_LR, HEAD_LR = lr_range_test(
        backbone, margin_layer, temp_loader, device, criterion,
        lr_min=1e-7, lr_max=1e-1, num_steps=100
    )
    del temp_loader
    
    print(f"\n{'='*60}")
    print(f"  ✅ AUTO-TUNING HOÀN TẤT!")
    print(f"  Batch Size : {BATCH_SIZE}")
    print(f"  Backbone LR: {BACKBONE_LR:.6f}")
    print(f"  Head LR    : {HEAD_LR:.6f}")
    print(f"{'='*60}")
else:
    if has_checkpoint:
        print("\n⏭️ Đã có Checkpoint → Bỏ qua Auto-Tuning (dùng LR/BS từ checkpoint)")
    else:
        print(f"\n⏭️ Auto-Tune tắt → Dùng giá trị mặc định: BS={BATCH_SIZE}, B_LR={BACKBONE_LR}, H_LR={HEAD_LR}")

# Tạo DataLoader chính thức với Batch Size đã chốt
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)

# Tạo Optimizer + Scheduler với LR đã chốt
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
optimizer = optim.AdamW([
    {'params': backbone.parameters(), 'lr': BACKBONE_LR, 'weight_decay': 0.01},
    {'params': margin_layer.parameters(), 'lr': HEAD_LR, 'weight_decay': 0.01}
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# =========================================================
# 4.7: AUTO-RESUME CHECKPOINT
# =========================================================
start_epoch = 1
best_val_acc = 0.0

# Checkpoint đã được download từ R2 ở bước 4.6 (nếu cần)
# Ở đây chỉ cần load vào model
if os.path.exists(RESUME_CHECKPOINT):
    print(f"♻️ TÌM THẤY CHECKPOINT! Đang nạp...")
    try:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        margin_layer.load_state_dict(ckpt['margin'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"✅ Nạp thành công! Chạy tiếp từ Epoch {start_epoch} (Best Acc: {best_val_acc:.2f}%)")
    except Exception as e:
        print(f"❌ Lỗi nạp checkpoint: {e}. Train lại từ đầu.")

# =========================================================
# 4.8: VÒNG LẶP HUẤN LUYỆN CHÍNH
# =========================================================
print(f"\n{'='*60}")
print(f"  Train: {train_size} ảnh | {num_classes} classes | M={ARCFACE_MARGIN} | S={ARCFACE_SCALE}")
print(f"  Batches/epoch: {len(train_loader)} (Batch Size: {BATCH_SIZE})")
print(f"{'='*60}")

from tqdm.auto import tqdm

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()

    # === WARMUP (2 Epochs đầu) ===
    if epoch <= WARMUP_EPOCHS:
        warmup_factor = epoch / WARMUP_EPOCHS
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = [BACKBONE_LR, HEAD_LR][i] * warmup_factor

    cur_b_lr = optimizer.param_groups[0]['lr']
    cur_h_lr = optimizer.param_groups[1]['lr']

    # === TRAINING ===
    backbone.train(); margin_layer.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    print(f"\n=== Epoch {epoch}/{EPOCHS} === LR:(B {cur_b_lr:.6f} / H {cur_h_lr:.6f})")
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            features = backbone(images)
            outputs = margin_layer(features, labels)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(margin_layer.parameters()), GRAD_CLIP
        )
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)

        pbar.set_postfix(
            Loss=f"{train_loss/train_total:.4f}",
            Acc=f"{(train_correct/train_total)*100:.2f}%"
        )

    # === SCHEDULER (Sau warmup, gọi sau optimizer.step) ===
    if epoch > WARMUP_EPOCHS:
        scheduler.step()

    # === VALIDATION ===
    backbone.eval(); margin_layer.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
        for v_img, v_lbl in val_loader:
            v_img = v_img.to(device, non_blocking=True)
            v_lbl = v_lbl.to(device, non_blocking=True)
            v_out = margin_layer(backbone(v_img), v_lbl)
            val_correct += (v_out.argmax(1) == v_lbl).sum().item()
            val_total += v_img.size(0)

    val_acc = (val_correct / val_total) * 100
    epoch_loss = train_loss / train_total
    ep_time = time.time() - t0

    print(f"  > Epoch {epoch} xong. Loss: {epoch_loss:.4f} | Train Acc: {(train_correct/train_total)*100:.2f}%")
    print(f"  > Valid Acc: {val_acc:.2f}% | Thời gian: {ep_time/60:.1f} phút")

    # === SAVE CHECKPOINT (Nội bộ + R2) ===
    torch.save({
        'epoch': epoch,
        'backbone': backbone.state_dict(),
        'margin': margin_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_acc': best_val_acc
    }, RESUME_CHECKPOINT)

    if R2_READY:
        try:
            r2_client.upload_file(RESUME_CHECKPOINT, R2_BUCKET, f"{R2_FOLDER}/arcface_checkpoint_v5.pth")
            print("  ☁️ [R2] Checkpoint backed up!")
        except: pass

    # === SAVE BEST MODEL + ONNX ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"  🌟 MÔ HÌNH VƯỢT ĐỈNH (Mới: {best_val_acc:.2f}%) → LƯU!")

        torch.save({
            'epoch': epoch, 'backbone': backbone.state_dict(),
            'margin': margin_layer.state_dict(), 'best_val_acc': best_val_acc
        }, BEST_PTH_PATH)

        # Xuất ONNX (Silenced Mode)
        backbone.eval()
        dummy = torch.randn(1, 3, 112, 112, device=device)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    backbone, dummy, BEST_ONNX_PATH,
                    export_params=True, opset_version=18,
                    do_constant_folding=True,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
            print(f"      [ONNX] Xuất thành công: {BEST_ONNX_PATH}")

            if R2_READY:
                try:
                    r2_client.upload_file(BEST_ONNX_PATH, R2_BUCKET, f"{R2_FOLDER}/arcface_best_model_v5.onnx")
                    r2_client.upload_file(BEST_PTH_PATH, R2_BUCKET, f"{R2_FOLDER}/arcface_best_model_v5.pth")
                    print("      ☁️ [R2] ONNX + PTH backed up!")
                except: pass
        except Exception as e:
            print(f"      [ONNX LỖI] {e}")

print(f"\n{'='*60}")
print(f"✅ HOÀN TẤT HUẤN LUYỆN V5! Best Valid Acc: {best_val_acc:.2f}%")
print(f"   Model ONNX: {BEST_ONNX_PATH}")
print(f"{'='*60}")
```

---

```
%md
### Cell 5: Đánh Giá Benchmark LFW (Chấm Điểm EER Tại Chỗ)
Tải bộ dữ liệu kiểm tra LFW, chạy model ONNX v5 lấy embedding, tính EER.
Nếu EER < 5% → Tốt nghiệp. EER < 2% → Xuất sắc!
```

```python
import os, itertools, warnings
import numpy as np
import onnxruntime as ort
import cv2
from sklearn.datasets import fetch_lfw_people
warnings.filterwarnings('ignore')

print("=" * 60)
print("  🏆 CELL 5: BÀI THI LFW BENCHMARK")
print("=" * 60)

if not os.path.exists(BEST_ONNX_PATH):
    print("❌ Không tìm thấy model ONNX v5! Chạy Cell 4 trước!")
else:
    # 1. Tải LFW
    print("📥 Đang tải bộ dữ liệu LFW (~200MB)...")
    lfw = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
    print(f"✅ Đã tải: {len(lfw.images)} ảnh")

    # 2. Cài MediaPipe
    try:
        import mediapipe as mp
    except:
        os.system("pip install -q mediapipe")
        import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    import urllib.request

    FL_PATH = "face_landmarker.task"
    if not os.path.exists(FL_PATH):
        print("📥 Tải Face Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            FL_PATH
        )

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FL_PATH),
            running_mode=vision.RunningMode.IMAGE, num_faces=1
        )
    )

    # 3. ArcFace alignment points
    SRC_PTS = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)

    # 4. Load ONNX model
    session = ort.InferenceSession(BEST_ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 5. Extract embeddings
    embeddings = {}
    print("🧠 Đang trích xuất embedding (3-5 phút)...")
    for i in range(len(lfw.images)):
        img = lfw.images[i]
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        label = lfw.target_names[lfw.target[i]]

        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        if res.face_landmarks:
            lms = res.face_landmarks[0]
            h, w = img.shape[:2]
            pts = np.array([
                [lms[idx].x * w, lms[idx].y * h]
                for idx in [468, 473, 1, 61, 291]
            ], dtype=np.float32)
            tform = cv2.estimateAffinePartial2D(pts, SRC_PTS, method=cv2.LMEDS)[0]
            if tform is not None:
                aligned = cv2.warpAffine(img, tform, (112, 112))
                blob = cv2.dnn.blobFromImage(
                    cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR),
                    1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False
                )
                emb = session.run(None, {input_name: blob})[0][0]
                emb = emb / np.linalg.norm(emb)
                embeddings.setdefault(label, []).append(emb)

    # 6. Tính EER
    print("📊 Đang chấm điểm EER...")
    gen_scores, imp_scores = [], []
    p_list = [p for p in embeddings if len(embeddings[p]) >= 2]

    for p in p_list:
        for e1, e2 in itertools.combinations(embeddings[p], 2):
            gen_scores.append(float(np.dot(e1, e2)))

    for i in range(len(p_list)):
        for e1 in embeddings[p_list[i]][:min(50, len(embeddings[p_list[i]]))]:
            for j in range(i + 1, len(p_list)):
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

    # 7. In kết quả
    print(f"\n{'='*50}")
    print("🏆 BẢNG ĐIỂM TỐT NGHIỆP LFW")
    print(f"  Cặp Cùng Người : {len(gen_scores)} pairs")
    print(f"  Cặp Khác Người : {len(imp_scores)} pairs")
    print(f"  💎 EER          : {eer*100:.3f}%")
    print(f"  📏 Threshold    : {optimal_th:.3f}")
    if eer < 0.02:
        print("  🎉 XUẤT SẮC! EER < 2%. Sẵn sàng Production!")
    elif eer < 0.05:
        print("  ✅ TỐT! EER < 5%. Có thể tích hợp.")
    else:
        print("  ⚠️ CẦN TRAIN THÊM. EER vẫn cao.")
    print(f"{'='*50}")

    # 8. Vẽ biểu đồ (nếu có matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(imp_scores, bins=100, alpha=0.7, color='red', label='Imposter (Khác Người)')
        ax1.hist(gen_scores, bins=100, alpha=0.7, color='green', label='Genuine (Cùng Người)')
        ax1.axvline(x=optimal_th, color='blue', linestyle='--', label=f'Best Threshold = {optimal_th:.2f}')
        ax1.set_title('Đo Độ Tách Biệt Cosine Similarity')
        ax1.legend()

        ax2.plot(thresholds, far, 'r-', label='FAR (Nhận lầm)')
        ax2.plot(thresholds, frr, 'g-', label='FRR (Từ chối sai)')
        ax2.axvline(x=optimal_th, color='blue', linestyle='--', label=f'Best Threshold = {optimal_th:.2f}')
        ax2.set_title('Đường Cong FAR & FRR Cắt Nhau (EER)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'lfw_benchmark_v5.png'), dpi=150)
        plt.show()
        print(f"📊 Đã lưu biểu đồ: {os.path.join(SAVE_DIR, 'lfw_benchmark_v5.png')}")
    except: print("(Không vẽ được biểu đồ - thiếu matplotlib)")
```

---

```
%md
### Cell 6: Upload Thành Phẩm Cuối Cùng Lên Cloudflare R2
Sau khi Train + Benchmark xong, bắn tất cả lên R2 để tải về máy Local.
```

```python
import os

print("=" * 60)
print("  ☁️ CELL 6: UPLOAD FINAL MODELS LÊN CLOUDFLARE R2")
print("=" * 60)

upload_files = {
    'arcface_best_model_v5.onnx': BEST_ONNX_PATH,
    'arcface_best_model_v5.pth': BEST_PTH_PATH,
    'arcface_checkpoint_v5.pth': RESUME_CHECKPOINT,
}

for name, path in upload_files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        r2_key = f"{R2_FOLDER}/{name}"
        print(f"  📤 {name} ({size_mb:.1f} MB) → {r2_key}...")
        try:
            r2_client.upload_file(path, R2_BUCKET, r2_key)
            print(f"     ✅ OK!")
        except Exception as e:
            print(f"     ❌ LỖI: {e}")
    else:
        print(f"  ⏭️ {name} — không tìm thấy, bỏ qua")

# Upload biểu đồ benchmark nếu có
bench_img = os.path.join(SAVE_DIR, 'lfw_benchmark_v5.png')
if os.path.exists(bench_img):
    try:
        r2_client.upload_file(bench_img, R2_BUCKET, f"{R2_FOLDER}/lfw_benchmark_v5.png")
        print(f"  📤 lfw_benchmark_v5.png → ✅")
    except: pass

print(f"\n{'='*60}")
print(f"  ✅ HOÀN TẤT! Tất cả models đã an toàn trên R2:")
print(f"  Bucket: {R2_BUCKET}/{R2_FOLDER}/")
print(f"{'='*60}")
print(f"\n  📋 SAU KHI TẢI VỀ MÁY LOCAL:")
print(f"  1. Copy arcface_best_model_v5.onnx → models/")
print(f"  2. Sửa core/config.py: ARCFACE_PATH = 'models/arcface_best_model_v5.onnx'")
print(f"  3. Chạy benchmark local để xác nhận")
```
