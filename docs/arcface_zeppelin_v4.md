# ArcFace v4 — Anti-Collapse Training + MiniFASNet Integration (Zeppelin)

> **v4 FIX:** Sửa toàn bộ lỗi Mode Collapse từ v3. Tích hợp MiniFASNet export.
> Thay đổi chính: LR thấp, Curriculum Margin, Embedding Spread Monitor, Gradient Clip chặt.

---

```
%md
### Cell 1: Cài đặt thư viện (TỰ ĐỘNG — Kaggle / Zeppelin / Colab)

**Trên Kaggle — Lần đầu:** Tự cài thư viện vào `extra_libs` (~2 phút).
**Trên Kaggle — Từ lần 2:** Nếu đã tạo Dataset `my-extra-libs`, add vào Input → instant load!

📋 **Hướng dẫn tạo Dataset (làm 1 lần, dùng mãi mãi):**
1. Chạy Cell 1 lần đầu → cài thư viện vào `/kaggle/working/extra_libs`
2. Bấm **Save Version** → Quick Save
3. Vào Output → **"New Dataset"** → tên `my-extra-libs` → Create
4. Từ nay: **+ Add Input** → tìm `my-extra-libs` → Add → xong!
```

```python
import subprocess, sys, os, importlib

# ============================================================
# TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG
# ============================================================
IS_KAGGLE = os.path.exists('/kaggle')
print(f"🔍 Môi trường: {'KAGGLE' if IS_KAGGLE else 'ZEPPELIN / COLAB / LOCAL'}")

# ============================================================
# TÌM THƯ VIỆN PHỤ TRỢ (3 cấp ưu tiên trên Kaggle)
# ============================================================
EXTRA_LIBS = None

if IS_KAGGLE:
    DATASET_LIBS = '/kaggle/input/my-extra-libs/extra_libs'
    WORKING_LIBS = '/kaggle/working/extra_libs'
    
    if os.path.exists(DATASET_LIBS) and len(os.listdir(DATASET_LIBS)) > 5:
        EXTRA_LIBS = DATASET_LIBS
        print(f"⚡ INSTANT LOAD: Dùng thư viện từ Kaggle Dataset!")
    elif os.path.exists(WORKING_LIBS) and len(os.listdir(WORKING_LIBS)) > 5:
        EXTRA_LIBS = WORKING_LIBS
        print(f"✅ Dùng thư viện từ working dir")
    else:
        EXTRA_LIBS = WORKING_LIBS
        print(f"📦 Lần đầu: Sẽ cài thư viện mới...")
else:
    EXTRA_LIBS = os.path.abspath("./extra_libs")

os.makedirs(EXTRA_LIBS, exist_ok=True)
if EXTRA_LIBS not in sys.path:
    sys.path.insert(0, EXTRA_LIBS)

# ============================================================
# PYTORCH: Kaggle cài sẵn, Zeppelin cần cài thêm
# ============================================================
if IS_KAGGLE:
    print("[OK] Kaggle đã cài sẵn PyTorch. BỎ QUA.")
    import torch
    print(f"  PyTorch: {torch.__version__}")
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"  ✅ TPU DETECTED: {device}")
    except (ImportError, RuntimeError):
        if torch.cuda.is_available():
            print(f"  ✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ CPU ONLY")
else:
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
        capture_output=True, text=True, timeout=30
    )
    torch_ok = False
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 3 and lines[1].strip() == 'True':
            torch_ok = True
            print(f"[OK] PyTorch {lines[0]} | CUDA: True | GPU: {lines[2]}")
    if not torch_ok:
        print("[INFO] Chua co PyTorch CUDA. Bat dau cai dat...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], capture_output=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--default-timeout=1000", "--retries=10",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], timeout=3600)

# ============================================================
# CÀI THƯ VIỆN PHỤ TRỢ (chỉ cài nếu chưa có)
# ============================================================
need_install = []
for pkg in ["gdown", "tqdm", "cv2", "huggingface_hub", "datasets", "psutil"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        need_install.append(pkg.replace("cv2", "opencv-python"))

if need_install:
    print(f"\n📦 Cai {len(need_install)} package vao {EXTRA_LIBS}...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", EXTRA_LIBS,
         *need_install, "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"],
        capture_output=True, text=True)
    if r.returncode == 0: print("[OK] Cai thanh cong!")
    else: print(f"[LOI] {r.stderr[-300:]}")
    importlib.invalidate_caches()
else:
    print("✅ Thu vien phu tro da co san.")

# ============================================================
# TONG KET
# ============================================================
import torch
print(f"\n{'='*50}")
print(f"  ✅ TAT CA THU VIEN DA SAN SANG!")
print(f"  PyTorch : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}  : {torch.cuda.get_device_name(i)}")
try:
    import torch_xla.core.xla_model as xm
    print(f"  TPU     : {xm.xla_device()}")
except: pass
print(f"  Libs    : {EXTRA_LIBS}")
print(f"{'='*50}")

if IS_KAGGLE and 'working' in EXTRA_LIBS:
    print(f"\n💡 TIP: Để lần sau load instant (0 giây):")
    print(f"   1. Bấm 'Save Version' (góc phải trên)")
    print(f"   2. Vào Output → 'New Dataset' → tên: my-extra-libs")
    print(f"   3. Lần sau: + Add Input → my-extra-libs → Done!")
```

---

```
%md
### Cell 2: Kiểm tra tài nguyên + Tạo thư mục
```

```python
import os, psutil, shutil, subprocess

print("==== THONG TIN TAI NGUYEN ====")
ram = psutil.virtual_memory()
print(f"RAM: Trong {ram.available / (1024**3):.2f} GB / Tong {ram.total / (1024**3):.2f} GB")
total, used, free = shutil.disk_usage("/")
print(f"Disk: Trong {free / (1024**3):.2f} GB / Tong {total / (1024**3):.2f} GB")

try:
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.free,memory.used", "--format=csv"]).decode('utf-8')
    print("\nGPU:\n", gpu_info)
except Exception:
    print("\nKhong tim thay nvidia-smi.")

print("\n==== TAO THU MUC ====")
drive_base = './Workspace'
for d in ['FaceModels', 'FaceData/CASIA-WebFace']:
    os.makedirs(os.path.join(drive_base, d), exist_ok=True)
print("Thu muc Dataset:", os.path.abspath(os.path.join(drive_base, 'FaceData/CASIA-WebFace')))
print("Thu muc Model:", os.path.abspath(os.path.join(drive_base, 'FaceModels')))
```

---

```
%md
### Cell 2.5: Tạo file Token xác thực Google Drive
```

```python
import json, os

drive_base = './Workspace'
SERVICE_ACCOUNT_FILE = os.path.join(drive_base, 'service_account.json')

service_account_info = {
  "type": "service_account",
  "project_id": "ai-english-app-488114",
  "private_key_id": "379d2470f9a26492358658ffe30036e0ca663795",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCsQaD9TNBnkbvE\npkQ/l4/ecIh1+GeuO+KqjJ/t8ihYCqR5R8lLh0Ajx+LXC3Z5xr1yI0hegZE+lvMw\nlbeg/LLrCNdyhw5CGmv8M5u08n+tXkeWoUpr3vy3NAA+77ZIU80vTPo3daRcTLxC\nx/OkUGbn0KZx8dTFX9qp2vN7SBj/i/ul45T5oqbV4rm/DSlf+FesVAix0U4mn+gJ\nu71oeUOBqidDw3yPdfaPzNHaGRC6gGNzWj9vwtQnzIgOQvz2uRV8ttDl9RrPxSYB\nNRaMeAnvwDnFZuIERa7FQkiYLY8VtT9yfvKFk7PM8We7RvoKY6Q1MoYp5asFRzkO\nN/w8pc7nAgMBAAECggEADGmcQrRuzg/D0otSnqx6uwIr6yhPvE/naPfCLIDHJHT7\n3htsp2sjFZD5SRqz2lmWXCursteuUi7JbmZTeZ/L+sDD2aZewieqQV2sqjDRP3ps\nOS/9L2G9nyv5mo9meLbFMdPfsfBhvB7xb/R9m4kJqUmxPMO629Ao7xVudyM+Xeen\nnP+UYp6h5zo/ogtXKfTB3S/OPS4Vqa3tGQulJ+Tllu1BRxabge1fGqne7epXQmUN\nCIOH1hLzAxcqoINfgxzYuMRFMBnGXrHPvieRKTk7hS2jrGEao5MFtuTILJ6uzWGA\nDDhfOeoaC7dhMzwFqlmmx7NjRYU7tj86Tomj9vMdxQKBgQDZ/5XqDnC/FQJi3CQC\nkZoLgGSZsDl4uDdxH0+MXxbIeiGDFZHupdFL1KTi2p4jR6LMktUduN/YnLKzya6v\nu7VUecWi9SqpbNz0/nt3v3t4Ealbj97atrWcP8hCXnrIPeeOrkrj+fQJsCnqrP3x\ncbq9nbehHKWEvrMdRySeSSP3hQKBgQDKSMGLd4o8ptIk8cPtFcPr2JSVha6OPblU\nJHeCnlHkxddlMtWqZbu0MEZkzgb+NqZvoGNJXL/Zy97WpLDsJMv7eCQ4I4gzngnp\nnqqWYfUhR/olP5lf0N/TGhb0V4VTukVAWl2Ob1z9d8yrgaEL7+QrtzXHwcq/jTBA\n3ngx3+76ewKBgGDSuA8A4uq7thealxPc/4JkQEpSjvQjPdysf3RlN4VLWV8TUYGh\nfFgl1iY720joJFKgVK+i0SkwT96ykTfUMzGV9EXwhkZB59GYxdXl4jzt+z6DpAOY\nawk++v8fX4FYnWQt3W1sMwuqhrOIqjF2xe46Ark4M/vFh2BVnNoAukzRAoGAC6oE\nVwlulC6+YVj7hjlCEeBsOO77cMJIZHyx03tTl5B86h3zSh8RosNB2+AxcQkvsbIa\n54kMmv9xewkGFqfMh1SXKhGKcgeD5M+8YG2HmAKxfuJa8rZ1oZOYjUBatMhB5AMR\n7/ul7guxZpZT9f9ANfEbjcgz67W1eZgacC3Mdu8CgYAe+/6YMnQPgD6ztWxeNHku\nQT5Zp/Zd3ksClXth0e4HIfCq0xappUpADx0oOM35xNsMkPsSB5RwjnWQX+ARMsUR\nE5QivV4Bhyqd19A/cJ9Khr9ur1toOzZnCe4RrlJJGhzFfijlbUOFTk91XV14HU9s\nywKfEbClqJIF4Y8upGeYWQ==\n-----END PRIVATE KEY-----\n",
  "client_email": "io-lab-bot@ai-english-app-488114.iam.gserviceaccount.com",
  "client_id": "110036153339586722043",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/io-lab-bot%40ai-english-app-488114.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

os.makedirs(drive_base, exist_ok=True)
with open(SERVICE_ACCOUNT_FILE, 'w', encoding='utf-8') as f:
    json.dump(service_account_info, f, indent=2)
print(f"File Token da tao tai: {os.path.abspath(SERVICE_ACCOUNT_FILE)}")
```

---

```
%md
### Cell 3: Tải Data từ Google Drive & TỰ ĐỘNG DỌN RÁC
⚠️ **QUAN TRỌNG TRÊN KAGGLE**: Kaggle chỉ cho tối đa **19.5GB** dung lượng tại `/kaggle/working`. 
Code này được trang bị tính năng **Tự Động Dọn Rác**:
- Tải file `FaceData.zip` (9.5GB) về máy.
- Giải nén ra thư mục (chiếm thêm 4.5GB).
- **Xóa ngay lập tức** file `.zip` để lấy lại 9.5GB dung lượng trống, chống lỗi "Disk is full" (Hết ổ cứng) trong lúc train AI.
```

```python
import sys, os
sys.path.insert(0, os.path.abspath("./extra_libs"))
import gdown, zipfile

drive_base = './Workspace'
DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
MODEL_DIR = os.path.join(drive_base, 'FaceModels')
zip_path = os.path.join(drive_base, 'FaceData.zip')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("==== KHOI TAO DOWNLOAD & DON RAC ====")
FACE_DATA_ZIP_ID = "1DjOS4P5rYa1TWttIsf938jUu6U1sDvQ6"

# Đếm TẤT CẢ file (parquet, jpg, png, bất kỳ) — không chỉ riêng parquet
total_files = 0
for root, dirs, files in os.walk(DATA_DIR):
    total_files += len(files)

is_ready = os.path.exists(os.path.join(DATA_DIR, 'DATA_READY.txt'))

# Nếu đã có data (bất kỳ dạng nào) hoặc đã đánh dấu sẵn sàng → BỎ QUA tải
if is_ready or total_files > 10:
    print(f"✅ Du lieu da co san ({total_files} files). BO QUA download.")
    # Don dep rac neu file zip van con ton tai
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("🗑️ Đã dọn dẹp file FaceData.zip rác từ phiên trước!")
    else:
        print("✨ Tình trạng ổ cứng: Sạch sẽ! (Không phát hiện file zip rác)")
    # Đánh dấu DATA_READY nếu chưa có (phòng lần sau)
    if not is_ready:
        with open(os.path.join(DATA_DIR, 'DATA_READY.txt'), 'w') as f:
            f.write("OK")
elif FACE_DATA_ZIP_ID == "THAY_ID_VAO_DAY":
    print("LOI: Chua thay FACE_DATA_ZIP_ID.")
else:
    if not os.path.exists(zip_path):
        print("Dang tai FaceData.zip tu Google Drive...")
        
        # Ưu tiên Service Account API (không bị chặn rate limit)
        SERVICE_ACCOUNT_FILE = os.path.join(drive_base, 'service_account.json')
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            print("  → Dùng Service Account API (ổn định hơn gdown)")
            import io as _io
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive'])
            svc = build('drive', 'v3', credentials=creds)
            request = svc.files().get_media(fileId=FACE_DATA_ZIP_ID)
            fh = _io.FileIO(zip_path, 'wb')
            dl = MediaIoBaseDownload(fh, request, chunksize=50*1024*1024)
            done = False
            while not done:
                status, done = dl.next_chunk()
                if status:
                    print(f"  Downloading: {int(status.progress()*100)}%")
            fh.close()
            print("  Download hoan tat!")
        else:
            # Fallback: dùng gdown nếu không có Service Account
            print("  → Dùng gdown (có thể bị chặn nếu quá nhiều lượt tải)")
            gdown.download(id=FACE_DATA_ZIP_ID, output=zip_path, quiet=False)
        
    print("Dang giai nen...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    print(f"Giai nen hoan tat. Tong files: {len(os.listdir(DATA_DIR))}")
    
    # TINH NANG DON RAC: Xoa ngay lap tuc sau khi giai nen de giai phong 9.5GB
    try:
        os.remove(zip_path)
        print("🧹 ĐÃ DỌN RÁC: Xóa thành công file FaceData.zip để giải phóng 9.5GB bộ nhớ!")
    except Exception as e:
        print(f"Khong the xoa file zip: {e}")
        
    with open(os.path.join(DATA_DIR, 'DATA_READY.txt'), 'w') as f:
        f.write("OK")

# Tai FaceModels bang API
SERVICE_ACCOUNT_FILE = os.path.join(drive_base, 'service_account.json')
if os.path.exists(SERVICE_ACCOUNT_FILE):
    import io as _io
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive'])
    svc = build('drive', 'v3', credentials=creds)
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')] if os.path.exists(MODEL_DIR) else []
    
    if len(model_files) > 0:
        print(f"\nModel da co san ({model_files}). Bo qua.")
    else:
        print("\nDang tai Face Models tu Google Drive...")
        results = svc.files().list(
            q="'1KOKna_ch1aZd6oxL706mtrblhhuIu5cH' in parents and trashed=false",
            fields="files(id, name)").execute()
        for item in results.get('files', []):
            out_path = os.path.join(MODEL_DIR, item['name'])
            if os.path.exists(out_path): continue
            print(f"  Tai: {item['name']}")
            request = svc.files().get_media(fileId=item['id'])
            fh = _io.FileIO(out_path, 'wb')
            dl = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = dl.next_chunk()
        print("Tai model xong.")
print("\nHoan tat Cell 3. San sang train!")
```

---

```
%md
### Cell 4: Kiểm tra cấu trúc dữ liệu
```

```python
import os, glob

drive_base = './Workspace'
DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
print(f"Kiem tra {os.path.abspath(DATA_DIR)}...")

parquets = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
jpgs = glob.glob(f"{DATA_DIR}/**/*.jpg", recursive=True)
pngs = glob.glob(f"{DATA_DIR}/**/*.png", recursive=True)
folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

print(f"  Parquet: {len(parquets)} | JPG: {len(jpgs)} | PNG: {len(pngs)} | Folders: {len(folders)}")
if parquets: print(f"\nDataset dang Parquet. OK.")
elif jpgs or pngs: print(f"\nDataset dang anh ({len(jpgs)+len(pngs)} files).")
else: print("\nKhong tim thay du lieu!")
```

---

> ⚠️ **Cell 5, 6, 7 cũ (Zeppelin GPU) đã được gộp hết vào Cell 7-KAGGLE bên dưới.**
> Trên Kaggle: Chỉ cần chạy Cell 1 → 2 → 2.5 → 3 → **Cell 7-KAGGLE** → Cell 8.

```
%md
### Cell 5: Model + DataLoader (v4 — Anti-Collapse)
⚠️ THAY ĐỔI QUAN TRỌNG so với v3:
- ArcFace margin bắt đầu m=0.20, tăng dần lên 0.50 (Curriculum)
- Scale s=30 thay vì s=64 (giảm gradient magnitude)
- Thêm Embedding Spread Monitor
```

```python
import sys, os
sys.path.insert(0, os.path.abspath("./extra_libs"))
import math, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from torch.utils.data import DataLoader

# ================= DATASET =================
class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds, lbl_col, label_map=None):
        self.ds = hf_ds
        self.lbl_c = lbl_col
        self.label_map = label_map
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image'].convert('RGB')
        tensor = self.transform(image)
        raw_label = item[self.lbl_c]
        mapped_label = self.label_map[raw_label] if self.label_map is not None else raw_label
        return tensor, torch.tensor(mapped_label, dtype=torch.long)

def get_dataloader(data_dir, batch_size=64, num_workers=0):
    parquet_files = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)
    parquet_files = [f for f in parquet_files if not f.endswith('.metadata')]
    if not parquet_files:
        raise FileNotFoundError(f"Khong co parquet o {data_dir}")
    print(f"Loading {len(parquet_files)} parquet tu {data_dir}...")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    cols = ds.column_names
    lbl_col = next((c for c in cols if c in ['label', 'labels', 'target']), cols[1] if len(cols)>1 else cols[-1])
    feature = ds.features[lbl_col]
    if hasattr(feature, 'num_classes'):
        num_classes = feature.num_classes
        label_map = None
        print(f"ClassFeature: {num_classes} classes.")
    else:
        unique_labels = sorted(list(set(ds[lbl_col])))
        num_classes = len(unique_labels)
        label_map = {orig: i for i, orig in enumerate(unique_labels)}
        print(f"Tu tao mapping: {num_classes} classes.")
    dataset = HFDatasetWrapper(ds, lbl_col, label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=num_workers, pin_memory=True)
    return loader, int(num_classes), label_map

# ================= ARCFACE v4 — CURRICULUM MARGIN =================
class ArcFaceMarginProduct(nn.Module):
    """ArcFace with Curriculum Margin to prevent collapse.
    
    v4 FIX: 
    - s=30 (thay vì 64) → giảm gradient magnitude
    - m bắt đầu 0.20, tăng dần lên target_m theo epoch
    - Clamp cosine chặt hơn
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.20, 
                 target_m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.target_m = target_m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self._update_trig()

    def _update_trig(self):
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def set_margin(self, m):
        """Curriculum: tăng margin dần theo epoch."""
        self.m = min(m, self.target_m)
        self._update_trig()

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

def build_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 512)
    return model

# ================= EMBEDDING SPREAD MONITOR =================
def check_embedding_spread(features, threshold=0.05):
    """Kiểm tra embedding có bị collapse không.
    
    Nếu std(pairwise_cosine) < threshold → COLLAPSE DETECTED.
    Returns: (spread_std, is_collapsed)
    """
    with torch.no_grad():
        normed = F.normalize(features, dim=1)
        # Lấy sample nhỏ để tính nhanh (max 256 samples)
        if normed.size(0) > 256:
            idx = torch.randperm(normed.size(0))[:256]
            normed = normed[idx]
        sim_matrix = normed @ normed.T
        # Lấy phần tam giác trên (bỏ diagonal)
        mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
        pairwise_sims = sim_matrix[mask]
        spread_std = pairwise_sims.std().item()
        spread_mean = pairwise_sims.mean().item()
        is_collapsed = spread_std < threshold
        return spread_std, spread_mean, is_collapsed

print("Cell 5 OK. Model + DataLoader + AntiCollapse san sang.")
```

---

```
%md
### Cell 6: Kiểm tra nhanh (30 steps)
```

```python
import os, torch

print("=" * 60)
print("KIEM TRA NHANH — 30 STEPS")
print("=" * 60)

drive_base = './Workspace'
DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
_loader, _nc, _ = get_dataloader(DATA_DIR, batch_size=64, num_workers=0)
device = torch.device('cuda:1')
_imgs, _lbls = next(iter(_loader))
print(f"Shape: {_imgs.shape}, Labels: {_lbls[:10].tolist()}")

_bb = build_model(_nc).to(device)
_mg = ArcFaceMarginProduct(in_features=512, out_features=_nc, s=30.0, m=0.20).to(device)
_cr = torch.nn.CrossEntropyLoss()
_op = torch.optim.SGD(list(_bb.parameters())+list(_mg.parameters()), lr=0.001)
_imgs, _lbls = _imgs.to(device), _lbls.to(device)

print("\nTrain thu...")
for _i in range(30):
    _op.zero_grad()
    _feat = _bb(_imgs)
    _out = _mg(_feat, _lbls)
    _loss = _cr(_out, _lbls)
    _loss.backward()
    torch.nn.utils.clip_grad_norm_(list(_bb.parameters())+list(_mg.parameters()), 1.0)
    _op.step()
    if _i % 10 == 0:
        _acc = (_out.argmax(1)==_lbls).float().mean().item()
        _std, _mean, _collapsed = check_embedding_spread(_feat)
        status = "COLLAPSE!" if _collapsed else "OK"
        print(f"Step {_i:3d} | Loss: {_loss.item():.4f} | Acc: {_acc*100:.1f}% | Spread: std={_std:.4f} mean={_mean:.4f} [{status}]")

_final = (_out.argmax(1)==_lbls).float().mean().item()
_std, _mean, _collapsed = check_embedding_spread(_feat)
print(f"\nFinal: Acc={_final*100:.1f}% | Spread std={_std:.4f} | {'PASS' if not _collapsed else 'FAIL - COLLAPSE!'}")
del _bb, _mg, _cr, _op, _loader
if torch.cuda.is_available(): torch.cuda.empty_cache()
```

---

```
%md
### Cell 7: Training Loop v4 (Anti-Collapse + AMP)

⚠️ **THAY ĐỔI CHỐNG COLLAPSE so với v3:**
1. ✅ LR thấp: head=0.01, backbone=1e-4 (CỐ ĐỊNH, không scale theo batch)
2. ✅ Curriculum Margin: m tăng từ 0.20→0.50 dần qua epoch
3. ✅ Scale s=30 thay vì s=64
4. ✅ Gradient clip max_norm=1.0 thay vì 5.0
5. ✅ Warmup LR tuyến tính đúng cách (set absolute, không nhân chồng)
6. ✅ Embedding Spread Monitor — tự dừng nếu phát hiện collapse
7. ✅ Validation bằng cosine similarity thay vì chỉ Acc
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

# ===== RE-DEFINE (Cell 7 tu du, chay doc lap) =====
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

def get_dataloader(data_dir, batch_size=64, num_workers=0):
    pqs = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)
    pqs = [f for f in pqs if not f.endswith('.metadata')]
    if not pqs: raise FileNotFoundError(f"Khong co parquet o {data_dir}")
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

class ArcFaceMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.20, target_m=0.50):
        super().__init__()
        self.s, self.m, self.target_m = s, m, target_m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._update()
    def _update(self):
        self.cos_m = math.cos(self.m); self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m); self.mm = math.sin(math.pi - self.m) * self.m
    def set_margin(self, m):
        self.m = min(m, self.target_m); self._update()
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1+1e-7, 1-1e-7)
        sine = (1.0 - cosine.pow(2)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh = torch.zeros_like(cosine); oh.scatter_(1, label.view(-1,1).long(), 1)
        return ((oh * phi) + ((1-oh) * cosine)) * self.s

def build_model(nc):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 512)
    return m

def check_spread(features, threshold=0.05):
    with torch.no_grad():
        n = F.normalize(features, dim=1)
        if n.size(0) > 256: n = n[torch.randperm(n.size(0))[:256]]
        sim = n @ n.T
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        ps = sim[mask]
        return ps.std().item(), ps.mean().item(), ps.std().item() < threshold

# ===== MAIN TRAINING =====
def main():
    GPU_ID = 1
    print("=" * 60)
    print("  ArcFace v4 — ANTI-COLLAPSE TRAINING")
    print("=" * 60)

    drive_base = './Workspace'
    DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
    SAVE_DIR = os.path.join(drive_base, 'FaceModels')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ====== v4 HYPER-PARAMETERS (CHỐNG COLLAPSE) ======
    EPOCHS          = 30
    EMBEDDING_SIZE  = 512
    NUM_WORKERS     = 0
    BATCH_SIZE      = 1792        # Giữ nguyên batch size
    USE_AMP         = True
    FREEZE_EPOCHS   = 3
    WARMUP_EPOCHS   = 5           # Tăng warmup từ 2 → 5
    GRAD_CLIP       = 1.0         # FIX: Giảm từ 5.0 → 1.0
    PATIENCE        = 10

    # FIX #1: LR CỐ ĐỊNH, THẤP — không scale theo batch
    HEAD_LR         = 0.01        # FIX: Cố định 0.01 (v3 bị 0.7!)
    BACKBONE_LR     = 1e-4        # FIX: Cố định 1e-4 (v3 bị 0.007!)

    # FIX #2: Curriculum Margin — tăng dần từ 0.20 → 0.50
    MARGIN_START    = 0.20
    MARGIN_END      = 0.50
    MARGIN_WARMUP   = 15          # Epoch đạt max margin

    # FIX #3: Scale s=30 thay vì s=64
    ARCFACE_SCALE   = 30.0

    CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'arcface_checkpoint_v4.pth')
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'arcface_best_model_v4.pth')
    # ==================================================

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{GPU_ID}')
    print(f"Device: {device} — {torch.cuda.get_device_name(GPU_ID)}")
    print(f"LR: head={HEAD_LR} backbone={BACKBONE_LR} (CO DINH)")
    print(f"Margin: {MARGIN_START} -> {MARGIN_END} (curriculum {MARGIN_WARMUP} epochs)")
    print(f"Scale: {ARCFACE_SCALE} | Grad clip: {GRAD_CLIP}")

    full_loader, num_classes, label_map = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    full_dataset = full_loader.dataset
    total_size = len(full_dataset)
    val_size = max(1000, int(total_size * 0.01))
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    backbone = build_model(num_classes).to(device)
    margin_layer = ArcFaceMarginProduct(
        in_features=EMBEDDING_SIZE, out_features=num_classes,
        s=ARCFACE_SCALE, m=MARGIN_START, target_m=MARGIN_END
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # FIX #4: Optimizer với LR cố định thấp
    optimizer = optim.SGD([
        {'params': backbone.parameters(), 'lr': BACKBONE_LR, 'weight_decay': 5e-4},
        {'params': margin_layer.parameters(), 'lr': HEAD_LR, 'weight_decay': 5e-4}
    ], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

    def freeze_backbone(model):
        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
        print("  Backbone FROZEN")

    def unfreeze_backbone(model):
        for p in model.parameters(): p.requires_grad = True
        print("  Backbone UNFROZEN")

    start_epoch, best_val_acc, best_epoch, trigger_times = 1, 0.0, 0, 0

    if os.path.exists(CHECKPOINT_PATH):
        print("Tim thay checkpoint v4, khoi phuc...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        margin_layer.load_state_dict(ckpt['margin'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and USE_AMP: scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        trigger_times = ckpt.get('trigger_times', 0)
        print(f"  Resume tu Epoch {start_epoch} | Best Val Acc: {best_val_acc*100:.2f}%")
        if start_epoch > FREEZE_EPOCHS + 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch - FREEZE_EPOCHS - 1):
                    scheduler.step()

    print(f"\n{'='*60}")
    print(f"  Train: {train_size} anh | {num_classes} classes | batch={BATCH_SIZE}")
    print(f"  Val: {val_size} | Batches/epoch: {len(train_loader)}")
    print(f"{'='*60}")

    collapse_count = 0  # Đếm số epoch liên tiếp bị collapse

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        # FIX #2: Curriculum Margin — tăng dần
        if epoch <= MARGIN_WARMUP:
            curr_m = MARGIN_START + (MARGIN_END - MARGIN_START) * (epoch / MARGIN_WARMUP)
        else:
            curr_m = MARGIN_END
        margin_layer.set_margin(curr_m)

        # FIX #5: Warmup LR tuyến tính ĐÚNG CÁCH (set absolute, không nhân chồng)
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = epoch / WARMUP_EPOCHS
            base_lrs = [BACKBONE_LR, HEAD_LR]
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = base_lrs[i] * warmup_factor

        if epoch <= FREEZE_EPOCHS:
            if epoch == start_epoch: freeze_backbone(backbone)
        elif epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone(backbone)

        backbone.train(); margin_layer.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        epoch_spread_std, epoch_spread_mean = 0.0, 0.0
        spread_samples = 0

        phase_str = "FROZEN" if epoch <= FREEZE_EPOCHS else "FULL"
        total_batches = len(train_loader)
        print(f"\n  === Epoch {epoch}/{EPOCHS} [{phase_str}] margin={curr_m:.3f} === {total_batches} batches")
        last_print = 0

        for bi, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                features = backbone(images)
                output = margin_layer(features, labels)
                loss = criterion(output, labels)
                with torch.no_grad():
                    raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s

            train_correct += (raw_logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # FIX #6: Gradient clip chặt hơn
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(margin_layer.parameters()),
                max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Spread monitor (mỗi 50 batch)
            if bi % 50 == 0:
                s_std, s_mean, _ = check_spread(features.float())
                epoch_spread_std += s_std
                epoch_spread_mean += s_mean
                spread_samples += 1

            now = time.time()
            if now - last_print >= 30.0 or bi == total_batches - 1:
                last_print = now
                pct = (bi+1)/total_batches*100
                avg_l = train_loss/(bi+1)
                acc = train_correct/train_total*100 if train_total > 0 else 0
                elapsed = now - t0
                speed = elapsed/(bi+1)
                eta_s = speed*(total_batches-bi-1)
                vram = torch.cuda.memory_reserved(GPU_ID)/(1024**3)
                done = int(20*(bi+1)/total_batches)
                bar = '#'*done + '.'*(20-done)
                em, es = int(elapsed//60), int(elapsed%60)
                rm, rs = int(eta_s//60), int(eta_s%60)
                print(f"  E{epoch}/{EPOCHS}: {pct:3.0f}%|{bar}| {bi+1}/{total_batches} [{em:02d}:{es:02d}<{rm:02d}:{rs:02d}] Loss:{avg_l:.3f} Acc:{acc:.1f}% VRAM:{vram:.1f}G")

        if epoch > FREEZE_EPOCHS:
            scheduler.step()

        # VALIDATION
        backbone.eval(); margin_layer.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_cosine_sims = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    features = backbone(images)
                    output = margin_layer(features, labels)
                    loss = criterion(output, labels)
                    raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s
                val_correct += (raw_logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                # Cosine sim monitoring
                s_std, s_mean, _ = check_spread(features.float())
                val_cosine_sims.append((s_std, s_mean))

        avg_tl = train_loss / len(train_loader)
        avg_ta = train_correct / train_total
        avg_vl = val_loss / len(val_loader)
        avg_va = val_correct / val_total if val_total > 0 else 0
        elapsed = time.time() - t0

        # Spread summary
        avg_spread_std = epoch_spread_std / max(spread_samples, 1)
        avg_spread_mean = epoch_spread_mean / max(spread_samples, 1)
        val_spread_std = sum(s[0] for s in val_cosine_sims) / max(len(val_cosine_sims), 1)
        is_collapsed = avg_spread_std < 0.05

        collapse_tag = " ⚠️ COLLAPSE!" if is_collapsed else " ✅"

        print(f"\n  ┌────────────── KET QUA EPOCH {epoch} ──────────────┐")
        print(f"  │ Train Loss: {avg_tl:.4f}  | Train Acc: {avg_ta*100:6.2f}%  │")
        print(f"  │ Val   Loss: {avg_vl:.4f}  | Val   Acc: {avg_va*100:6.2f}%  │")
        print(f"  │ Margin: {curr_m:.3f} | Scale: {ARCFACE_SCALE}              │")
        print(f"  │ Spread: std={avg_spread_std:.4f} mean={avg_spread_mean:.4f}{collapse_tag} │")
        print(f"  │ Time: {int(elapsed//60)}m{int(elapsed%60):02d}s                              │")
        print(f"  │ LR: bb={optimizer.param_groups[0]['lr']:.6f} head={optimizer.param_groups[1]['lr']:.6f} │")
        print(f"  └{'─'*49}┘")

        # FIX #7: Tự động dừng nếu collapse liên tiếp
        if is_collapsed:
            collapse_count += 1
            print(f"  ⚠️ COLLAPSE DETECTED! ({collapse_count}/3)")
            if collapse_count >= 3:
                print(f"\n  🛑 ABORT: Model bi collapse 3 epoch lien tiep. Dung train.")
                print(f"  → Giam LR hoac tang WARMUP_EPOCHS roi chay lai.")
                break
        else:
            collapse_count = 0

        # FIX #8: Early Stopping dựa trên Val ACCURACY (không phải Loss)
        # Vì Curriculum Margin làm Loss luôn tăng, nên dùng Loss sẽ bị dừng oan
        if avg_va > best_val_acc:
            best_val_acc = avg_va; best_epoch = epoch; trigger_times = 0
            torch.save(backbone.state_dict(), BEST_MODEL_PATH)
            print(f"  ★ [NEW BEST] Val Acc: {avg_va*100:.2f}% -> {BEST_MODEL_PATH}")
        else:
            trigger_times += 1
            print(f"  (No improvement x{trigger_times}/{PATIENCE})")

        torch.save({
            'epoch': epoch, 'backbone': backbone.state_dict(),
            'margin': margin_layer.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'best_val_acc': best_val_acc,
            'best_epoch': best_epoch, 'trigger_times': trigger_times,
            'num_classes': num_classes, 'batch_size': BATCH_SIZE,
        }, CHECKPOINT_PATH)

        if trigger_times >= PATIENCE:
            print(f"\n  Early Stopping tai Epoch {epoch}.")
            break

    print(f"\n{'='*55}")
    print(f"  HOAN TAT! Best Epoch: {best_epoch} | Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"  Model: {os.path.abspath(BEST_MODEL_PATH)}")
    print(f"{'='*55}")

main()
```

---

```
%md
### Cell 7-KAGGLE: Training Loop v4 — TỰ ĐỘNG NHẬN DIỆN PHẦN CỨNG

⚠️ **DÙNG CELL NÀY TRÊN KAGGLE — TỰ CHẠY ĐÚNG DÙ TPU, GPU HAY CPU**

Tự động phát hiện phần cứng:
1. ✅ Nếu có TPU → Dùng `torch_xla` + `xm.optimizer_step()` + bfloat16 tự nhiên
2. ✅ Nếu có GPU (P100/T4) → Dùng CUDA + AMP + GradScaler
3. ✅ Nếu chỉ có CPU → Chạy bình thường (chậm nhưng vẫn đúng)
4. ✅ BATCH_SIZE tự điều chỉnh: TPU=128, GPU=768, CPU=32
5. ✅ Checkpoint tương thích chéo: Train trên TPU → Resume trên GPU (và ngược lại)
6. ✅ Auto-backup checkpoint lên Google Drive sau mỗi epoch
```

```python
import subprocess, sys, os
# Cài thư viện nếu thiếu (bỏ qua nếu đã có)
try:
    import datasets, huggingface_hub
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'datasets', 'huggingface_hub'])

# Tắt torch dynamo (tránh xung đột trên TPU) — cách an toàn
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import math, glob, gc, time, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
warnings.filterwarnings('ignore')

gc.collect()

# ============================================================
# TỰ ĐỘNG NHẬN DIỆN PHẦN CỨNG: TPU → GPU → CPU
# ============================================================
HW_MODE = 'cpu'  # Mặc định
xm, pl_module = None, None

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl_module
    device = xm.xla_device()
    HW_MODE = 'tpu'
    print(f"✅ TPU DETECTED: {device}")
except (ImportError, RuntimeError):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        HW_MODE = 'gpu'
        NUM_GPUS = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
        for i in range(NUM_GPUS):
            print(f"  ✅ GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/(1024**3):.1f}GB)")
        if NUM_GPUS > 1:
            print(f"  🔥 {NUM_GPUS} GPUs — Sẽ dùng DataParallel để tăng tốc!")
    else:
        device = torch.device('cpu')
        NUM_GPUS = 0
        print("⚠️ CPU ONLY — Training se rat cham!")

# ============================================================
# TỰ ĐỘNG TÌM BATCH_SIZE TỐI ƯU
# ============================================================
USE_AMP = (HW_MODE == 'gpu')  # AMP chỉ dùng cho GPU CUDA

def auto_find_batch_size(hw_mode, dev):
    """Tự dò tìm batch_size lớn nhất mà GPU chịu được"""
    if hw_mode == 'tpu':
        print("  TPU: Dùng batch=128 (mặc định tối ưu cho TPU v3)")
        return 128
    if hw_mode == 'cpu':
        print("  CPU: Dùng batch=32")
        return 32
    
    # GPU: Dò tìm bằng cách thử forward+backward
    from torchvision.models import resnet50
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU VRAM: {vram_gb:.1f} GB — Đang dò tìm batch_size tối ưu...")
    
    # Bắt đầu từ batch lớn, giảm dần nếu OOM
    test_batches = [1024, 768, 512, 384, 256, 128, 64, 32]
    best_batch = 32  # Fallback
    
    test_model = resnet50()
    test_model.fc = torch.nn.Linear(test_model.fc.in_features, 512)
    test_model = test_model.to(dev)
    test_head = torch.nn.Linear(512, 1000).to(dev)
    
    for bs in test_batches:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, 3, 112, 112, device=dev)
            y = torch.randint(0, 1000, (bs,), device=dev)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                feat = test_model(x)
                out = test_head(feat)
                loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            
            best_batch = bs
            print(f"  ✅ batch_size={bs} — OK! (VRAM: {torch.cuda.memory_reserved(0)/(1024**3):.1f}G / {vram_gb:.1f}G)")
            del x, y, feat, out, loss
            torch.cuda.empty_cache()
            break  # Dùng batch lớn nhất tìm được
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  ❌ batch_size={bs} — OOM, thử nhỏ hơn...")
                torch.cuda.empty_cache()
            else:
                raise
    
    del test_model, test_head
    torch.cuda.empty_cache()
    gc.collect()
    
    # Giảm 10% để dành VRAM cho training thực tế
    safe_batch = max(32, int(best_batch * 0.9))
    # Làm tròn về bội số 32
    safe_batch = (safe_batch // 32) * 32
    print(f"  🎯 BATCH_SIZE tối ưu: {safe_batch} (giảm 10% dự phòng từ {best_batch})")
    return safe_batch

AUTO_BATCH_SIZE = auto_find_batch_size(HW_MODE, device)
if HW_MODE == 'gpu' and NUM_GPUS > 1:
    AUTO_BATCH_SIZE = AUTO_BATCH_SIZE * NUM_GPUS
    print(f"  📊 x{NUM_GPUS} GPUs → Tổng BATCH_SIZE: {AUTO_BATCH_SIZE}")
print(f"\n🔧 HW_MODE: {HW_MODE} | BATCH_SIZE: {AUTO_BATCH_SIZE} | GPUs: {NUM_GPUS if HW_MODE=='gpu' else 'N/A'} | AMP: {USE_AMP}")

# ============================================================
# HELPER FUNCTIONS: Thống nhất API giữa TPU và GPU
# ============================================================
def hw_save(obj, path):
    """Lưu model/checkpoint — tự chọn xm.save hoặc torch.save"""
    if HW_MODE == 'tpu':
        xm.save(obj, path)
    else:
        torch.save(obj, path)

def hw_optimizer_step(optimizer):
    """Bước optimizer — tự chọn xm.optimizer_step hoặc optimizer.step"""
    if HW_MODE == 'tpu':
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()

def hw_wrap_loader(loader):
    """Wrap DataLoader cho TPU (MpDeviceLoader) hoặc trả về nguyên cho GPU/CPU"""
    if HW_MODE == 'tpu':
        return pl_module.MpDeviceLoader(loader, device)
    return loader

# ============================================================
# AUTO-BACKUP: Upload/Download checkpoint lên Cloudflare R2
# Chống mất dữ liệu khi Kaggle restart session
# ============================================================
R2_ACCESS_KEY_ID = "a7684a3235bf1f8e3870d82c6dc5ef69"
R2_SECRET_ACCESS_KEY = "a8bf552923ce489626300dc18fe320b3aebba50d52f1439599ce43f955395833"
R2_ENDPOINT = "https://7970c4a57482708b85fec0d3b79dba4d.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "edu-learning-storage"
R2_CHECKPOINT_FOLDER = "models/v4/checkpoints"

def _get_r2_client():
    """Tạo Cloudflare R2 client"""
    try:
        import boto3
        return boto3.client('s3', endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto')
    except ImportError:
        # Cài boto3 nếu chưa có
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'boto3'], capture_output=True)
        import boto3
        return boto3.client('s3', endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto')

def backup_to_r2(local_path, filename=None):
    """Upload checkpoint lên Cloudflare R2"""
    if not os.path.exists(local_path):
        return
    fname = filename or os.path.basename(local_path)
    try:
        s3 = _get_r2_client()
        r2_key = f"{R2_CHECKPOINT_FOLDER}/{fname}"
        s3.upload_file(local_path, R2_BUCKET_NAME, r2_key)
        size_mb = os.path.getsize(local_path) / (1024*1024)
        print(f"  ☁️ Backup {fname} ({size_mb:.1f}MB) → R2 OK!")
    except Exception as e:
        print(f"  ⚠️ Backup R2 lỗi: {e}")

def restore_from_r2(local_path, filename=None):
    """Download checkpoint từ Cloudflare R2 về local"""
    fname = filename or os.path.basename(local_path)
    try:
        s3 = _get_r2_client()
        r2_key = f"{R2_CHECKPOINT_FOLDER}/{fname}"
        s3.download_file(R2_BUCKET_NAME, r2_key, local_path)
        size_mb = os.path.getsize(local_path) / (1024*1024)
        print(f"  ☁️ Tải {fname} ({size_mb:.1f}MB) từ R2 OK!")
        return True
    except Exception:
        return False

# ============================================================
# RE-DEFINE (Cell chay doc lap, tu du)
# ============================================================
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

def get_dataloader(data_dir, batch_size=64, num_workers=2):
    pqs = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)
    pqs = [f for f in pqs if not f.endswith('.metadata')]
    if not pqs: raise FileNotFoundError(f"Khong co parquet o {data_dir}")
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
                      shuffle=True, drop_last=True, num_workers=num_workers), nc, lmap

class ArcFaceMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.20, target_m=0.50):
        super().__init__()
        self.s, self.m, self.target_m = s, m, target_m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._update()
    def _update(self):
        self.cos_m = math.cos(self.m); self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m); self.mm = math.sin(math.pi - self.m) * self.m
    def set_margin(self, m):
        self.m = min(m, self.target_m); self._update()
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1+1e-7, 1-1e-7)
        sine = (1.0 - cosine.pow(2)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh = torch.zeros_like(cosine); oh.scatter_(1, label.view(-1,1).long(), 1)
        return ((oh * phi) + ((1-oh) * cosine)) * self.s

def build_model(nc):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 512)
    return m

def check_spread(features, threshold=0.05):
    with torch.no_grad():
        feat_cpu = features.float().cpu() if HW_MODE == 'tpu' else features.float()
        n = F.normalize(feat_cpu, dim=1)
        if n.size(0) > 256: n = n[torch.randperm(n.size(0))[:256]]
        sim = n @ n.T
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        ps = sim[mask]
        return ps.std().item(), ps.mean().item(), ps.std().item() < threshold

# ============================================================
# MAIN TRAINING — TỰ ĐỘNG CHẠY ĐÚNG TRÊN MỌI PHẦN CỨNG
# ============================================================
def main():
    print("=" * 60)
    print(f"  ArcFace v4 — ANTI-COLLAPSE TRAINING ({HW_MODE.upper()})")
    print("=" * 60)

    # === Tự động tìm data ở cả 2 vị trí ===
    POSSIBLE_DATA_DIRS = [
        '/kaggle/input/casia-webface',                          # Kaggle Dataset (Input)
        '/kaggle/input/casia-webface/CASIA-WebFace',            # Kaggle Dataset (lồng thư mục)
        './Workspace/FaceData/CASIA-WebFace',                   # Google Drive download
        '/kaggle/working/Workspace/FaceData/CASIA-WebFace',     # Google Drive (absolute)
    ]
    DATA_DIR = None
    for d in POSSIBLE_DATA_DIRS:
        if os.path.exists(d) and len(os.listdir(d)) > 0:
            DATA_DIR = d
            break
    if DATA_DIR is None:
        raise FileNotFoundError(
            f"Khong tim thay data o bat ky vi tri nao!\n"
            f"Da kiem tra: {POSSIBLE_DATA_DIRS}\n"
            f"Hay chay Cell 3 truoc de tai data.")
    print(f"  📁 Data: {os.path.abspath(DATA_DIR)}")
    
    SAVE_DIR = '/kaggle/working/FaceModels'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === Tự động khôi phục checkpoint từ Google Drive nếu local trống ===
    CHECKPOINT_PATH_TMP = os.path.join(SAVE_DIR, 'arcface_checkpoint_v4.pth')
    BEST_MODEL_PATH_TMP = os.path.join(SAVE_DIR, 'arcface_best_model_v4.pth')
    if not os.path.exists(CHECKPOINT_PATH_TMP):
        print("\n  🔄 Local trống, thử khôi phục checkpoint từ Cloudflare R2...")
        restore_from_r2(CHECKPOINT_PATH_TMP)
        restore_from_r2(BEST_MODEL_PATH_TMP)

    # ====== v4 HYPER-PARAMETERS ======
    EPOCHS          = 30
    EMBEDDING_SIZE  = 512
    NUM_WORKERS     = 2
    BATCH_SIZE      = AUTO_BATCH_SIZE
    FREEZE_EPOCHS   = 3
    WARMUP_EPOCHS   = 5
    GRAD_CLIP       = 1.0
    PATIENCE        = 10
    HEAD_LR         = 0.01
    BACKBONE_LR     = 1e-4
    MARGIN_START    = 0.20
    MARGIN_END      = 0.50
    MARGIN_WARMUP   = 15
    ARCFACE_SCALE   = 30.0

    CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'arcface_checkpoint_v4.pth')
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'arcface_best_model_v4.pth')

    print(f"Device: {device}")
    print(f"LR: head={HEAD_LR} backbone={BACKBONE_LR} (CO DINH)")
    print(f"Margin: {MARGIN_START} -> {MARGIN_END} (curriculum {MARGIN_WARMUP} epochs)")
    print(f"Scale: {ARCFACE_SCALE} | Grad clip: {GRAD_CLIP} | Batch: {BATCH_SIZE}")

    full_loader, num_classes, label_map = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    full_dataset = full_loader.dataset
    total_size = len(full_dataset)
    val_size = max(1000, int(total_size * 0.01))
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    backbone = build_model(num_classes).to(device)
    margin_layer = ArcFaceMarginProduct(
        in_features=EMBEDDING_SIZE, out_features=num_classes,
        s=ARCFACE_SCALE, m=MARGIN_START, target_m=MARGIN_END
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    # AMP GradScaler chỉ dùng cho GPU
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP) if USE_AMP else None

    optimizer = optim.SGD([
        {'params': backbone.parameters(), 'lr': BACKBONE_LR, 'weight_decay': 5e-4},
        {'params': margin_layer.parameters(), 'lr': HEAD_LR, 'weight_decay': 5e-4}
    ], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

    # Helper để lấy model gốc (bỏ DataParallel wrapper nếu có)
    def get_base_model(model):
        return model.module if hasattr(model, 'module') else model

    def freeze_backbone(model):
        base = get_base_model(model)
        for p in base.parameters(): p.requires_grad = False
        for p in base.fc.parameters(): p.requires_grad = True
        print("  Backbone FROZEN")

    def unfreeze_backbone(model):
        base = get_base_model(model)
        for p in base.parameters(): p.requires_grad = True
        print("  Backbone UNFROZEN")

    start_epoch, best_val_acc, best_epoch, trigger_times = 1, 0.0, 0, 0

    # Load checkpoint TRƯỚC KHI wrap DataParallel
    # (checkpoint luôn lưu state_dict không có prefix "module.")
    if os.path.exists(CHECKPOINT_PATH):
        print("Tim thay checkpoint v4, khoi phuc...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        backbone.load_state_dict(ckpt['backbone'])
        margin_layer.load_state_dict(ckpt['margin'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        trigger_times = ckpt.get('trigger_times', 0)
        print(f"  Resume tu Epoch {start_epoch} | Best Val Acc: {best_val_acc*100:.2f}%")
        backbone = backbone.to(device)
        margin_layer = margin_layer.to(device)
        if start_epoch > FREEZE_EPOCHS + 1:
            for _ in range(start_epoch - FREEZE_EPOCHS - 1):
                scheduler.step()

    # Multi-GPU: Wrap backbone với DataParallel SAU KHI load checkpoint
    if HW_MODE == 'gpu' and NUM_GPUS > 1:
        backbone = nn.DataParallel(backbone)
        print(f"  🔥 DataParallel: backbone chạy trên {NUM_GPUS} GPUs!")

    print(f"\n{'='*60}")
    print(f"  Train: {train_size} anh | {num_classes} classes | batch={BATCH_SIZE}")
    print(f"  Val: {val_size} | Batches/epoch: {len(train_loader)}")
    print(f"{'='*60}")

    collapse_count = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        # Curriculum Margin
        if epoch <= MARGIN_WARMUP:
            curr_m = MARGIN_START + (MARGIN_END - MARGIN_START) * (epoch / MARGIN_WARMUP)
        else:
            curr_m = MARGIN_END
        margin_layer.set_margin(curr_m)

        # Warmup LR
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = epoch / WARMUP_EPOCHS
            base_lrs = [BACKBONE_LR, HEAD_LR]
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = base_lrs[i] * warmup_factor

        if epoch <= FREEZE_EPOCHS:
            if epoch == start_epoch: freeze_backbone(backbone)
        elif epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone(backbone)

        backbone.train(); margin_layer.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        epoch_spread_std, epoch_spread_mean = 0.0, 0.0
        spread_samples = 0

        phase_str = "FROZEN" if epoch <= FREEZE_EPOCHS else "FULL"
        total_batches = len(train_loader)

        # Wrap loader cho TPU nếu cần
        active_loader = hw_wrap_loader(train_loader)

        # TQDM progress bar
        from tqdm import tqdm
        pbar = tqdm(enumerate(active_loader), total=total_batches,
                    desc=f"E{epoch}/{EPOCHS} [{phase_str}] m={curr_m:.2f}",
                    ncols=120, leave=True)

        for bi, (images, labels) in pbar:
            if HW_MODE == 'gpu':
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass — GPU dùng AMP, TPU/CPU chạy thường
            if USE_AMP:
                with torch.amp.autocast('cuda', enabled=True):
                    features = backbone(images)
                    output = margin_layer(features, labels)
                    loss = criterion(output, labels)
                    with torch.no_grad():
                        raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s
            else:
                features = backbone(images)
                output = margin_layer(features, labels)
                loss = criterion(output, labels)
                with torch.no_grad():
                    raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s

            train_correct += (raw_logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

            # Backward + Optimizer step
            if USE_AMP and scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(margin_layer.parameters()),
                    max_norm=GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(margin_layer.parameters()),
                    max_norm=GRAD_CLIP)
                hw_optimizer_step(optimizer)

            train_loss += loss.item()

            # Spread monitor
            if bi % 50 == 0:
                s_std, s_mean, _ = check_spread(features)
                epoch_spread_std += s_std
                epoch_spread_mean += s_mean
                spread_samples += 1

            # Cập nhật tqdm postfix (stream trên 1 dòng)
            if bi % 5 == 0 or bi == total_batches - 1:
                avg_l = train_loss / (bi + 1)
                acc = train_correct / train_total * 100 if train_total > 0 else 0
                postfix = {'Loss': f'{avg_l:.3f}', 'Acc': f'{acc:.1f}%'}
                if HW_MODE == 'gpu':
                    vram = torch.cuda.memory_reserved(0) / (1024**3)
                    postfix['VRAM'] = f'{vram:.1f}G'
                pbar.set_postfix(postfix)

        if epoch > FREEZE_EPOCHS:
            scheduler.step()

        # VALIDATION
        backbone.eval(); margin_layer.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_cosine_sims = []
        active_val_loader = hw_wrap_loader(val_loader)
        with torch.no_grad():
            for images, labels in active_val_loader:
                if HW_MODE == 'gpu':
                    images, labels = images.to(device), labels.to(device)
                if USE_AMP:
                    with torch.amp.autocast('cuda', enabled=True):
                        features = backbone(images)
                        output = margin_layer(features, labels)
                        loss = criterion(output, labels)
                        raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s
                else:
                    features = backbone(images)
                    output = margin_layer(features, labels)
                    loss = criterion(output, labels)
                    raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s
                val_correct += (raw_logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                s_std, s_mean, _ = check_spread(features)
                val_cosine_sims.append((s_std, s_mean))

        avg_tl = train_loss / len(train_loader)
        avg_ta = train_correct / train_total
        avg_vl = val_loss / len(val_loader)
        avg_va = val_correct / val_total if val_total > 0 else 0
        elapsed = time.time() - t0

        avg_spread_std = epoch_spread_std / max(spread_samples, 1)
        avg_spread_mean = epoch_spread_mean / max(spread_samples, 1)
        is_collapsed = avg_spread_std < 0.05
        collapse_tag = " ⚠️ COLLAPSE!" if is_collapsed else " ✅"

        print(f"\n  ┌────────────── KET QUA EPOCH {epoch} ──────────────┐")
        print(f"  │ Train Loss: {avg_tl:.4f}  | Train Acc: {avg_ta*100:6.2f}%  │")
        print(f"  │ Val   Loss: {avg_vl:.4f}  | Val   Acc: {avg_va*100:6.2f}%  │")
        print(f"  │ Margin: {curr_m:.3f} | Scale: {ARCFACE_SCALE}              │")
        print(f"  │ Spread: std={avg_spread_std:.4f} mean={avg_spread_mean:.4f}{collapse_tag} │")
        print(f"  │ Time: {int(elapsed//60)}m{int(elapsed%60):02d}s | Mode: {HW_MODE.upper()}              │")
        print(f"  │ LR: bb={optimizer.param_groups[0]['lr']:.6f} head={optimizer.param_groups[1]['lr']:.6f} │")
        print(f"  └{'─'*49}┘")

        if is_collapsed:
            collapse_count += 1
            print(f"  ⚠️ COLLAPSE DETECTED! ({collapse_count}/3)")
            if collapse_count >= 3:
                print(f"\n  🛑 ABORT: Collapse 3 epoch lien tiep.")
                break
        else:
            collapse_count = 0

        # Early Stopping dựa trên Val ACCURACY
        if avg_va > best_val_acc:
            best_val_acc = avg_va; best_epoch = epoch; trigger_times = 0
            bb_state = backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict()
            hw_save(bb_state, BEST_MODEL_PATH)
            print(f"  ★ [NEW BEST] Val Acc: {avg_va*100:.2f}% -> {BEST_MODEL_PATH}")
        else:
            trigger_times += 1
            print(f"  (No improvement x{trigger_times}/{PATIENCE})")

        # Save checkpoint (tương thích chéo TPU ↔ GPU)
        ckpt_data = {
            'epoch': epoch, 'backbone': backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict(),
            'margin': margin_layer.state_dict(), 'optimizer': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch, 'trigger_times': trigger_times,
            'num_classes': num_classes, 'batch_size': BATCH_SIZE,
        }
        if scaler:
            ckpt_data['scaler'] = scaler.state_dict()
        hw_save(ckpt_data, CHECKPOINT_PATH)

        # AUTO-BACKUP lên Cloudflare R2 sau mỗi epoch
        backup_to_r2(CHECKPOINT_PATH)
        if avg_va > best_val_acc - 0.001:
            backup_to_r2(BEST_MODEL_PATH)

        if trigger_times >= PATIENCE:
            print(f"\n  Early Stopping tai Epoch {epoch}.")
            break

    print(f"\n{'='*55}")
    print(f"  HOAN TAT! Best Epoch: {best_epoch} | Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"  Model: {os.path.abspath(BEST_MODEL_PATH)}")
    print(f"{'='*55}")

main()
```

---

```
%md
### Cell 8: Export ArcFace v4 + MiniFASNet → ONNX → Upload Cloudflare R2
Xuất cả 2 model (ArcFace recognition + MiniFASNet anti-spoofing) thành ONNX,
nén ZIP, upload lên Cloudflare R2. Tải về máy local rồi copy vào `models/`.
```

```python
import subprocess, sys, os, importlib

EXTRA_LIBS = os.path.abspath("./extra_libs")
os.makedirs(EXTRA_LIBS, exist_ok=True)
if EXTRA_LIBS not in sys.path:
    sys.path.insert(0, EXTRA_LIBS)

try: import onnx
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", EXTRA_LIBS, "onnx", "onnxruntime"], capture_output=True, text=True)
    importlib.invalidate_caches(); import onnx

try: import boto3
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", EXTRA_LIBS, "boto3"], capture_output=True, text=True)
    importlib.invalidate_caches(); import boto3

import torch, torch.nn as nn, zipfile
from torchvision.models import resnet50

drive_base = './Workspace'

# ================= 1. CONVERT ARCFACE =================
PTH_PATH = os.path.join(drive_base, 'FaceModels', 'arcface_best_model_v4.pth')
ONNX_PATH = os.path.join(drive_base, 'FaceModels', 'arcface_best_model_v4.onnx')

if not os.path.exists(PTH_PATH):
    print(f"LOI: Khong tim thay {PTH_PATH}. Chay Cell 7 truoc!")
else:
    print("1. Convert ArcFace v4 .pth -> .onnx...")
    model = resnet50(); model.fc = nn.Linear(model.fc.in_features, 512)
    model.load_state_dict(torch.load(PTH_PATH, map_location='cpu')); model.eval()
    torch.onnx.export(model, torch.randn(1,3,112,112), ONNX_PATH,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}})
    print(f"   -> ArcFace ONNX: {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1024/1024:.1f} MB)")

# ================= 2. LOCATE MINIFASNET =================
MINIFAS_DIR = os.path.join(drive_base, 'minifasnetv2', 'AntiSpoofModels')
minifas_onnx = os.path.join(MINIFAS_DIR, 'anti_spoofing_v2.onnx')
minifas_q = os.path.join(MINIFAS_DIR, 'anti_spoofing_v2_q.onnx')

has_minifas = os.path.exists(minifas_q) or os.path.exists(minifas_onnx)
if has_minifas:
    print(f"\n2. MiniFASNet found!")
    if os.path.exists(minifas_q): print(f"   Quantized: {minifas_q}")
    if os.path.exists(minifas_onnx): print(f"   FP32: {minifas_onnx}")
else:
    print(f"\n2. MiniFASNet ONNX khong tim thay tai {MINIFAS_DIR}")
    print(f"   -> Chay MiniFASNet_Zeppelin_FineTune.md truoc (Step 7 + Step 8)")

# ================= 3. ZIP ALL =================
ZIP_PATH = os.path.join(drive_base, 'FaceModels', 'Models_v4_Export.zip')
print(f"\n3. Nen tat ca models...")
if os.path.exists(ZIP_PATH): os.remove(ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # ArcFace
    if os.path.exists(ONNX_PATH):
        zipf.write(ONNX_PATH, 'arcface_best_model_v4.onnx')
    if os.path.exists(PTH_PATH):
        zipf.write(PTH_PATH, 'arcface_best_model_v4.pth')
    # MiniFASNet
    if os.path.exists(minifas_q):
        zipf.write(minifas_q, 'anti_spoofing_v2_q.onnx')
    if os.path.exists(minifas_onnx):
        zipf.write(minifas_onnx, 'anti_spoofing_v2.onnx')

print(f"   -> ZIP: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB)")

# ================= 4. UPLOAD CLOUDFLARE R2 =================
print("\n4. Upload len Cloudflare R2...")

R2_ACCESS_KEY_ID = "a7684a3235bf1f8e3870d82c6dc5ef69"
R2_SECRET_ACCESS_KEY = "a8bf552923ce489626300dc18fe320b3aebba50d52f1439599ce43f955395833"
R2_ENDPOINT = "https://7970c4a57482708b85fec0d3b79dba4d.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "edu-learning-storage"
R2_FOLDER = "models/v4"

s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto')

upload_files = [f for f in [ONNX_PATH, PTH_PATH, minifas_onnx, minifas_q, ZIP_PATH] if os.path.exists(f)]

for fpath in upload_files:
    fname = os.path.basename(fpath)
    fsize = os.path.getsize(fpath) / (1024*1024)
    r2_key = f"{R2_FOLDER}/{fname}"
    print(f"   Uploading {fname} ({fsize:.1f} MB) -> {r2_key}...")
    s3.upload_file(fpath, R2_BUCKET_NAME, r2_key)
    print(f"   [OK] {fname}")

print("\n" + "="*60)
print("  HOAN TAT! Tat ca models da upload len R2:")
print(f"  Bucket: {R2_BUCKET_NAME} / {R2_FOLDER}/")
for f in upload_files: print(f"    - {os.path.basename(f)}")
print("="*60)
print("\n  SAU KHI TAI VE MAY LOCAL:")
print("  1. Copy arcface_best_model_v4.onnx -> models/")
print("  2. Copy anti_spoofing_v2_q.onnx -> models/")
print("  3. Cap nhat config.py: ARCFACE_PATH = 'models/arcface_best_model_v4.onnx'")
```
