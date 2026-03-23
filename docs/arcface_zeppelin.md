# Kịch Bản Huấn Luyện ArcFace Trên IO Lab (Apache Zeppelin - Windows)

Toàn bộ quy trình đã được tối ưu. Data được nén thành file zip trên Google Drive trước, sau đó IO Lab chỉ cần tải 1 file duy nhất rồi giải nén. Không tải từng ảnh lẻ.

---

### Chuẩn bị trước khi chạy (Chỉ làm 1 lần)

**Bước A: Nén dữ liệu trên Google Colab**

Mở Google Colab, chạy đoạn code sau để nén FaceData thành 1 file zip trên Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil, os

# Sửa đường dẫn cho đúng thư mục FaceData trên Drive của bạn
src = '/content/drive/MyDrive/FaceData'
print("Nội dung thư mục:", os.listdir(src))

print("Đang nén FaceData...")
shutil.make_archive('/content/drive/MyDrive/FaceData_zipped', 'zip', src)
print("Xong! File FaceData_zipped.zip đã xuất hiện trên Drive.")
```

**Bước B: Share file zip**

1. Lên Google Drive, bấm chuột phải vào `FaceData_zipped.zip` → Share → Anyone with the link.
2. Copy link, lấy phần **FILE_ID** trong `https://drive.google.com/file/d/FILE_ID/view`.
3. Dán FILE_ID vào biến `FACE_DATA_ZIP_ID` ở Cell 3 bên dưới.

**Bước C: Share thư mục FaceModels cho Bot**

Lên Google Drive, mở thư mục `FaceModels`, bấm Share, thêm email sau với quyền **Editor**:
`io-lab-bot@ai-english-app-488114.iam.gserviceaccount.com`

---

```
%md
### Cell 1: Cai dat thu vien (Thong minh - Tu bo qua neu da cai)
Lan dau: Cai PyTorch CUDA + thu vien phu tro (~10 phut). Sau khi cai xong can Restart Interpreter roi chay lai Cell nay.
Lan sau: Kiem tra thay da co -> BO QUA ngay (~2 giay).
```

```python
import subprocess, sys, os, importlib

# GPU_ID = 2: RTX 3090. Doi gia tri neu can GPU khac.
# Dung torch.device('cuda:2') truc tiep vi CUDA_VISIBLE_DEVICES khong hoat dong tren Zeppelin Windows.

# Thu muc cai them (co quyen ghi, tranh loi Access Denied cua venv)
EXTRA_LIBS = os.path.abspath("./extra_libs")
os.makedirs(EXTRA_LIBS, exist_ok=True)
if EXTRA_LIBS not in sys.path:
    sys.path.insert(0, EXTRA_LIBS)

# --- Kiem tra torch CUDA ---
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
    print("[1/2] Cai PyTorch CUDA 12.4 (~10 phut)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--default-timeout=1000", "--retries=10",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ], timeout=3600)

# --- Cai thu vien phu vao extra_libs (tranh loi Access Denied) ---
need_install = []
for pkg in ["gdown", "tqdm", "cv2", "huggingface_hub", "datasets", "psutil"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        need_install.append(pkg.replace("cv2", "opencv-python"))

if need_install:
    print(f"\nCai {len(need_install)} package vao {EXTRA_LIBS}...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", EXTRA_LIBS,
         *need_install, "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"],
        capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Cai thanh cong!")
    else:
        print(f"[LOI] {r.stderr[-300:]}")
    importlib.invalidate_caches()
else:
    print("[OK] Thu vien phu tro da co san.")

# --- Import tat ca ---
import torch, gdown, tqdm, psutil, cv2, huggingface_hub, datasets

print(f"\n[OK] Tat ca thu vien da san sang!")
print(f"  PyTorch : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}  : {torch.cuda.get_device_name(i)}")
print(f"  gdown   : {gdown.__version__} | OpenCV: {cv2.__version__}")
print("[OK] San sang. Chuyen sang Cell tiep theo.")
```
---

```
%md
### Cell 2: Kiem tra tai nguyen va tao thu muc
Kiem tra RAM, Disk, GPU truoc khi bat dau. Tao cac thu muc can thiet.
```

```python
import os
import psutil
import shutil
import subprocess

print("==== THÔNG TIN TÀI NGUYÊN ====")
ram = psutil.virtual_memory()
print(f"RAM: Trống {ram.available / (1024**3):.2f} GB / Tổng {ram.total / (1024**3):.2f} GB")
total, used, free = shutil.disk_usage("/")
print(f"Disk: Trống {free / (1024**3):.2f} GB / Tổng {total / (1024**3):.2f} GB")

try:
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.free,memory.used", "--format=csv"]).decode('utf-8')
    print("\nGPU:\n", gpu_info)
except Exception:
    print("\nKhông tìm thấy nvidia-smi. Hệ thống đang chạy CPU.")

print("\n==== TẠO THƯ MỤC ====")
drive_base = './Workspace' 
for d in ['FaceModels', 'FaceData/CASIA-WebFace']:
    os.makedirs(os.path.join(drive_base, d), exist_ok=True)

print("Thư mục Dataset:", os.path.abspath(os.path.join(drive_base, 'FaceData/CASIA-WebFace')))
print("Thư mục Model:", os.path.abspath(os.path.join(drive_base, 'FaceModels')))
```

---

```
%md
### Cell 2.5: Tao file Token xac thuc Google Drive
Tao file service_account.json de Zeppelin co quyen tai/luu model len Google Drive.
```

```python
import json
import os

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
### Cell 3: Tai du lieu tu Google Drive (1 file zip duy nhat)
Tai 1 cuc zip nhanh gap hang tram lan so voi tai tung anh le.
Ban can thay FILE_ID bang ID that sau khi nen tren Colab.
```

```python
import sys, os
sys.path.insert(0, os.path.abspath("./extra_libs"))
import gdown
import zipfile

drive_base = './Workspace'
DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
MODEL_DIR = os.path.join(drive_base, 'FaceModels')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================
# THAY FILE_ID BANG ID THAT CUA FILE FaceData_zipped.zip TREN DRIVE
FACE_DATA_ZIP_ID = "1DjOS4P5rYa1TWttIsf938jUu6U1sDvQ6"
# =============================================================

zip_path = os.path.join(drive_base, 'FaceData.zip')

# Kiem tra xem data da co san hoan chinh chua
data_files = []
has_parquet = False
for root, dirs, files in os.walk(DATA_DIR):
    data_files.extend(files)
    if any(f.endswith('.parquet') for f in files):
        has_parquet = True

is_ready = os.path.exists(os.path.join(DATA_DIR, 'DATA_READY.txt'))

if is_ready or has_parquet or len(data_files) > 100:
    print(f"Du lieu da co san hoan chinh ({len(data_files)} files). BO QUA TAI VA GIAI NEN.")
elif FACE_DATA_ZIP_ID == "THAY_ID_VAO_DAY":
    print("LOI: Ban chua thay FACE_DATA_ZIP_ID bang ID that.")
    print("Hay doc lai phan 'Chuan bi truoc khi chay' o dau tai lieu.")
else:
    # Tai file zip
    if not os.path.exists(zip_path):
        print("Dang tai FaceData.zip tu Google Drive...")
        gdown.download(id=FACE_DATA_ZIP_ID, output=zip_path, quiet=False)
        print("Tai zip xong.")
    
    # Giai nen
    print("Dang giai nen...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    print(f"Giai nen hoan tat. Tong files: {len(os.listdir(DATA_DIR))}")
    
    # Xoa file zip de tiet kiem dung luong
    os.remove(zip_path)
    
    # Tao file danh dau hoan tat 100%
    with open(os.path.join(DATA_DIR, 'DATA_READY.txt'), 'w') as f:
        f.write("OK")
    print("Da xoa file zip tam.")

# Tai FaceModels bang API (thu muc nho, it file)
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
            if os.path.exists(out_path):
                continue
            print(f"  Tai: {item['name']}")
            request = svc.files().get_media(fileId=item['id'])
            fh = _io.FileIO(out_path, 'wb')
            dl = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = dl.next_chunk()
        print("Tai model xong.")

print("\nHoan tat Cell 3.")
```

---

```
%md
### Cell 4: Kiem tra cau truc du lieu
Kiem tra xem du lieu da duoc tai va giai nen dung chua (parquet, jpg, png...).
```

```python
import os
import glob

drive_base = './Workspace' 
DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')

print(f"Kiem tra {os.path.abspath(DATA_DIR)}...")

parquets = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
jpgs = glob.glob(f"{DATA_DIR}/**/*.jpg", recursive=True)
pngs = glob.glob(f"{DATA_DIR}/**/*.png", recursive=True)
folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

print(f"  Parquet files: {len(parquets)}")
print(f"  JPG files: {len(jpgs)}")
print(f"  PNG files: {len(pngs)}")
print(f"  Sub-folders: {len(folders)}")

if parquets:
    print(f"\nDataset dang Parquet ({len(parquets)} files). Train se doc truc tiep.")
elif jpgs or pngs:
    print(f"\nDataset dang anh ({len(jpgs)+len(pngs)} files trong {len(folders)} folders).")
else:
    print("\nKhong tim thay du lieu. Kiem tra lai Cell 3.")
```

---

```
%md
### Cell 4.5: Stress Test GPU - Tim BATCH TOI DA thuc te
Chay thu forward + backward thuc te voi AMP de tim batch_size toi uu cho RTX 3090.
Ket qua se duoc hien thi de ban dien vao Cell 7.
```

```python
import os, sys
import subprocess
import time
import gc
import psutil

# Fix loi torch._dynamo tren Python 3.13
os.environ['TORCHDYNAMO_DISABLE'] = '1'
for key in list(sys.modules.keys()):
    if 'torch._dynamo' in key:
        del sys.modules[key]

import torch
import torch.nn as nn

print("=" * 70)
print("  STRESS TEST GPU - TIM BATCH TOI DA THUC TE (AMP)")
print("=" * 70)

# ================================================================
# 1. Thong tin he thong
# ================================================================
cpu_count_logical = os.cpu_count() or 1
ram = psutil.virtual_memory()
print(f"\n--- CPU: {cpu_count_logical} loi | RAM: {ram.available/(1024**3):.1f}/{ram.total/(1024**3):.1f} GB ---")

gpu_list = []
try:
    gpu_csv = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits"
    ]).decode('utf-8').strip()
    for line in gpu_csv.split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            gpu_list.append({
                'idx': int(parts[0]), 'name': parts[1],
                'total_mb': int(parts[2]), 'free_mb': int(parts[3])
            })
            print(f"  GPU {parts[0]}: {parts[1]} ({parts[2]} MB, trong {parts[3]} MB)")
except Exception:
    print("  Khong co GPU!")

# Disk I/O
test_file = './Workspace/_bench.bin'
try:
    chunk = b'x' * (50 * 1024 * 1024)
    with open(test_file, 'wb') as f: f.write(chunk)
    t0 = time.time()
    with open(test_file, 'rb') as f: _ = f.read()
    avg_read = 50 / max(time.time() - t0, 0.001)
    os.remove(test_file)
    print(f"  Disk doc: {avg_read:.0f} MB/s")
except:
    avg_read = 80

if not gpu_list:
    print("[LOI] Khong tim thay GPU.")
else:
    best_gpu = max(gpu_list, key=lambda g: g['free_mb'])
    gpu_idx = best_gpu['idx']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    device = torch.device('cuda')
    print(f"\n  => Test tren GPU {gpu_idx}: {best_gpu['name']} ({best_gpu['total_mb']} MB)")
    print(f"  PyTorch CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")

    # ================================================================
    # 2. STRESS TEST THUC TE
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  CHAY THU FORWARD + BACKWARD THUC TE VOI AMP")
    print(f"{'=' * 70}")
    
    IMG_SIZE = 112
    EMBEDDING_DIM = 512
    NUM_CLASSES = 10575
    
    # Tao ResNet50 (tranh circular import cua torch._dynamo)
    try:
        import torchvision
        backbone = torchvision.models.resnet50(weights=None)
    except (AttributeError, ImportError):
        # Fallback: dung model tuong duong
        from torchvision.models import resnet50 as _r50
        backbone = _r50(pretrained=False)
    backbone.fc = nn.Linear(backbone.fc.in_features, EMBEDDING_DIM)
    backbone = backbone.to(device)
    arcface = nn.Linear(EMBEDDING_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(arcface.parameters()),
        lr=0.01, momentum=0.9
    )
    
    batch_sizes = [128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096]
    results = []
    max_ok_batch = 0
    
    print(f"\n  {'Batch':>6} | {'VRAM':>8} | {'%':>5} | {'Toc do':>10} | Ket qua")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*5}-+-{'-'*10}-+--------")
    
    for bs in batch_sizes:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            images = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=device)
            labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            
            t0 = time.time()
            with torch.amp.autocast('cuda'):
                features = backbone(images)
                logits = arcface(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            elapsed = time.time() - t0
            speed = bs / elapsed
            vram_used = torch.cuda.max_memory_allocated() / (1024**2)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            pct = vram_used / vram_total * 100
            
            results.append({'batch': bs, 'vram_mb': vram_used, 'pct': pct, 'speed': speed})
            max_ok_batch = bs
            
            tag = "OK" if pct < 85 else "SAT NGUONG"
            print(f"  {bs:>6} | {vram_used:>6.0f}M | {pct:>4.1f}% | {speed:>6.0f} img/s | {tag}")
            
            del images, labels, features, logits, loss
            if pct > 95:
                print(f"         => VRAM gan het, dung.")
                break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {bs:>6} |    ---   |  ---  |     ---    | OOM!")
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                print(f"  {bs:>6} | LOI: {str(e)[:40]}")
                break
    
    del backbone, arcface, criterion, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    # ================================================================
    # 3. DE XUAT
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  KET QUA VA DE XUAT")
    print(f"{'=' * 70}")
    
    if results:
        target_batch = int(max_ok_batch * 0.7)
        recommended_batch = max(64, (target_batch // 64) * 64)
        best_r = max(results, key=lambda r: r['batch'])
        
        print(f"\n  Batch MAX thanh cong  : {max_ok_batch}")
        print(f"  VRAM tai MAX          : {best_r['vram_mb']:.0f} MB ({best_r['pct']:.1f}%)")
        print(f"  Toc do tai MAX        : {best_r['speed']:.0f} img/s")
        
        print(f"\n  >>> DE XUAT CAU HINH (70% MAX) <<<")
        print(f"  CUDA_VISIBLE_DEVICES = '{gpu_idx}'")
        print(f"  batch_size   = {recommended_batch}")
        
        recommended_workers = max(0, min(cpu_count_logical - 2, 8))
        if avg_read < 100:
            recommended_workers = max(0, recommended_workers // 2)
        print(f"  num_workers  = {recommended_workers}")
        print(f"  pin_memory   = True")
        print(f"  USE_AMP      = True")
        
        closest = min(results, key=lambda r: abs(r['batch'] - recommended_batch))
        est_speed = closest['speed'] * (recommended_batch / closest['batch']) if closest['batch'] > 0 else 100
        total_images = 490000
        sec_epoch = total_images / max(est_speed, 1)
        print(f"\n  Uoc tinh toc do     : ~{est_speed:.0f} img/s")
        print(f"  Uoc tinh 1 epoch    : {int(sec_epoch//60)} phut {int(sec_epoch%60)} giay")
        print(f"  Uoc tinh 30 epoch   : {int(sec_epoch*30/3600)} gio {int((sec_epoch*30%3600)/60)} phut")
    
    print(f"\n{'=' * 70}")
    print("  DAN CAC GIA TRI TREN VAO CELL 7 TRUOC KHI BAM CHAY!")
    print(f"{'=' * 70}")
```

---

```
%md
### Cell 5: DataLoader & ArcFace Model
Khai bao thu vien, DataLoader chuan bi du lieu va Model ArcFace de train.
Cell nay load truc tiep tu file parquet goc va resize anh online (khong can cache).
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

# ================= CLASS DATASET =================
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

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image'].convert('RGB')
        tensor = self.transform(image)

        raw_label = item[self.lbl_c]
        mapped_label = self.label_map[raw_label] if self.label_map is not None else raw_label
        return tensor, torch.tensor(mapped_label, dtype=torch.long)

# ================= DATALOADER BOOTSTRAP =================
def get_dataloader(data_dir, batch_size=64, num_workers=0):
    parquet_files = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)
    parquet_files = [f for f in parquet_files if not f.endswith('.metadata')]
    
    # Fallback mien la co file parquet
    if not parquet_files:
        raise FileNotFoundError(f"Khong co parquet o {data_dir}")

    print(f"Loading {len(parquet_files)} parquet tu {data_dir}...")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")

    cols = ds.column_names
    lbl_col = next((c for c in cols if c in ['label', 'labels', 'target']), cols[1] if len(cols)>1 else cols[-1])
    
    all_labels = ds[lbl_col]
    
    feature = ds.features[lbl_col]
    if hasattr(feature, 'num_classes'):
        num_classes = feature.num_classes
        # If it's a ClassLabel feature, unique_labels might not be directly available
        # We assume the labels are already integers from 0 to num_classes-1
        label_map = None 
        print(f"Phat hien ClassFeature, co {num_classes} classes.")
    else:
        unique_labels = sorted(list(set(all_labels)))
        num_classes = len(unique_labels)
        label_map = {orig: i for i, orig in enumerate(unique_labels)}
        print(f"Tu tao mapping cho {num_classes} classes.")

    dataset = HFDatasetWrapper(ds, lbl_col, label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                    drop_last=True, num_workers=num_workers, pin_memory=True) # pin_memory de copy vao GPU nhanh
    
    return loader, int(num_classes), label_map

class ArcFaceMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
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

print("Khai bao DataLoader va ArcFace hoan tat.")
```

---

```
%md
### Cell 6: Kiem tra Label va Model truoc khi train
Chay thu 1 batch nho (30 step) de xac nhan model va data tuong thich.
Neu Acc > 50% -> PASS. San sang train chinh thuc.
```

```python
import os
import torch

print("=" * 60)
print("Kiem tra Label va Model truoc khi train")
print("=" * 60)

drive_base = './Workspace' 
DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')

_test_loader, _test_nc, _test_lmap = get_dataloader(DATA_DIR, batch_size=64, num_workers=0)
device = torch.device('cuda:1')  # RTX 3090

_imgs, _lbls = next(iter(_test_loader))
print(f"Shape: {_imgs.shape}, Labels: {_lbls[:10].tolist()}")

_bb = build_model(_test_nc).to(device)
_mg = ArcFaceMarginProduct(in_features=512, out_features=_test_nc).to(device)
_cr = torch.nn.CrossEntropyLoss()
_op = torch.optim.SGD(list(_bb.parameters())+list(_mg.parameters()), lr=0.01)
_imgs, _lbls = _imgs.to(device), _lbls.to(device)

print("\nTrain thu 1 batch...")
for _i in range(30):
    _op.zero_grad()
    _out = _mg(_bb(_imgs), _lbls)
    _loss = _cr(_out, _lbls)
    _loss.backward()
    _op.step()
    if _i % 10 == 0:
        _acc = (_out.argmax(1)==_lbls).float().mean().item()
        print(f"Step {_i:3d} | Loss: {_loss.item():.4f} | Acc: {_acc*100:.1f}%")

_final = (_out.argmax(1)==_lbls).float().mean().item()
if _final > 0.5:
    print(f"\nPASS. Acc={_final*100:.1f}%. San sang train.")
else:
    print(f"\nFAIL. Acc={_final*100:.1f}%. Kiem tra lai data.")
del _bb, _mg, _cr, _op, _test_loader
if torch.cuda.is_available(): torch.cuda.empty_cache()
```

---

```
%md
### Cell 7: Training Loop (AMP Mixed Precision)
Bat dau qua trinh train truc tiep tu file parquet (khong can cache).
Giai thuat su dung CUDA AMP (Mixed Precision) de toi da hoa toc do tren card RTX 3090.
Thoi gian uoc tinh: ~15 phut/epoch x 30 epoch = ~7.5 gio.
```

```python
import sys, os
sys.path.insert(0, os.path.abspath("./extra_libs"))

# Fix loi torch._dynamo tren Python 3.13
os.environ['TORCHDYNAMO_DISABLE'] = '1'
for key in list(sys.modules.keys()):
    if 'torch._dynamo' in key:
        del sys.modules[key]

import math, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import warnings
warnings.filterwarnings('ignore', message='.*_MultiProcessingDataLoaderIter.*')

# Don dep worker cu (neu co)
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
_old_wrapper = os.path.join(os.getcwd(), 'hf_dataset_wrapper.py')
if os.path.exists(_old_wrapper):
    os.remove(_old_wrapper)

# ================= DINH NGHIA TRUC TIEP (tu Cell 5 — de Cell 7 chay doc lap) =================
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

    def __len__(self):
        return len(self.ds)

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
    all_labels = ds[lbl_col]
    feature = ds.features[lbl_col]
    if hasattr(feature, 'num_classes'):
        num_classes = feature.num_classes
        label_map = None
        print(f"Phat hien ClassFeature, co {num_classes} classes.")
    else:
        unique_labels = sorted(list(set(all_labels)))
        num_classes = len(unique_labels)
        label_map = {orig: i for i, orig in enumerate(unique_labels)}
        print(f"Tu tao mapping cho {num_classes} classes.")
    dataset = HFDatasetWrapper(ds, lbl_col, label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=num_workers, pin_memory=True)
    return loader, int(num_classes), label_map

class ArcFaceMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
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

# ================= AUTO BATCH SIZE FINDER (Max VRAM) =================
def auto_find_batch_size(device, embedding_dim=512, num_classes=10575, safety=1.0):
    """Tu dong tim batch_size lon nhat co the chay duoc tren GPU hien tai.
    
    Cach hoat dong:
    - Thu forward + backward voi batch tang dan (binary search)
    - Khi gap OOM -> quay lai batch truoc do
    - Lay 75% cua max de dam bao on dinh khi train that
    
    Returns: batch_size (int, lam tron xuo boi 64)
    """
    print("\n" + "=" * 60)
    print("  AUTO-TUNING: Tim batch_size toi uu...")
    print("=" * 60)
    
    # Tao model tam de test
    _bb = resnet50(weights=None)
    _bb.fc = nn.Linear(_bb.fc.in_features, embedding_dim)
    _bb = _bb.to(device)
    _head = nn.Linear(embedding_dim, num_classes).to(device)
    _crit = nn.CrossEntropyLoss()
    _scaler = torch.amp.GradScaler('cuda')
    _opt = torch.optim.SGD(list(_bb.parameters()) + list(_head.parameters()), lr=0.01)
    
    # Binary search: tim max batch
    candidates = [64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048]
    max_ok = 64  # fallback nho nhat
    
    for bs in candidates:
        _opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            imgs = torch.randn(bs, 3, 112, 112, device=device)
            lbls = torch.randint(0, num_classes, (bs,), device=device)
            
            with torch.amp.autocast('cuda'):
                feat = _bb(imgs)
                logits = _head(feat)
                loss = _crit(logits, lbls)
            _scaler.scale(loss).backward()
            _scaler.step(_opt)
            _scaler.update()
            _opt.zero_grad(set_to_none=True)
            
            vram = torch.cuda.max_memory_allocated(device) / (1024**3)
            vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            pct = vram / vram_total * 100
            
            max_ok = bs
            print(f"  Batch {bs:>5} -> VRAM: {vram:.1f}/{vram_total:.1f} GB ({pct:.0f}%) OK", flush=True)
            
            del imgs, lbls, feat, logits, loss
            torch.cuda.reset_peak_memory_stats(device)
            
            if pct > 92:
                print(f"  => VRAM gan het, dung tai {bs}.")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch {bs:>5} -> OOM! Dung tai {max_ok}.")
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise e
    
    # Don dep model tam
    del _bb, _head, _crit, _opt, _scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    # Lay % an toan, lam tron xuong boi 64
    safe_batch = int(max_ok * safety)
    safe_batch = max(64, (safe_batch // 64) * 64)
    
    print(f"\n  >> KET QUA AUTO-TUNE:")
    print(f"     Max batch OK  : {max_ok}")
    print(f"     Safe batch    : {safe_batch} ({safety*100:.0f}% cua max, boi 64)")
    print("=" * 60)
    
    return safe_batch


def auto_scale_lr(batch_size, base_batch=256):
    """Tu dong scale learning rate theo Linear Scaling Rule.
    
    Nguyen ly: Khi tang batch_size gap K lan, tang LR gap K lan.
    Base: batch=256 -> head_lr=0.1, backbone_lr=1e-3
    
    Returns: (head_lr, backbone_lr)
    """
    scale = batch_size / base_batch
    head_lr = 0.1 * scale
    backbone_lr = 1e-3 * scale
    
    # Clamp de tranh LR qua lon gay NaN
    head_lr = min(head_lr, 1.0)
    backbone_lr = min(backbone_lr, 0.01)
    
    return head_lr, backbone_lr


# ================= MAIN TRAINING =================
def main():
    # Chon GPU truc tiep (CUDA_VISIBLE_DEVICES khong hoat dong tren Zeppelin Windows)
    GPU_ID = 1  # RTX 3090 (24GB) - thay doi neu can

    print("Khoi dong huan luyen ArcFace (AMP Mixed Precision)")

    # =====================================================
    # CAU HINH - TU DONG TOI UU
    # =====================================================
    drive_base = './Workspace'
    DATA_DIR = os.path.join(drive_base, 'FaceData', 'CASIA-WebFace')
    SAVE_DIR = os.path.join(drive_base, 'FaceModels')
    os.makedirs(SAVE_DIR, exist_ok=True)

    NUM_WORKERS = 0           # Zeppelin Windows khong ho tro multiprocessing
    EPOCHS = 30
    EMBEDDING_SIZE = 512
    CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'arcface_checkpoint_v3.pth')
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'arcface_best_model_v3.pth')

    # Tang toc GPU
    torch.backends.cudnn.benchmark = True
    USE_AMP = True

    device = torch.device(f'cuda:{GPU_ID}')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(GPU_ID)}")
        print(f"AMP   : {'BAT' if USE_AMP else 'TAT'}")

    # >>> BATCH SIZE FIXED Tùy Chọn (~ 20GB VRAM) <<<
    # Batch = 1792 (Bội số của 128) giúp tối ưu hoàn hảo Tensor Core, ăn tầm 20.1GB VRAM cực đẹp
    BATCH_SIZE = 1792
    
    HEAD_LR, BACKBONE_LR = auto_scale_lr(BATCH_SIZE)
    print(f"  Auto LR: head={HEAD_LR:.4f} backbone={BACKBONE_LR:.5f} (scaled tu batch={BATCH_SIZE})")

    FREEZE_EPOCHS = 3
    WARMUP_EPOCHS = 2
    GRAD_CLIP = 5.0
    PATIENCE = 10
    # =====================================================

    full_loader, num_classes, label_map = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)

    if full_loader is not None:
        full_dataset = full_loader.dataset
        total_size = len(full_dataset)
        val_size = max(1000, int(total_size * 0.01))
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True
        )

        backbone = build_model(num_classes).to(device)
        margin_layer = ArcFaceMarginProduct(in_features=EMBEDDING_SIZE, out_features=num_classes, s=64.0, m=0.50).to(device)
        criterion = torch.nn.CrossEntropyLoss()

        # AMP: GradScaler chong underflow khi dung float16
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        optimizer = optim.SGD([
            {'params': backbone.parameters(), 'lr': BACKBONE_LR, 'weight_decay': 5e-4},
            {'params': margin_layer.parameters(), 'lr': HEAD_LR, 'weight_decay': 5e-4}
        ], momentum=0.9)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

        def freeze_backbone(model):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            print("Backbone FROZEN", flush=True)

        def unfreeze_backbone(model):
            for param in model.parameters():
                param.requires_grad = True
            print("Backbone UNFROZEN", flush=True)

        start_epoch = 1
        best_val_loss = float('inf')
        best_epoch = 0
        trigger_times = 0

        if os.path.exists(CHECKPOINT_PATH):
            print("Tim thay checkpoint, dang khoi phuc...", flush=True)
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            backbone.load_state_dict(ckpt['backbone'])
            margin_layer.load_state_dict(ckpt['margin'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt and USE_AMP:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_loss', float('inf'))
            best_epoch = ckpt.get('best_epoch', 0)
            trigger_times = ckpt.get('trigger_times', 0)
            print(f"  Resume tu Epoch {start_epoch} | Best Loss: {best_val_loss:.4f}", flush=True)

            # Phuc hoi scheduler theo dung epoch hien tai (Fix loi LR bi nhay vot sau khi resume)
            if start_epoch > FREEZE_EPOCHS + 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for _ in range(start_epoch - FREEZE_EPOCHS - 1):
                        scheduler.step()

        print(f"\nBat dau train: {total_size} anh | {num_classes} classes | batch={BATCH_SIZE}", flush=True)
        print(f"Train set: {train_size} | Val set: {val_size}", flush=True)
        print(f"Batches/epoch: {len(train_loader)}", flush=True)

        for epoch in range(start_epoch, EPOCHS + 1):
            t0 = time.time()

            # Warmup: tang LR tu tu trong vai epoch dau
            if epoch <= WARMUP_EPOCHS and epoch <= FREEZE_EPOCHS:
                warmup_factor = epoch / WARMUP_EPOCHS
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * warmup_factor

            if epoch <= FREEZE_EPOCHS:
                if epoch == start_epoch:
                    freeze_backbone(backbone)
            else:
                if epoch == FREEZE_EPOCHS + 1:
                    unfreeze_backbone(backbone)

            backbone.train()
            margin_layer.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            phase_str = "PHASE 1 - FROZEN" if epoch <= FREEZE_EPOCHS else "PHASE 2 - FULL"
            total_batches = len(train_loader)
            print(f"\n  === Epoch {epoch}/{EPOCHS} [{phase_str}] === {total_batches} batches", flush=True)
            last_print = 0

            for bi, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    features = backbone(images)
                    output = margin_layer(features, labels)
                    loss = criterion(output, labels)
                    
                    # Fix loi Acc: ArcFace giam logit cua class dung qua margin nen output.argmax() hay sai
                    # -> Can lay logits goc de tinh True Accuracy
                    with torch.no_grad():
                        raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s

                train_correct += (raw_logits.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(margin_layer.parameters()),
                    max_norm=GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                now = time.time()

                if now - last_print >= 30.0 or bi == total_batches - 1:
                    last_print = now
                    pct = (bi + 1) / total_batches * 100
                    avg_l = train_loss / (bi + 1)
                    acc = train_correct / train_total * 100 if train_total > 0 else 0
                    elapsed = now - t0
                    speed = elapsed / (bi + 1) if (bi + 1) > 0 else 0
                    eta_s = speed * (total_batches - bi - 1)
                    vram = torch.cuda.memory_reserved(GPU_ID) / (1024**3)
                    done = int(20 * (bi + 1) / total_batches)
                    bar = '#' * done + '.' * (20 - done)
                    em, es = int(elapsed // 60), int(elapsed % 60)
                    rm, rs = int(eta_s // 60), int(eta_s % 60)
                    print(f"  E{epoch}/{EPOCHS}: {pct:3.0f}%|{bar}| {bi+1}/{total_batches} [{em:02d}:{es:02d}<{rm:02d}:{rs:02d}] Loss:{avg_l:.3f} Acc:{acc:.1f}% VRAM:{vram:.1f}G", flush=True)

            if epoch > FREEZE_EPOCHS:
                scheduler.step()

            # --- VALIDATION ---
            print(f"  Dang danh gia Validation ({len(val_loader)} batches)...", flush=True)
            backbone.eval()
            margin_layer.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for vi, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        features = backbone(images)
                        output = margin_layer(features, labels)
                        loss = criterion(output, labels)
                        
                        raw_logits = F.linear(F.normalize(features), F.normalize(margin_layer.weight)) * margin_layer.s
                        
                    val_correct += (raw_logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
                    val_loss += loss.item()
            print(f"  Validation xong!", flush=True)

            avg_tl = train_loss / len(train_loader)
            avg_ta = train_correct / train_total
            avg_vl = val_loss / len(val_loader)
            avg_va = val_correct / val_total if val_total > 0 else 0
            elapsed = time.time() - t0

            vram_used = torch.cuda.memory_allocated(GPU_ID) / (1024**2)
            vram_cached = torch.cuda.memory_reserved(GPU_ID) / (1024**2)

            print(f"\n  ┌─────────────── KET QUA EPOCH {epoch} ───────────────┐", flush=True)
            print(f"  │ Train Loss : {avg_tl:.4f}    | Train Acc : {avg_ta*100:6.2f}%  │", flush=True)
            print(f"  │ Val   Loss : {avg_vl:.4f}    | Val   Acc : {avg_va*100:6.2f}%  │", flush=True)
            print(f"  │ Time: {int(elapsed//60)}m{int(elapsed%60):02d}s | VRAM: {vram_used:.0f}MB / {vram_cached:.0f}MB │", flush=True)
            print(f"  │ LR: bb={optimizer.param_groups[0]['lr']:.6f} head={optimizer.param_groups[1]['lr']:.6f} │", flush=True)
            print(f"  └{'─'*47}┘", flush=True)

            # Luu model tot nhat
            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                best_epoch = epoch
                trigger_times = 0
                torch.save(backbone.state_dict(), BEST_MODEL_PATH)
                print(f"  ★ [NEW BEST] Model saved -> {BEST_MODEL_PATH}", flush=True)
            else:
                trigger_times += 1
                print(f"  (No improvement x{trigger_times}/{PATIENCE})", flush=True)

            # Luu checkpoint
            torch.save({
                'epoch': epoch, 'backbone': backbone.state_dict(),
                'margin': margin_layer.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_loss': best_val_loss, 'best_epoch': best_epoch,
                'trigger_times': trigger_times, 'num_classes': num_classes,
                'batch_size': BATCH_SIZE,
            }, CHECKPOINT_PATH)
            print(f"  Checkpoint saved -> {CHECKPOINT_PATH}", flush=True)

            if trigger_times >= PATIENCE:
                print(f"\n  Early Stopping tai Epoch {epoch}.", flush=True)
                break

        print(f"\n{'='*55}", flush=True)
        print(f"  HOAN TAT! Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}", flush=True)
        print(f"  Model luu tai: {os.path.abspath(BEST_MODEL_PATH)}", flush=True)
        print(f"{'='*55}", flush=True)

main()
```

---

```
%md
### Cell 8: Convert Model sang `.onnx` va Upload len Cloudflare R2
Cell nay thuc hien 3 cong viec tu dong:
1. Convert model `.pth` sang `.onnx` (nhe, nhanh cho production).
2. Nen ca 2 file thanh `.zip`.
3. Upload tat ca len Cloudflare R2 (S3-compatible storage).
```

```python
import subprocess, sys, os, importlib

# Dam bao extra_libs nam trong sys.path (giong Cell 1)
EXTRA_LIBS = os.path.abspath("./extra_libs")
os.makedirs(EXTRA_LIBS, exist_ok=True)
if EXTRA_LIBS not in sys.path:
    sys.path.insert(0, EXTRA_LIBS)

# Cai onnx + onnxruntime vao extra_libs (tranh loi Access Denied cua venv)
try:
    import onnx
except ImportError:
    print("[INFO] Chua co onnx. Dang cai vao extra_libs...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", EXTRA_LIBS,
         "onnx", "onnxruntime"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("[OK] Cai onnx thanh cong!")
    else:
        print(f"[LOI] {r.stderr[-500:]}")
    importlib.invalidate_caches()
    import onnx

import torch
import torch.nn as nn
from torchvision.models import resnet50
import zipfile

# Cai boto3 vao extra_libs neu chua co
try:
    import boto3
except ImportError:
    print("[INFO] Chua co boto3. Dang cai...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", EXTRA_LIBS, "boto3"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("[OK] Cai boto3 thanh cong!")
    else:
        print(f"[LOI] {r.stderr[-500:]}")
    importlib.invalidate_caches()
    import boto3

# ================= CAU HINH =================
drive_base = './Workspace'
PTH_PATH = os.path.join(drive_base, 'FaceModels', 'arcface_best_model_v3.pth')
ONNX_PATH = os.path.join(drive_base, 'FaceModels', 'arcface_best_model_v3.onnx')
ZIP_PATH = os.path.join(drive_base, 'FaceModels', 'ArcFace_Model_Export.zip')


# ================= 1. CONVERT .pth -> .onnx =================
print("1. Convert .pth -> .onnx (Cho suy luan Production)...")
print(f"   Loading model tu {PTH_PATH}...")
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, 512)
model.load_state_dict(torch.load(PTH_PATH, map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(model, dummy_input, ONNX_PATH,
                  export_params=True, opset_version=14,
                  do_constant_folding=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f"   -> Convert ONNX thanh cong: {ONNX_PATH}")

# ================= 2. NEN ZIP =================
print("\n2. Nen file thanh ZIP...")
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(PTH_PATH, os.path.basename(PTH_PATH))
    zipf.write(ONNX_PATH, os.path.basename(ONNX_PATH))
zip_size_mb = os.path.getsize(ZIP_PATH) / (1024*1024)
print(f"   -> Zip thanh cong: {ZIP_PATH} ({zip_size_mb:.1f} MB)")

# ================= 3. UPLOAD LEN CLOUDFLARE R2 =================
print("\n3. Upload len Cloudflare R2...")

R2_ACCESS_KEY_ID = "a7684a3235bf1f8e3870d82c6dc5ef69"
R2_SECRET_ACCESS_KEY = "a8bf552923ce489626300dc18fe320b3aebba50d52f1439599ce43f955395833"
R2_ENDPOINT = "https://7970c4a57482708b85fec0d3b79dba4d.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "edu-learning-storage"
R2_FOLDER = "models/arcface"  # Thu muc tren R2

s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto'
)

upload_files = [PTH_PATH, ONNX_PATH, ZIP_PATH]

for fpath in upload_files:
    fname = os.path.basename(fpath)
    fsize = os.path.getsize(fpath) / (1024*1024)
    r2_key = f"{R2_FOLDER}/{fname}"
    
    print(f"   Uploading {fname} ({fsize:.1f} MB) -> {r2_key} ...")
    s3.upload_file(fpath, R2_BUCKET_NAME, r2_key)
    print(f"   [OK] {fname} da upload thanh cong!")

print("\n" + "="*60)
print("  HOAN TAT! Tat ca file da upload len Cloudflare R2:")
print(f"  Bucket : {R2_BUCKET_NAME}")
print(f"  Folder : {R2_FOLDER}/")
for fpath in upload_files:
    print(f"    - {os.path.basename(fpath)}")
print("="*60)
```
