# 🛡️ Fine-Tune MiniFASNet v2 — Multi-Source Anti-Spoofing

**Mục tiêu:** Fine-tune MiniFASNet trên **nhiều dataset kết hợp** để model không overfit theo 1 camera/ánh sáng.

| Thông số | Giá trị                                                   |
| -------- | --------------------------------------------------------- |
| Model    | MiniFASNetV2 (MobileNetV2 backbone, ~2.2M params)         |
| Input    | `(B, 3, 128, 128)` RGB normalized `[0,1]`                 |
| Output   | `(B, 2)` logits `[real_logit, spoof_logit]`               |
| Datasets | OULU-NPU + Replay-Attack + CASIA-FASD + AxonData + Custom |
| Platform | Apache Zeppelin (Windows, GPU)                            |

### Tại sao phải Multi-Source?

| Cách train                     | Kết quả                                                |
| ------------------------------ | ------------------------------------------------------ |
| 1 dataset                      | Học thuộc camera/ánh sáng dataset đó → fail ngoài đời  |
| Multi-dataset + domain shuffle | Hiểu bản chất texture giả → hoạt động ở mọi môi trường |

## 📋 Step 1: Kiểm tra môi trường & GPU

```python
import subprocess, sys, os, gc

os.environ['TORCHDYNAMO_DISABLE'] = '1'
for k in list(sys.modules.keys()):
    if 'torch._dynamo' in k: del sys.modules[k]

EXTRA_LIBS = os.path.abspath('./extra_libs')
os.makedirs(EXTRA_LIBS, exist_ok=True)
sys.path.insert(0, EXTRA_LIBS)

def install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--target', EXTRA_LIBS, pkg])

LIBS_TO_INSTALL = {
    'huggingface_hub': 'huggingface_hub',
    'datasets': 'datasets',
    'tqdm': 'tqdm',
    'onnx': 'onnx',
    'onnxruntime': 'onnxruntime',
    'opencv-python': 'cv2',
    'torchvision': 'torchvision',
    'pillow': 'PIL',
}
for pkg, import_name in LIBS_TO_INSTALL.items():
    try: __import__(import_name)
    except ImportError: print(f'Installing {pkg}...'); install(pkg)

import torch, numpy as np, time
print(f'Python  : {sys.version.split()[0]}')
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} | {p.total_memory/1024**3:.1f} GB')
```

## 🔑 Step 2: HuggingFace Login

```python
from huggingface_hub import login
HF_TOKEN = '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'
login(token=HF_TOKEN)
print('OK!')
```

## 📥 Step 3: Tải Multi-Source Datasets

**FIX:** Tải vào thư mục TẠM ngắn gọn trước (tránh lỗi Windows MAX_PATH trên NAS), rồi copy sang thư mục chính.

```python
import sys, os, time, tempfile, shutil
sys.path.insert(0, os.path.abspath('./extra_libs'))
from huggingface_hub import snapshot_download

HF_TOKEN = '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'

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
os.makedirs(WORK_DIR, exist_ok=True)
DATA_ROOT = os.path.join(WORK_DIR, 'AntiSpoofData')
os.makedirs(DATA_ROOT, exist_ok=True)

# Thu muc tam NGAN GON de tranh loi Windows MAX_PATH (260 ky tu)
# snapshot_download tao .cache/huggingface/download/ ben trong local_dir
# => Neu local_dir nam tren NAS (duong dan dai) => loi [Errno 2]
# => Tai vao temp truoc, roi copy sang NAS
TMP_DL = os.path.join(tempfile.gettempdir(), 'hfdl')
os.makedirs(TMP_DL, exist_ok=True)
print(f'Temp download dir: {TMP_DL}')

DATASETS = {
    'Selfie_Real': 'AxonData/Selfie_and_Official_ID_Photo_Dataset',
    'Replay_Attack_Mobile': 'AxonData/Replay_attack_mobile',
    'Display_Attack': 'AxonData/Display_replay_attacks',
    'Print_Attack': 'AxonData/Anti_Spoofing_Cut_print_attack',
}

for name, repo in DATASETS.items():
    final_dir = os.path.join(DATA_ROOT, name)
    tmp_dir = os.path.join(TMP_DL, name)

    # Kiem tra xem da tai thanh cong truoc do chua
    if os.path.exists(final_dir):
        parquets = [f for f in os.listdir(final_dir) if f.endswith('.parquet')]
        images = [f for f in os.listdir(final_dir) if f.endswith(('.jpg','.png'))]
        if parquets or images:
            print(f'[SKIP] {name}: da co {len(parquets)} parquet, {len(images)} images')
            continue

    print(f'[DOWNLOAD] {repo} -> {name}')
    t0 = time.time()
    try:
        # Tai vao thu muc TMP ngan gon (VD: C:/Users/.../Temp/hfdl/Selfie_Real)
        snapshot_download(
            repo_id=repo,
            repo_type='dataset',
            local_dir=tmp_dir,
            token=HF_TOKEN,
        )

        # Copy tu tmp sang NAS (chi copy file data, bo .cache)
        os.makedirs(final_dir, exist_ok=True)
        for root, dirs, files in os.walk(tmp_dir):
            # Bo qua thu muc .cache
            dirs[:] = [d for d in dirs if d != '.cache']
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, tmp_dir)
                dst = os.path.join(final_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        # Xoa tmp sau khi copy xong
        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f'  OK ({time.time()-t0:.0f}s)')
    except Exception as e:
        print(f'  ERROR: {e}')

# ===== Offline / Manual =====
MANUAL_DIRS = {
    'CelebA_Spoof': os.path.join(DATA_ROOT, 'CelebA_Spoof'),
    'Custom_Real': os.path.join(DATA_ROOT, 'Custom_Real'),
    'Custom_Spoof': os.path.join(DATA_ROOT, 'Custom_Spoof'),
}
for name, path in MANUAL_DIRS.items():
    os.makedirs(path, exist_ok=True)
    n = len([f for f in os.listdir(path) if not f.startswith('.')]) if os.path.exists(path) else 0
    status = f'{n} items' if n > 0 else 'EMPTY'
    print(f'  {name}: {status}')
```

## 🎬 Step 4: Trích xuất Frame từ Video (cho datasets dạng video)

Nhiều dataset anti-spoofing dùng video `.mp4`. Script này extract frames tự động.

```python
import sys, os, glob, cv2
sys.path.insert(0, os.path.abspath('./extra_libs'))
from tqdm import tqdm

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
os.makedirs(WORK_DIR, exist_ok=True)
drive_base = WORK_DIR
DATA_ROOT = os.path.join(drive_base, 'AntiSpoofData')

def extract_frames(video_dir, output_dir, max_frames_per_video=10, target_size=128):
    '''Extract evenly-spaced frames from all videos in a directory.'''
    os.makedirs(output_dir, exist_ok=True)
    videos = glob.glob(os.path.join(video_dir, '**/*.mp4'), recursive=True)
    videos += glob.glob(os.path.join(video_dir, '**/*.avi'), recursive=True)
    if not videos:
        print(f'  No videos found in {video_dir}')
        return 0

    count = 0
    for vpath in tqdm(videos, desc='Extracting'):
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); continue
        step = max(1, total // max_frames_per_video)
        vname = os.path.splitext(os.path.basename(vpath))[0]
        for fi in range(0, min(total, max_frames_per_video * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (target_size, target_size))
            out_name = f'{vname}_f{fi:04d}.jpg'
            cv2.imwrite(os.path.join(output_dir, out_name), frame)
            count += 1
        cap.release()
    return count

# Extract từ tất cả thư mục có video
for sub in os.listdir(DATA_ROOT):
    sub_path = os.path.join(DATA_ROOT, sub)
    if not os.path.isdir(sub_path): continue
    videos = glob.glob(os.path.join(sub_path, '**/*.mp4'), recursive=True)
    videos += glob.glob(os.path.join(sub_path, '**/*.avi'), recursive=True)
    if not videos: continue
    frames_dir = os.path.join(sub_path, '_extracted_frames')
    existing = len(glob.glob(os.path.join(frames_dir, '*.jpg')))
    if existing > 0:
        print(f'[SKIP] {sub}: {existing} frames already extracted')
        continue
    print(f'\n[EXTRACT] {sub}: {len(videos)} videos')
    n = extract_frames(sub_path, frames_dir, max_frames_per_video=10)
    print(f'  Extracted {n} frames -> {frames_dir}')
```

## 🏗️ Step 5: Xây dựng Multi-Source Dataset + Domain Shuffle

**Key insight:** Gộp tất cả data thành 1 dataset duy nhất, shuffle xáo trộn domain,
để model không học theo pattern của 1 camera/ánh sáng cụ thể.

| Label | Ý nghĩa         | Sources                                                                            |
| ----- | --------------- | ---------------------------------------------------------------------------------- |
| 0     | Real (mặt thật) | Selfie_Real + CelebA live + Custom_Real                                            |
| 1     | Spoof (giả mạo) | Replay_Attack_Mobile + Display_Attack + Print_Attack + CelebA spoof + Custom_Spoof |

```python
import sys, os, glob, random
sys.path.insert(0, os.path.abspath('./extra_libs'))
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
os.makedirs(WORK_DIR, exist_ok=True)
drive_base = WORK_DIR
DATA_ROOT = os.path.join(drive_base, 'AntiSpoofData')

# ===== DOMAIN CONFIG: ten thu muc -> (label, type) =====
# Ten phai KHOP CHINH XAC voi ten folder trong AntiSpoofData/
DOMAIN_MAP = {
    # HuggingFace datasets (co file .parquet)
    'Selfie_Real': (0, 'hf'),                  # 0 = REAL
    'Replay_Attack_Mobile': (1, 'hf'),         # 1 = SPOOF
    'Display_Attack': (1, 'hf'),               # 1 = SPOOF
    'Print_Attack': (1, 'hf'),                 # 1 = SPOOF
    # Offline / Manual datasets
    'CelebA_Spoof': (None, 'folder'),          # Auto detect live/spoof subfolders
    'Custom_Real': (0, 'folder'),              # 0 = REAL
    'Custom_Spoof': (1, 'folder'),             # 1 = SPOOF
}

class MultiSourceAntiSpoofDataset(Dataset):
    '''Multi-source anti-spoofing dataset with domain shuffle.
    Tải data từ nhiều nguồn, gộp lại, shuffle domain.'''

    IMG_SIZE = 128

    def __init__(self, data_root, domain_map):
        self.samples = []  # [(path_or_ref, label, domain_name)]
        self._hf_cache = {}
        self.train_tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.4, 0.4, 0.3, 0.15),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(0.15, p=0.3),
            transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
        ])
        self.val_tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.is_train = True
        self._load_all(data_root, domain_map)

    def _find_images(self, path):
        imgs = []
        for ext in ['*.jpg','*.jpeg','*.png','*.bmp']:
            imgs += glob.glob(os.path.join(path, '**', ext), recursive=True)
        return imgs

    def _load_all(self, root, dmap):
        print('Loading multi-source dataset...')
        for name, (label, dtype) in dmap.items():
            path = os.path.join(root, name)
            if not os.path.exists(path): continue

            if dtype == 'hf':
                parquets = glob.glob(os.path.join(path, '**/*.parquet'), recursive=True)
                if parquets:
                    ds = load_dataset('parquet', data_files=parquets, split='train')
                    self._hf_cache[name] = ds
                    for i in range(len(ds)):
                        self.samples.append(('hf', name, i, label, name))
                    print(f'  {name}: {len(ds)} (parquet, label={label})')
                else:
                    imgs = self._find_images(path)
                    for p in imgs:
                        self.samples.append(('file', p, None, label, name))
                    print(f'  {name}: {len(imgs)} (images, label={label})')

            elif dtype == 'folder':
                if label is not None:
                    imgs = self._find_images(path)
                    for p in imgs:
                        self.samples.append(('file', p, None, label, name))
                    if imgs: print(f'  {name}: {len(imgs)} (folder, label={label})')
                else:
                    # Auto-detect: subfolder 'live'/'real' = 0, 'spoof'/'attack'/'fake' = 1
                    for sub in os.listdir(path):
                        sp = os.path.join(path, sub)
                        if not os.path.isdir(sp): continue
                        sl = sub.lower()
                        if any(k in sl for k in ['live','real','genuine']): lbl = 0
                        elif any(k in sl for k in ['spoof','attack','fake','print','replay']): lbl = 1
                        else: continue
                        imgs = self._find_images(sp)
                        for p in imgs:
                            self.samples.append(('file', p, None, lbl, f'{name}/{sub}'))
                        if imgs: print(f'  {name}/{sub}: {len(imgs)} (auto, label={lbl})')

        # Domain shuffle
        random.shuffle(self.samples)
        labels = [s[3] for s in self.samples]
        n_real = labels.count(0)
        n_spoof = labels.count(1)
        domains = set(s[4] for s in self.samples)
        print(f'\nTotal: {len(self.samples)} | Real: {n_real} | Spoof: {n_spoof} | Domains: {len(domains)}')

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            if s[0] == 'hf':
                ds = self._hf_cache[s[1]]
                item = ds[s[2]]
                img = item.get('image')
                img = np.array(img.convert('RGB')) if img else np.zeros((128,128,3), dtype=np.uint8)
            else:
                img = cv2.imread(s[1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((128,128,3), dtype=np.uint8)
        except Exception:
            img = np.zeros((128,128,3), dtype=np.uint8)
        tfm = self.train_tfm if self.is_train else self.val_tfm
        return tfm(img), torch.tensor(s[3], dtype=torch.long)

# ===== BUILD =====
full_ds = MultiSourceAntiSpoofDataset(DATA_ROOT, DOMAIN_MAP)
val_n = min(max(50, int(len(full_ds) * 0.15)), 200)
train_n = len(full_ds) - val_n
train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))
print(f'Train: {train_n} | Val: {val_n}')
```

## 🧠 Step 6: MiniFASNetV2 Architecture + Auto Batch Size Finder

Định nghĩa kiến trúc model và tìm batch size tối ưu cho GPU.

| Thành phần | Chi tiết                                                    |
| ---------- | ----------------------------------------------------------- |
| Backbone   | MobileNetV2 (pretrained ImageNet)                           |
| Head       | Dropout → Linear(1280,128) → ReLU → Dropout → Linear(128,2) |
| Parameters | ~2.2M                                                       |
| Auto Batch | Thử batch size từ 32→2048, chọn lớn nhất không OOM          |

```python
import sys, os, gc
os.environ['TORCHDYNAMO_DISABLE'] = '1'
for k in list(sys.modules.keys()):
    if 'torch._dynamo' in k: del sys.modules[k]

EXTRA_LIBS = os.path.abspath('./extra_libs')
sys.path.insert(0, EXTRA_LIBS)

import torch, torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# =================== MiniFASNetV2 Architecture ===================
class MiniFASNetV2(nn.Module):
    '''MobileNetV2 backbone + lightweight head for anti-spoofing.
    Input:  (B, 3, 128, 128) RGB normalized [0,1]
    Output: (B, 2) logits [real_logit, spoof_logit]
    ~2.2M parameters.
    '''
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1280, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))

# =================== Model Summary ===================
model = MiniFASNetV2(num_classes=2)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'MiniFASNetV2 Architecture:')
print(f'  Total params:     {total_params:,}')
print(f'  Trainable params: {trainable_params:,}')
print(f'  Input:  (B, 3, 128, 128)')
print(f'  Output: (B, 2)')

# Verify forward pass
dummy = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    out = model(dummy)
print(f'  Test forward: input={dummy.shape} -> output={out.shape}')
print(f'  Output values: {out[0].tolist()}')
del model, dummy, out

# =================== Auto Batch Size Finder ===================
def auto_find_batch(dev, safety=0.85):
    '''Tự tìm batch size lớn nhất mà GPU chịu được.
    Thử từ nhỏ đến lớn, dừng khi OOM hoặc VRAM > 90%.'''
    print('\n  Auto Batch Finder...')
    m = MiniFASNetV2().to(dev)
    c = nn.CrossEntropyLoss()
    s = torch.amp.GradScaler('cuda')
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    mx = 32
    for bs in [32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048]:
        o.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); gc.collect()
        try:
            x = torch.randn(bs, 3, 128, 128, device=dev)
            y = torch.randint(0, 2, (bs,), device=dev)
            with torch.amp.autocast('cuda'):
                loss = c(m(x), y)
            s.scale(loss).backward()
            s.step(o); s.update()
            v = torch.cuda.max_memory_allocated(dev) / 1024**3
            vt = torch.cuda.get_device_properties(dev).total_memory / 1024**3
            mx = bs
            print(f'    Batch {bs:>5} -> {v:.1f}/{vt:.1f}GB ({v/vt*100:.0f}%) OK')
            del x, y, loss
            torch.cuda.reset_peak_memory_stats(dev)
            if v / vt > 0.9: break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'    Batch {bs:>5} -> OOM! Max={mx}')
                torch.cuda.empty_cache(); gc.collect()
                break
            raise
    del m, c, s, o
    torch.cuda.empty_cache(); gc.collect()
    safe = max(32, (int(mx * safety) // 32) * 32)
    print(f'  >> Batch size: {safe}')
    return safe

# =================== Run Batch Finder ===================
GPU_ID = 1  # Doi GPU o day

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
os.makedirs(WORK_DIR, exist_ok=True)
SAVE_DIR = os.path.join(WORK_DIR, 'AntiSpoofModels')
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_CACHE = os.path.join(SAVE_DIR, 'batch_size.txt')

if torch.cuda.is_available():
    device = torch.device(f'cuda:{GPU_ID}')
    print(f'\nGPU: {torch.cuda.get_device_name(GPU_ID)}')
    if os.path.exists(BATCH_CACHE):
        BATCH_SIZE = int(open(BATCH_CACHE).read().strip())
        print(f'Batch size (cached): {BATCH_SIZE}')
    else:
        BATCH_SIZE = auto_find_batch(device)
        with open(BATCH_CACHE, 'w') as f: f.write(str(BATCH_SIZE))
        print(f'Batch size saved to cache: {BATCH_SIZE}')
else:
    print('WARNING: No CUDA GPU available!')
    BATCH_SIZE = 32
    print(f'Using default batch size: {BATCH_SIZE}')

```

## 🏋️ Step 7: Training (SELF-CONTAINED — Chạy lại được khi bị đứt)

**Cell này TỰ ĐỦ.** Nếu Zeppelin bị mất kết nối, chỉ cần chạy lại cell này:

1. ✅ Tự rebuild dataset (dùng data đã tải ở Step 3)
2. ✅ Tự tạo lại model + optimizer
3. ✅ Tự load checkpoint và resume từ epoch đã dừng
4. ✅ Tự tìm batch size tối ưu (hoặc dùng cache từ lần trước)

**Không cần chạy lại Step 1-6!**

```python
# =====================================================================
#  SELF-CONTAINED TRAINING CELL — Chay lai duoc khi bi dut ket noi
#  Boc trong def main() de Zeppelin khong auto-print model/tensor
# =====================================================================

import sys, os, time, gc, glob, random, warnings
warnings.filterwarnings('ignore')

os.environ['TORCHDYNAMO_DISABLE'] = '1'
for k in list(sys.modules.keys()):
    if 'torch._dynamo' in k: del sys.modules[k]

EXTRA_LIBS = os.path.abspath('./extra_libs')
sys.path.insert(0, EXTRA_LIBS)

import numpy as np, cv2
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import load_dataset
from tqdm import tqdm

# === CHAN ZEPPELIN TU DONG IN OBJECT ===
_orig_displayhook = sys.displayhook
def _silent_displayhook(obj):
    if obj is not None:
        pass  # Nuot het, chi print() moi hien
sys.displayhook = _silent_displayhook

# =================== CLASS DEFINITIONS (ngoai main) ===================
class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1280, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))
    def __repr__(self):
        return f'MiniFASNetV2(params={sum(p.numel() for p in self.parameters()):,})'

IMG_SIZE = 128

class _Dataset(Dataset):
    def __init__(self, root, dmap):
        self.samples = []
        self._hf = {}
        self.train_tfm = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.4, 0.4, 0.3, 0.15),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(0.15, p=0.3),
            transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
        ])
        self.val_tfm = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.is_train = True
        self._load(root, dmap)
    def _imgs(self, p):
        r = []
        for e in ['*.jpg','*.jpeg','*.png','*.bmp']:
            r += glob.glob(os.path.join(p, '**', e), recursive=True)
        return r
    def _load(self, root, dmap):
        print('Rebuilding dataset (crash-safe)...')
        for name, (label, dt) in dmap.items():
            path = os.path.join(root, name)
            if not os.path.exists(path): continue
            if dt == 'hf':
                pqs = glob.glob(os.path.join(path, '**/*.parquet'), recursive=True)
                if pqs:
                    try:
                        ds = load_dataset('parquet', data_files=pqs, split='train')
                        self._hf[name] = ds
                        for i in range(len(ds)): self.samples.append(('hf',name,i,label,name))
                        print(f'  {name}: {len(ds)} (parquet, label={label})')
                    except Exception as e:
                        print(f'  WARN: Failed to load {name}: {e}')
                else:
                    imgs = self._imgs(path)
                    for p2 in imgs: self.samples.append(('file',p2,None,label,name))
                    if imgs: print(f'  {name}: {len(imgs)} (images, label={label})')
            elif dt == 'folder':
                if label is not None:
                    imgs = self._imgs(path)
                    for p2 in imgs: self.samples.append(('file',p2,None,label,name))
                    if imgs: print(f'  {name}: {len(imgs)} (folder, label={label})')
                else:
                    for sub in os.listdir(path):
                        sp = os.path.join(path, sub)
                        if not os.path.isdir(sp): continue
                        sl = sub.lower()
                        if any(k in sl for k in ['live','real','genuine']): lbl = 0
                        elif any(k in sl for k in ['spoof','attack','fake','print','replay']): lbl = 1
                        else: continue
                        imgs = self._imgs(sp)
                        for p2 in imgs: self.samples.append(('file',p2,None,lbl,f'{name}/{sub}'))
                        if imgs: print(f'  {name}/{sub}: {len(imgs)} (auto, label={lbl})')
        random.shuffle(self.samples)
        ls2 = [s[3] for s in self.samples]
        print(f'  Total: {len(self.samples)} | Real: {ls2.count(0)} | Spoof: {ls2.count(1)}')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            if s[0]=='hf':
                it = self._hf[s[1]][s[2]]
                img = it.get('image')
                img = np.array(img.convert('RGB')) if img else np.zeros((128,128,3),dtype=np.uint8)
            else:
                img = cv2.imread(s[1])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if img is not None else np.zeros((128,128,3),dtype=np.uint8)
        except Exception:
            img = np.zeros((128,128,3),dtype=np.uint8)
        return (self.train_tfm if self.is_train else self.val_tfm)(img), torch.tensor(s[3],dtype=torch.long)

class _ValSubset(torch.utils.data.Subset):
    def __getitem__(self, idx):
        self.dataset.is_train = False
        result = super().__getitem__(idx)
        self.dataset.is_train = True
        return result


# =====================================================================
#  MAIN FUNCTION — Tat ca logic chay trong day, Zeppelin ko auto-print
# =====================================================================
def main():
    gc.collect()
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === CONFIG ===
    GPU_ID = 1
    EPOCHS = 25
    FREEZE_EPOCHS = 3
    NUM_WORKERS = 0
    USE_AMP = True
    PATIENCE = 8
    GRAD_CLIP = 5.0

    WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
    os.makedirs(WORK_DIR, exist_ok=True)
    DATA_ROOT = os.path.join(WORK_DIR, 'AntiSpoofData')
    SAVE_DIR = os.path.join(WORK_DIR, 'AntiSpoofModels')
    os.makedirs(SAVE_DIR, exist_ok=True)
    CKPT = os.path.join(SAVE_DIR, 'minifasnet_v2_checkpoint.pth')
    BEST = os.path.join(SAVE_DIR, 'minifasnet_v2_best.pth')
    BATCH_CACHE = os.path.join(SAVE_DIR, 'batch_size.txt')

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{GPU_ID}')
    print(f'Device: {device} — {torch.cuda.get_device_name(GPU_ID)}')

    # === DOMAIN MAP ===
    DOMAIN_MAP = {
        'Selfie_Real': (0, 'hf'),
        'Replay_Attack_Mobile': (1, 'hf'),
        'Display_Attack': (1, 'hf'),
        'Print_Attack': (1, 'hf'),
        'CelebA_Spoof': (None, 'folder'),
        'Custom_Real': (0, 'folder'),
        'Custom_Spoof': (1, 'folder'),
    }

    # === BUILD DATASET ===
    full_ds = _Dataset(DATA_ROOT, DOMAIN_MAP)
    val_n = min(max(50, int(len(full_ds) * 0.15)), 200)
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_ds = _ValSubset(full_ds, val_ds.indices)
    print(f'Split: Train={train_n} Val={val_n}')

    # === AUTO BATCH (cached) ===
    def auto_find_batch(dev, safety=0.85):
        print('\n  Auto Batch Finder...')
        m = MiniFASNetV2().to(dev); c = nn.CrossEntropyLoss()
        s = torch.amp.GradScaler('cuda'); o = torch.optim.Adam(m.parameters(), lr=1e-3)
        mx = 32
        for bs in [32,64,128,256,384,512,640,768,1024,1280,1536,2048]:
            o.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()
            try:
                x = torch.randn(bs,3,128,128,device=dev)
                y = torch.randint(0,2,(bs,),device=dev)
                with torch.amp.autocast('cuda'):
                    loss = c(m(x), y)
                s.scale(loss).backward(); s.step(o); s.update()
                v = torch.cuda.max_memory_allocated(dev)/1024**3
                vt = torch.cuda.get_device_properties(dev).total_memory/1024**3
                mx = bs
                print(f'    Batch {bs:>5} -> {v:.1f}/{vt:.1f}GB ({v/vt*100:.0f}%) OK')
                del x, y, loss
                torch.cuda.reset_peak_memory_stats(dev)
                if v/vt > 0.9:
                    break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f'    Batch {bs:>5} -> OOM! Max={mx}')
                    torch.cuda.empty_cache(); gc.collect(); break
                raise
        del m, c, s, o; torch.cuda.empty_cache(); gc.collect()
        safe = max(32, (int(mx * safety) // 32) * 32)
        print(f'  >> Batch size: {safe}')
        return safe

    if os.path.exists(BATCH_CACHE):
        BATCH_SIZE = int(open(BATCH_CACHE).read().strip())
        print(f'\nBatch size (cached): {BATCH_SIZE}')
    else:
        BATCH_SIZE = auto_find_batch(device)
        with open(BATCH_CACHE, 'w') as f:
            f.write(str(BATCH_SIZE))

    # Cap batch size cho dataset nho
    max_bs = max(32, train_n // 4)
    if BATCH_SIZE > max_bs:
        print(f'  WARNING: Batch {BATCH_SIZE} > train_n//4 ({max_bs}). Giam xuong {max_bs}')
        BATCH_SIZE = max_bs
    print(f'  Final batch size: {BATCH_SIZE} ({train_n // BATCH_SIZE} batches/epoch)')

    # === LOADERS ===
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # === MODEL + OPTIMIZER ===
    model = MiniFASNetV2(num_classes=2).to(device)
    model.train()

    ls = [full_ds.samples[i][3] for i in range(len(full_ds))]
    n0, n1 = ls.count(0), ls.count(1)
    w = torch.tensor([len(ls)/(2*max(n0,1)), len(ls)/(2*max(n1,1))],
                      dtype=torch.float32).to(device)
    print(f'Class weights: Real={w[0]:.2f} Spoof={w[1]:.2f}')

    criterion = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
    ], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

    def freeze_bb():
        for p in model.features.parameters():
            p.requires_grad = False
    def unfreeze_bb():
        for p in model.features.parameters():
            p.requires_grad = True

    # === LOAD CHECKPOINT ===
    start_epoch, best_acc, best_ep, trigger = 1, 0.0, 0, 0
    if os.path.exists(CKPT):
        print(f'\n  CHECKPOINT FOUND! Loading...')
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if 'scaler' in ck:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_acc = ck.get('best_acc', 0)
        best_ep = ck.get('best_epoch', 0)
        trigger = ck.get('trigger', 0)
        print(f'  >> Resume tu Epoch {start_epoch}/{EPOCHS}')
        print(f'  >> Best Acc so far: {best_acc:.2f}% (Epoch {best_ep})')
        print(f'  >> Early stop counter: {trigger}/{PATIENCE}')
        if start_epoch > FREEZE_EPOCHS + 1:
            for _ in range(start_epoch - FREEZE_EPOCHS - 1):
                scheduler.step()
    else:
        print('\n  No checkpoint found. Training from scratch.')

    # === TRAINING LOOP ===
    if len(full_ds) == 0:
        print('ERROR: Dataset rong! Kiem tra DATA_ROOT va chay Step 3 truoc.')
        return
    if start_epoch > EPOCHS:
        print(f'\n  Da train xong {EPOCHS} epochs roi! Khong can chay lai.')
        print(f'  Best: Epoch {best_ep}, Acc: {best_acc:.2f}%')
        return

    sep = '=' * 55
    print(f'\n{sep}')
    print(f'  TRAINING: {train_n} samples, batch={BATCH_SIZE}, {len(train_loader)} batches/ep')
    print(f'  Epochs: {start_epoch} -> {EPOCHS}')
    print(sep)

    for ep in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        if ep <= FREEZE_EPOCHS:
            if ep == start_epoch:
                freeze_bb()
        elif ep == FREEZE_EPOCHS + 1:
            unfreeze_bb()

        model.train()
        tl, tc, tt = 0.0, 0, 0
        phase = 'FROZEN' if ep <= FREEZE_EPOCHS else 'FULL'

        pbar = tqdm(train_loader, desc=f'Epoch {ep}/{EPOCHS} [{phase}]',
                    leave=False, ncols=100)

        for bi, (imgs, lbls) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            tl += loss.item()
            tc += (logits.argmax(1) == lbls).sum().item()
            tt += lbls.size(0)

            vram = torch.cuda.memory_reserved(GPU_ID) / 1024**3
            pbar.set_postfix({
                'Loss': f'{tl/(bi+1):.4f}',
                'Acc': f'{tc/max(tt,1)*100:.1f}%',
                'VRAM': f'{vram:.1f}G'
            })

        pbar.close()

        if ep > FREEZE_EPOCHS:
            scheduler.step()

        # === VALIDATION ===
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    lo = model(imgs)
                    loss = criterion(lo, lbls)
                vc += (lo.argmax(1) == lbls).sum().item()
                vt += lbls.size(0)
                vl += loss.item()

        ta = tc / max(tt, 1) * 100
        va = vc / vt * 100 if vt > 0 else 0
        el = time.time() - t0
        hr = '-' * 41

        print(f'  +{hr}+')
        print(f'  | Train Loss: {tl/len(train_loader):.4f} | Acc: {ta:6.2f}% |')
        print(f'  | Val   Loss: {vl/max(len(val_loader),1):.4f} | Acc: {va:6.2f}% |')
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f'  | Time: {int(el)}s | LR: {cur_lr:.6f}     |')
        print(f'  +{hr}+')

        if va > best_acc:
            best_acc, best_ep, trigger = va, ep, 0
            torch.save(model.state_dict(), BEST)
            print(f'  * NEW BEST: {va:.2f}% -> {BEST}')
        else:
            trigger += 1
            print(f'  (No improve x{trigger}/{PATIENCE})')

        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc, 'best_epoch': best_ep,
            'trigger': trigger, 'batch_size': BATCH_SIZE,
        }, CKPT)
        print(f'  Checkpoint saved (epoch {ep})', flush=True)

        if trigger >= PATIENCE:
            print(f'\n  Early Stop at epoch {ep}.')
            break

    print(f'\n{sep}')
    print(f'  DONE! Best: Epoch {best_ep}, Val Acc: {best_acc:.2f}%')
    print(f'  Model: {os.path.abspath(BEST)}')
    print(sep)


# =====================================================================
#  GOI MAIN — Chi dong nay la "bieu thuc cuoi", tra ve None -> ko in gi
# =====================================================================
main()
sys.displayhook = _orig_displayhook  # Khoi phuc lai cho cell khac
```

## 📦 Step 8: Export ONNX + INT8 Quantize (cũng SELF-CONTAINED)

Cell này cũng tự đủ — tự define lại model class rồi load best weights.

```python
# SELF-CONTAINED: Tu dinh nghia lai model + load best weights + export
import sys, os
sys.path.insert(0, os.path.abspath('./extra_libs'))
import torch, torch.nn as nn, numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1280, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
os.makedirs(WORK_DIR, exist_ok=True)
SAVE_DIR = os.path.join(WORK_DIR, 'AntiSpoofModels')
BEST = os.path.join(SAVE_DIR, 'minifasnet_v2_best.pth')

if not os.path.exists(BEST):
    print('ERROR: Chua co best model! Chay Step 7 truoc.')
else:
    model = MiniFASNetV2(num_classes=2)
    model.load_state_dict(torch.load(BEST, map_location='cpu'))
    model.eval()

    onnx_fp32 = os.path.join(SAVE_DIR, 'anti_spoofing_v2.onnx')
    torch.onnx.export(model, torch.randn(1,3,128,128), onnx_fp32,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}}, opset_version=13)
    print(f'FP32: {onnx_fp32} ({os.path.getsize(onnx_fp32)/1024:.0f} KB)')

    onnx_q = os.path.join(SAVE_DIR, 'anti_spoofing_v2_q.onnx')
    quantize_dynamic(onnx_fp32, onnx_q, weight_type=QuantType.QUInt8)
def freeze_bb():
        for p in model.features.parameters():
            p.requires_grad = False
    def unfreeze_bb():
        for p in model.features.parameters():
            p.requires_grad = True

    # === LOAD CHECKPOINT ===
    start_epoch, best_acc, best_ep, trigger = 1, 0.0, 0, 0
    if os.path.exists(CKPT):
        print(f'\n  CHECKPOINT FOUND! Loading...')
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if 'scaler' in ck:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_acc = ck.get('best_acc', 0)
        best_ep = ck.get('best_epoch', 0)
        trigger = ck.get('trigger', 0)
        print(f'  >> Resume tu Epoch {start_epoch}/{EPOCHS}')
        print(f'  >> Best Acc so far: {best_acc:.2f}% (Epoch {best_ep})')
        print(f'  >> Early stop counter: {trigger}/{PATIENCE}')
        if start_epoch > FREEZE_EPOCHS + 1:
            for _ in range(start_epoch - FREEZE_EPOCHS - 1):
                scheduler.step()
    else:
        print('\n  No checkpoint found. Training from scratch.')

    # === TRAINING LOOP ===
    if len(full_ds) == 0:
        print('ERROR: Dataset rong! Kiem tra DATA_ROOT va chay Step 3 truoc.')
        return
    if start_epoch > EPOCHS:
        print(f'\n  Da train xong {EPOCHS} epochs roi! Khong can chay lai.')
        print(f'  Best: Epoch {best_ep}, Acc: {best_acc:.2f}%')
        return

    sep = '=' * 55
    print(f'\n{sep}')
    print(f'  TRAINING: {train_n} samples, batch={BATCH_SIZE}, {len(train_loader)} batches/ep')
    print(f'  Epochs: {start_epoch} -> {EPOCHS}')
    print(sep)

    for ep in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        if ep <= FREEZE_EPOCHS:
            if ep == start_epoch:
                freeze_bb()
        elif ep == FREEZE_EPOCHS + 1:
            unfreeze_bb()

        model.train()
        tl, tc, tt = 0.0, 0, 0
        phase = 'FROZEN' if ep <= FREEZE_EPOCHS else 'FULL'

        pbar = tqdm(train_loader, desc=f'Epoch {ep}/{EPOCHS} [{phase}]',
                    leave=False, ncols=100)

        for bi, (imgs, lbls) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            tl += loss.item()
            tc += (logits.argmax(1) == lbls).sum().item()
            tt += lbls.size(0)

            vram = torch.cuda.memory_reserved(GPU_ID) / 1024**3
            pbar.set_postfix({
                'Loss': f'{tl/(bi+1):.4f}',
                'Acc': f'{tc/max(tt,1)*100:.1f}%',
                'VRAM': f'{vram:.1f}G'
            })

        pbar.close()

        if ep > FREEZE_EPOCHS:
            scheduler.step()

        # === VALIDATION ===
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    lo = model(imgs)
                    loss = criterion(lo, lbls)
                vc += (lo.argmax(1) == lbls).sum().item()
                vt += lbls.size(0)
                vl += loss.item()

        ta = tc / max(tt, 1) * 100
        va = vc / vt * 100 if vt > 0 else 0
        el = time.time() - t0
        hr = '-' * 41

        print(f'  +{hr}+')
        print(f'  | Train Loss: {tl/len(train_loader):.4f} | Acc: {ta:6.2f}% |')
        print(f'  | Val   Loss: {vl/max(len(val_loader),1):.4f} | Acc: {va:6.2f}% |')
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f'  | Time: {int(el)}s | LR: {cur_lr:.6f}     |')
        print(f'  +{hr}+')

        if va > best_acc:
            best_acc, best_ep, trigger = va, ep, 0
            torch.save(model.state_dict(), BEST)
            print(f'  * NEW BEST: {va:.2f}% -> {BEST}')
        else:
            trigger += 1
            print(f'  (No improve x{trigger}/{PATIENCE})')

        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc, 'best_epoch': best_ep,
            'trigger': trigger, 'batch_size': BATCH_SIZE,
        }, CKPT)
        print(f'  Checkpoint saved (epoch {ep})', flush=True)

        if trigger >= PATIENCE:
            print(f'\n  Early Stop at epoch {ep}.')
            break

    print(f'\n{sep}')
    print(f'  DONE! Best: Epoch {best_ep}, Val Acc: {best_acc:.2f}%')
    print(f'  Model: {os.path.abspath(BEST)}')
    print(sep)


# =====================================================================
#  GOI MAIN — Chi dong nay la "bieu thuc cuoi", tra ve None -> ko in gi
# =====================================================================
main()
sys.displayhook = _orig_displayhook  # Khoi phuc lai cho cell khac
```

## 📦 Step 8: Export ONNX + INT8 Quantize (cũng SELF-CONTAINED)

Cell này cũng tự đủ — tự define lại model class rồi load best weights.

```python
# SELF-CONTAINED: Tu dinh nghia lai model + load best weights + export
import sys, os
sys.path.insert(0, os.path.abspath('./extra_libs'))
import torch, torch.nn as nn, numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1280, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
os.makedirs(WORK_DIR, exist_ok=True)
SAVE_DIR = os.path.join(WORK_DIR, 'AntiSpoofModels')
BEST = os.path.join(SAVE_DIR, 'minifasnet_v2_best.pth')

if not os.path.exists(BEST):
    print('ERROR: Chua co best model! Chay Step 7 truoc.')
else:
    model = MiniFASNetV2(num_classes=2)
    model.load_state_dict(torch.load(BEST, map_location='cpu'))
    model.eval()

    onnx_fp32 = os.path.join(SAVE_DIR, 'anti_spoofing_v2.onnx')
    torch.onnx.export(model, torch.randn(1,3,128,128), onnx_fp32,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}}, opset_version=13)
    print(f'FP32: {onnx_fp32} ({os.path.getsize(onnx_fp32)/1024:.0f} KB)')

    onnx_q = os.path.join(SAVE_DIR, 'anti_spoofing_v2_q.onnx')
    quantize_dynamic(onnx_fp32, onnx_q, weight_type=QuantType.QUInt8)
    print(f'INT8: {onnx_q} ({os.path.getsize(onnx_q)/1024:.0f} KB)')

    sess = ort.InferenceSession(onnx_q, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'input': np.random.randn(1,3,128,128).astype(np.float32)})[0]
    print(f'Verify: shape={out.shape}, values={out[0]}')
    print(f'\n=== EXPORT OK! ===')
    print(f'Copy {onnx_q} -> models/anti_spoofing.onnx')
```

## 🚀 Step 9: Tải Model Về Máy (Upload lên GoFile / File.io)

Trên môi trường Zeppelin / Cloud thường khó tải trực tiếp file về máy cá nhân. Cell dưới đây sẽ nén các file ONNX vừa export vào một file ZIP và dùng `curl` để upload an toàn (không bị lỗi reset connection) lên dịch vụ chia sẻ file ẩn danh (GoFile hoặc File.io) để bạn lấy link tải trực tiếp.

```python
import os
import zipfile
import subprocess
import json

WORK_DIR = os.path.join('.', 'Workspace', 'minifasnetv2')
SAVE_DIR = os.path.join(WORK_DIR, 'AntiSpoofModels')
ZIP_FILE = os.path.join(SAVE_DIR, 'minifasnet_v2_onnx.zip')

files_to_zip = [
    'anti_spoofing_v2.onnx',
    'anti_spoofing_v2_q.onnx'
]

# 1. Nén file
print("1. Đang nén file model...")
curr_dir = os.getcwd()
os.chdir(SAVE_DIR)
with zipfile.ZipFile('minifasnet_v2_onnx.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_zip:
        if os.path.exists(f):
            zf.write(f)
            print(f"   - Đã thêm: {f}")
os.chdir(curr_dir)

print(f"   -> Nén xong: {ZIP_FILE} ({os.path.getsize(ZIP_FILE)/1024/1024:.2f} MB)")

# 2. Upload (Dùng thư viện chuẩn của Python để tránh lỗi kết nối Zeppelin)
print("\n2. Đang kết nối server GoFile.io...")

import urllib.request
import json
import uuid

def upload_multipart(url, file_path):
    boundary = uuid.uuid4().hex
    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    with open(file_path, 'rb') as f:
        file_content = f.read()
    filename = os.path.basename(file_path)

    # Xây dựng HTTP Body cho Multipart form-data
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: application/zip\r\n\r\n'
    ).encode('utf-8') + file_content + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

try:
    # Lấy thông tin server GoFile
    req = urllib.request.Request("https://api.gofile.io/servers")
    with urllib.request.urlopen(req) as response:
        server_data = json.loads(response.read().decode('utf-8'))
        server_name = server_data['data']['servers'][0]['name']

    print(f"   -> Uploading lên {server_name}...")
    upload_url = f"https://{server_name}.gofile.io/contents/uploadfile"

    res_data = upload_multipart(upload_url, ZIP_FILE)
    if res_data.get('status') == 'ok':
        print("\n" + "="*60)
        print("🎉 UPLOAD THÀNH CÔNG (GoFile)!")
        print(f"👉 LINK TẢI VỀ: {res_data['data']['downloadPage']}")
        print("="*60)
    else:
        raise Exception(f"GoFile API Error: {res_data}")

except Exception as e:
    print(f"\n[!] Lỗi upload GoFile ({e}). Chuyển sang File.io...")
    try:
        res_data2 = upload_multipart("https://file.io", ZIP_FILE)
        print("\n" + "="*60)
        print("🎉 UPLOAD THÀNH CÔNG (File.io)!")
        print(f"👉 LINK TẢI VỀ: {res_data2['link']}")
        print("❗ Lưu ý: Link File.io chỉ tải được MỘT LẦN duy nhất!")
        print("="*60)
    except Exception as e2:
        print("\n❌ Cả 2 cách đều thất bại:", e2)
```
