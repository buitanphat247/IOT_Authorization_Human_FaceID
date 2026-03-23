"""
Chẩn đoán Mode Collapse cho ArcFace model.
So sánh model fine-tuned vs pretrained: tạo embedding từ ảnh ngẫu nhiên
và kiểm tra xem các vector có bị collapse (giống nhau) không.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FINETUNED = os.path.join(MODELS_DIR, "arcface_best_model_v3.onnx")
PRETRAINED = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "w600k_r50.onnx")

def load_session(path):
    if not os.path.exists(path):
        return None
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def get_embedding(sess, img):
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    img = img.astype(np.float32)
    img = np.transpose((img - 127.5) / 127.5, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    emb = sess.run([out_name], {inp_name: img})[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb

def diagnose(name, sess):
    print(f"\n{'='*60}")
    print(f"  MODEL: {name}")
    print(f"  Input: {sess.get_inputs()[0].shape}")
    print(f"  Output: {sess.get_outputs()[0].shape}")
    print(f"{'='*60}")
    
    # Tạo 5 "khuôn mặt" ngẫu nhiên hoàn toàn khác nhau
    np.random.seed(42)
    faces = []
    for i in range(5):
        # Mỗi "mặt" là 1 ảnh random 112x112x3 hoàn toàn khác biệt
        face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        faces.append(face)
    
    # Trích xuất embedding
    embeddings = []
    for i, face in enumerate(faces):
        emb = get_embedding(sess, face)
        embeddings.append(emb)
        print(f"  Face {i}: norm={np.linalg.norm(emb):.4f}, mean={emb.mean():.6f}, std={emb.std():.6f}")
    
    # Tính cosine similarity giữa MỌI cặp
    print(f"\n  Cosine Similarity Matrix (5 ảnh ngẫu nhiên hoàn toàn khác nhau):")
    print(f"  {'':>8}", end="")
    for i in range(5):
        print(f"  Face{i}", end="")
    print()
    
    sims = []
    for i in range(5):
        print(f"  Face{i}: ", end="")
        for j in range(5):
            cos = float(np.dot(embeddings[i], embeddings[j]))
            if i != j:
                sims.append(cos)
            print(f"  {cos:.4f}", end="")
        print()
    
    avg_sim = np.mean(sims)
    min_sim = np.min(sims)
    max_sim = np.max(sims)
    
    print(f"\n  --- KẾT LUẬN ---")
    print(f"  Cosine trung bình giữa các mặt KHÁC NHAU: {avg_sim:.4f}")
    print(f"  Min: {min_sim:.4f} | Max: {max_sim:.4f}")
    
    if avg_sim > 0.8:
        print(f"  ❌ MODE COLLAPSE! Model trả ra vector GIỐNG NHAU cho mọi input.")
        print(f"     → Mọi khuôn mặt sẽ match với nhau → False Accept 100%!")
        return "COLLAPSED"
    elif avg_sim > 0.5:
        print(f"  ⚠️  CẢNH BÁO: Cosine trung bình cao bất thường ({avg_sim:.2f})")
        print(f"     → Model có dấu hiệu collapse nhẹ, cần kiểm tra thêm trên ảnh thật.")
        return "WARNING"
    else:
        print(f"  ✅ Model BÌNH THƯỜNG. Các vector đủ phân biệt.")
        return "OK"

if __name__ == "__main__":
    results = {}
    
    print("=" * 60)
    print("  CHẨN ĐOÁN MODE COLLAPSE - ARCFACE MODEL")
    print("=" * 60)
    
    # Test fine-tuned
    sess_ft = load_session(FINETUNED)
    if sess_ft:
        results["Fine-tuned v3"] = diagnose("Fine-tuned v3 (arcface_best_model_v3.onnx)", sess_ft)
    else:
        print(f"\n  [SKIP] Fine-tuned model không tìm thấy: {FINETUNED}")
    
    # Test pretrained
    sess_pt = load_session(PRETRAINED)
    if sess_pt:
        results["Pretrained"] = diagnose("Pretrained (w600k_r50.onnx)", sess_pt)
    else:
        print(f"\n  [SKIP] Pretrained model không tìm thấy: {PRETRAINED}")
    
    # Summary
    print(f"\n{'='*60}")
    print("  TÓM TẮT")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "❌" if status == "COLLAPSED" else ("⚠️" if status == "WARNING" else "✅")
        print(f"  {icon} {name}: {status}")
    
    if results.get("Fine-tuned v3") in ("COLLAPSED", "WARNING"):
        print(f"\n  → KHUYẾN NGHỊ: Dùng model Pretrained cho production.")
        print(f"  → Model fine-tuned cần retrain với:")
        print(f"     1. Learning rate thấp hơn (backbone_lr=1e-4)")
        print(f"     2. Thêm L2 regularization mạnh hơn")
        print(f"     3. Kiểm tra metric spread (std of embeddings) mỗi epoch")
        print(f"     4. Dùng ArcFace loss với margin nhỏ hơn (m=0.3 thay vì 0.5)")
