import os
import sys
import glob
import itertools
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Đảm bảo imports nội bộ hoạt động (trỏ về thư mục core)
core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, core_dir)

from logger import get_logger
from models.detector import FaceDetector
from models.recognizer import FaceRecognizer

logger = get_logger("benchmark")

class FaceBenchmark:
    """Công cụ Benchmark để tìm Threshold tối ưu cho Face Recognition
    dựa trên FAR (False Acceptance Rate) và FRR (False Rejection Rate).
    
    Yêu cầu dataset chuẩn theo cấu trúc:
    dataset_dir/
        person_A/
            img1.jpg
            img2.jpg
        person_B/
            img1.jpg
            img2.jpg
    """
    
    def __init__(self, dataset_dir):
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Thư mục dataset không tồn tại: {dataset_dir}")
            
        self.dataset_dir = dataset_dir
        self.detector = FaceDetector(mode="image", num_faces=1)
        self.recognizer = FaceRecognizer()
        
        self.embeddings = {}  # {person_name: [emb1, emb2, ...]}
        self.log_dir = "benchmark_results"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _extract_all_embeddings(self):
        """Quét toàn bộ dataset và trích xuất embedding."""
        logger.info(f"Đang quét thư mục: {self.dataset_dir}")
        persons = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
        
        if len(persons) < 2:
            logger.error("Cần ít nhất 2 người khác nhau trong dataset để tạo Negative Pairs!")
            return False
            
        total_images = 0
        valid_faces = 0
        
        for person in tqdm(persons, desc="Đang trích xuất embeddings"):
            self.embeddings[person] = []
            person_dir = os.path.join(self.dataset_dir, person)
            
            # Tìm tất cả ảnh jpg, png
            img_paths = glob.glob(os.path.join(person_dir, "*.jpg")) + glob.glob(os.path.join(person_dir, "*.png"))
            total_images += len(img_paths)
            
            for path in img_paths:
                img = cv2.imread(path)
                if img is None: continue
                
                faces = self.detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not faces: continue
                
                # Chỉ lấy khuôn mặt lớn nhất ở giữa
                from models.detector import get_center_face
                h, w = img.shape[:2]
                face = get_center_face(faces, w, h)
                
                if face:
                    emb = self.recognizer.get_embedding(img, face.lm5)
                    if emb is not None:
                        emb = emb / np.linalg.norm(emb) # L2 normalize
                        self.embeddings[person].append(emb)
                        valid_faces += 1
                        
        logger.info(f"Đã trích xuất {valid_faces}/{total_images} khuôn mặt hợp lệ từ {len(persons)} người.")
        return valid_faces > 0

    def generate_pairs(self):
        """Tạo Genuine (cùng người) và Imposter (khác người) pairs."""
        genuine_scores = []
        imposter_scores = []
        
        logger.info("Đang tạo các cặp đối chiếu (Pairwise comparisons)...")
        
        # 1. Genuine pairs (Cùng một người)
        for person, embs in self.embeddings.items():
            if len(embs) >= 2:
                # Tổ hợp chập 2 của tất cả ảnh của người đó
                for emb1, emb2 in itertools.combinations(embs, 2):
                    score = float(np.dot(emb1, emb2))
                    genuine_scores.append(score)
                    
        # 2. Imposter pairs (Khác người)
        persons = list(self.embeddings.keys())
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                
                # Lấy tất cả ảnh của người A so sánh với tất cả ảnh của người B
                for emb1 in self.embeddings[p1]:
                    for emb2 in self.embeddings[p2]:
                        score = float(np.dot(emb1, emb2))
                        imposter_scores.append(score)
                        
        logger.info(f"Tổng số Genuine Pairs (Cùng người): {len(genuine_scores)}")
        logger.info(f"Tổng số Imposter Pairs (Khác người): {len(imposter_scores)}")
        
        return np.array(genuine_scores), np.array(imposter_scores)

    def evaluate(self):
        """Chạy benchmark và tính toán các metrics."""
        if not self._extract_all_embeddings():
            return
            
        gen_scores, imp_scores = self.generate_pairs()
        
        if len(gen_scores) == 0 or len(imp_scores) == 0:
            logger.error("Không đủ dữ liệu để tạo cặp so sánh (Cần mỗi người >= 2 ảnh, và >= 2 người khác nhau).")
            return
            
        # Tính FAR và FRR trên các mức threshold từ 0.0 đến 1.0
        thresholds = np.linspace(0.0, 1.0, 1000)
        far = []
        frr = []
        
        for th in thresholds:
            # Tỷ lệ nhận diện sai người lạ thành người quen: Imposter score >= th
            false_accepts = np.sum(imp_scores >= th)
            far.append(false_accepts / len(imp_scores))
            
            # Tỷ lệ từ chối sai người quen thành người lạ: Genuine score < th
            false_rejects = np.sum(gen_scores < th)
            frr.append(false_rejects / len(gen_scores))
            
        far = np.array(far)
        frr = np.array(frr)
        
        # Tìm EER (Equal Error Rate) - điểm mà FAR gần bằng FRR nhất
        diff = np.abs(far - frr)
        min_diff_idx = np.argmin(diff)
        eer = (far[min_diff_idx] + frr[min_diff_idx]) / 2
        optimal_th = thresholds[min_diff_idx]
        
        # Mức threshold cho bảo mật cao (chấp nhận FRR để đảm bảo FAR <= 0.1%)
        strict_idx = np.where(far <= 0.001)[0]
        strict_th = thresholds[strict_idx[0]] if len(strict_idx) > 0 else optimal_th
        strict_frr = frr[strict_idx[0]] if len(strict_idx) > 0 else 1.0
        
        self._print_results(eer, optimal_th, strict_th, strict_frr, gen_scores, imp_scores)
        self._plot_results(thresholds, far, frr, gen_scores, imp_scores, optimal_th, strict_th)

    def _print_results(self, eer, optimal_th, strict_th, strict_frr, gen_scores, imp_scores):
        """In biểu đồ CLI và thông số."""
        print("\n" + "="*50)
        print("📊 KẾT QUẢ BENCHMARK (DATA-DRIVEN THRESHOLDS)")
        print("="*50)
        print(f"✅ Số lượng người (Classes)  : {len(self.embeddings)}")
        print(f"✅ Số cặp Cùng Người (Gen)   : {len(gen_scores)}")
        print(f"✅ Số cặp Khác Người (Imp)   : {len(imp_scores)}")
        print("-"*50)
        print(f"🎯 Equal Error Rate (EER)    : {eer*100:.3f}%")
        print(f"🔥 NGƯỠNG TỐI ƯU (Optimal)   : {optimal_th:.3f} (Sử dụng Balance)")
        print(f"🔒 NGƯỠNG BẢO MẬT CAO        : {strict_th:.3f} (FAR <= 0.1%, FRR = {strict_frr*100:.2f}%)")
        print("="*50)
        print("💡 Gợi ý cập nhật `config.py`:")
        print(f"  THRESHOLD_ACCEPT = {optimal_th:.2f} (Khuyên dùng)")
        print(f"  THRESHOLD_ACCEPT_HIGH_QUALITY = {min(optimal_th + 0.03, 1.0):.2f}")
        print(f"  THRESHOLD_ACCEPT_LOW_QUALITY = {max(optimal_th - 0.03, 0.0):.2f}")
        print(f"  THRESHOLD_REJECT = {max(optimal_th - 0.15, 0.0):.2f}")
        print("="*50 + "\n")

    def _plot_results(self, thresholds, far, frr, gen_scores, imp_scores, optimal_th, strict_th):
        """Vẽ biểu đồ phân bố và Curve FAR/FRR."""
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Phân bố điểm số theo Histogram
        plt.subplot(1, 2, 1)
        plt.hist(imp_scores, bins=50, alpha=0.6, color='red', label=f'Imposter - max: {np.max(imp_scores):.2f}')
        plt.hist(gen_scores, bins=50, alpha=0.6, color='green', label=f'Genuine - min: {np.min(gen_scores):.2f}')
        plt.axvline(optimal_th, color='blue', linestyle='dashed', linewidth=2, label=f'Optimal Th = {optimal_th:.2f}')
        plt.title('Score Distribution (Phân Bố Cosine Similarity)')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 2: FAR vs FRR Curve
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, far, 'r-', label='FAR (False Accept Rate)')
        plt.plot(thresholds, frr, 'g-', label='FRR (False Reject Rate)')
        plt.axvline(optimal_th, color='blue', linestyle='dashed', label=f'Optimal TH = {optimal_th:.2f}')
        plt.axvline(strict_th, color='black', linestyle='dotted', label=f'Strict TH = {strict_th:.2f} (FAR<=0.1%)')
        
        # Zoom vô vùng quan trọng
        plt.xlim([max(0.0, optimal_th - 0.3), min(1.0, optimal_th + 0.3)])
        plt.ylim([-0.05, 1.05])
        
        plt.title('FAR and FRR vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, "benchmark_thresholds.png")
        plt.savefig(save_path, dpi=300)
        logger.info(f"Đã lưu biểu đồ kết quả tại: {save_path}")
        print(f"\n📸 Biểu đồ đã lưu tại: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tìm Threshold tối ưu (Data-driven Benchmark)")
    parser.add_argument("--dataset", "-d", type=str, required=True, 
                        help="Đường dẫn đến thư mục dataset (gồm các thư mục con chứa ảnh từng người)")
    args = parser.parse_args()
    
    benchmark = FaceBenchmark(args.dataset)
    benchmark.evaluate()
