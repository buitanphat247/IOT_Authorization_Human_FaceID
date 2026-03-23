# Hướng Dẫn Fine-tune ArcFace (PyTorch)

Thư mục này chứa mã nguồn để **fine-tune** mô hình ArcFace (sử dụng kiến trúc ResNet) cho bài toán nhận diện khuôn mặt. Việc fine-tune ArcFace là giải pháp tối ưu nhất để cải thiện độ chính xác trên dữ liệu người thật hoặc nhận diện trong điều kiện khó (đeo kính, thiếu sáng, góc nghiêng).

Như bạn đã tìm hiểu, sử dụng dataset **VGGFace2** hoặc kết hợp với dữ liệu nhân viên nội bộ là một chiến lược rất chuẩn xác.

## Cấu Trúc Thư Mục
- `dataset.py`: Định nghĩa Pytorch Dataset để load ảnh face và label. Hỗ trợ Augmentation (cắt xén, lật, đổi màu) giúp model khái quát tốt hơn.
- `train.py`: File thực thi chính vòng lặp huấn luyện (Training Loop). Tại đây chúng ta định nghĩa ArcFace Loss (Additive Angular Margin Loss) và tối ưu hóa trọng số.
- `models/`: Có thể chứa pre-trained weights `.pth` tải về từ InsightFace.

## Yêu Cầu Cài Đặt (Requirements)
Để train, bạn cần cài đặt thư viện PyTorch (khuyến nghị có GPU CUDA):
```bash
pip install torch torchvision
pip install tqdm opencv-python onnx onnxruntime
```

## Các Bước Chuẩn Bị
1. **Tải Dữ Liệu:** Download dữ liệu VGGFace2 (hoặc dataset nội bộ), giải nén vào thư mục `finetune/data`.
   Cấu trúc mong muốn:
   ```
   data/
   ├── user_001/
   │   ├── img1.jpg
   │   └── img2.jpg
   ├── user_002/
   ...
   ```
2. **Pre-trained Model:** Tải ResNet50/ResNet100 ArcFace pre-trained weights (nếu muốn fine-tune từ model có sẵn thay vì train từ đầu ở mức random).
3. **Chạy Training:**
   ```bash
   python finetune/train.py
   ```
   Kết quả model sẽ được lưu ra dạng `.pth` hoặc export sang ONNX để cắm lại vào hệ thống nhận diện `core/detector.py` của chúng ta.
