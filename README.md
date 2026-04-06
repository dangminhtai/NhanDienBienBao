# TRAFFIC SIGN RECOGNITION SYSTEM (Hybrid v5.0 Scientific Edition)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20TensorFlow-orange.svg)

## 📖 Tổng quan (Abstract)
Hệ thống này đại diện cho một bước tiến trong việc kết hợp các phương pháp xử lý ảnh thị giác máy tính truyền thống và mạng nơ-ron tích chập (CNN). Mục tiêu cốt lõi là giải quyết hai thách thức kỹ thuật lớn trong lĩnh vực GTSRB (German Traffic Sign Recognition Benchmark):
1. **Detection**: Xác định vị trí các vùng ứng viên trong môi trường thực tế phức tạp.
2. **Recognition**: Phân loại chính xác 43 lớp biển báo với độ tin cậy được định lượng hóa theo phương pháp Maximum Margin (SVM).

---

## 🔬 Kiến trúc Kỹ thuật (Technical Architecture)

Hệ thống hoạt động theo mô hình **Pipeline Phễu Lọc (Funnel Filter Pipeline)** 6 giai đoạn:

### 1. Adaptive Pre-processing
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Thực hiện trong không gian màu **LAB** để tối ưu hóa dynamic range mà không làm sai lệch màu sắc tự nhiên.
- **HSV Space Mapping**: Tách biệt kênh Hue để giảm thiểu nhạy cảm với cường độ ánh sáng biến thiên.

### 2. Candidate Proposal Generation
Sử dụng bộ lọc màu đa kênh kết hợp hình thái học toán học:
- **Masking**: $\text{Mask}_{final} = \text{Mask}_{Red} \cup \text{Mask}_{Blue} \cup \text{Mask}_{Yellow}$
- **Morphology**: Áp dụng $Closing(9,9)$ để lấp đầy vùng lõi và $Opening(3,3)$ để triệt tiêu nhiễu Gauss.

### 3. Structural Analysis & Filtering
- **Geometry Logic**: Kiểm tra tỷ lệ khung hình ($0.6 < AR < 1.4$) và độ đặc (Solidity $> 0.3$).
- **Frequency Analysis**: Sử dụng phương sai **Laplacian** để xác định độ nét (Focus check), đảm bảo chỉ xử lý các ROI có độ chi tiết cao ($Var > 40$).

### 4. Hybrid Feature Extraction (Dual-Path)
- **Pathway A (Deep Features)**: Sử dụng CNN pre-trained trên tập GTSRB để trích xuất vector đặc trưng 256 chiều từ lớp cổ chai (Bottleneck layer).
- **Pathway B (Gradient Features)**: Trích xuất **HOG (Histogram of Oriented Gradients)** (9 orientations, 8x8 pixels/cell) tạo thành vector 324 chiều mô tả cấu trúc hình học.

### 5. Multi-Stage Classification
- **Stage 1 (Detection Verification)**: Sử dụng **Linear SVM** để phân loại nhị phân (Sign vs. Non-sign) dựa trên HOG features.
- **Stage 2 (Fine Grained Recognition)**: Sử dụng **SVM với Kernel RBF** để phân loại đa lớp trên Deep Features từ CNN.

### 6. Post-processing (Clustering)
- **NMS (Non-Maximum Suppression)**: Áp dụng thuật toán triệt tiêu cực đại cục bộ với ngưỡng IoU tùy chỉnh để lọc các box trùng lặp.

---

## 🛠️ Cấu trúc Dự án (Project Structure)

```bash
├── streamlit/
│   ├── app.py                      # Dashboard Entry Point
│   ├── src/
│   │   ├── detector.py             # Logic phát hiện (HSV + HOG + SVM)
│   │   ├── model_handler.py        # Pipeline CNN-SVM Integration
│   │   └── content_manager.py      # Hệ thống CMS Hot-reload (st.cache_data)
│   ├── views/                      # Các module giao diện (Single, Full, Video)
│   ├── components/                 # Các component "AI Anatomy" trực quan hóa
│   ├── config/                     # centralized content management (.json)
│   └── models/                     # Lưu trữ trọng số .h5 và .pkl
├── docs/                           # Kho lưu trữ tri thức toán học và kỹ thuật
└── README.md                       # Tài liệu hệ thống
```

---

## 🚀 Cài đặt & Vận hành

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.10+. Chạy lệnh sau để cài đặt dependencies:

```bash
pip install streamlit tensorflow opencv-python scikit-learn scikit-image joblib pandas numpy matplotlib
```

### 2. Khởi chạy Dashboard
Hệ thống cung cấp giao diện tương tác cao cấp:

```bash
streamlit run streamlit/app.py
```

---

## 📂 Tài liệu Phụ trợ (Extended Documentation)
Để hiểu sâu hơn về toán học và mã nguồn, tham khảo:
- [Kiến trúc Anatomy](file:///f:/X-FILE/Code_UNI/Python/Math%20for%20AI/CuoiKy/NhanDienBienBao/docs/demystify_pipeline.md): Bóc tách chi tiết từng lớp CNN và SVM.
- [Phân tích Overfitting](file:///f:/X-FILE/Code_UNI/Python/Math%20for%20AI/CuoiKy/NhanDienBienBao/docs/diagnosis_overfitting.md): Báo cáo kỹ thuật về quá trình huấn luyện.
- [Kế hoạch phát triển](file:///f:/X-FILE/Code_UNI/Python/Math%20for%20AI/CuoiKy/NhanDienBienBao/docs/next_step.md): Lộ trình v6.0 cho YOLO Lite và Multi-threading.

---
**© 2026 - Senior AI Research Project**  
*Mọi thay đổi văn bản vui lòng thực hiện tại `streamlit/config/content.json`.*
