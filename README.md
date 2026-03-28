# Hệ thống Nhận diện và Phát hiện Biển báo Giao thông v4.0 (Full Hybrid)

Dự án này triển khai một hệ thống AI tiên tiến kết hợp giữa **Học sâu (Deep Learning)** và **Học máy truyền thống (SVM)** để giải quyết cả hai tác vụ: **Phát hiện (Detection)** và **Nhận diện (Recognition)** biển báo giao thông.

## 🌟 Tính năng chính
- **Chế độ Dự đoán nhanh**: Tải lên ảnh biển báo đã cắt để nhận diện ngay lập tức với độ chính xác cao.
- **Chế độ Phát hiện & Nhận diện**: Tự động tìm kiếm nhiều biển báo bên trong một bức ảnh phong cảnh (Toàn cảnh) và định danh từng biển báo.
- **Minh bạch Toán học**: Giải trình chi tiết quy trình trích xuất đặc trưng sâu (Deep Features) và cơ chế phân loại Maximum Margin của SVM.

## 🏗️ Kiến trúc Hệ thống (Hybrid v4.0)

### 1. Quy trình Phát hiện (Detection)
Sử dụng phương pháp quét vùng ứng viên dựa trên màu sắc và hình dạng:
- **Tiền xử lý**: Lọc màu HSV (Đỏ/Xanh) + Xử lý hình thái học (Morphology).
- **Trích xuất đặc trưng**: HOG (Histogram of Oriented Gradients) với 324 đặc trưng.
- **Phân loại**: SVM Binary Classifier để xác định vùng đó có phải biển báo hay không.
- **Hậu xử lý**: NMS (Non-Maximum Suppression) để làm sạch các vùng chồng lấn.

### 2. Quy trình Nhận diện (Recognition)
Sử dụng sức mạnh của CNN để hiểu cấu trúc ảnh tinh vi:
- **Feature Extractor**: CNN (Keras) trích xuất 256 đặc trưng sâu từ ảnh 32x32.
- **Classifier**: Linear SVM phân loại 43 nhóm biển báo (Độ tin cậy dựa trên Decision Function).

## 🚀 Hướng dẫn Chạy ứng dụng

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.10+ và các thư viện cần thiết:
```bash
pip install streamlit tensorflow opencv-python scikit-learn scikit-image joblib pandas numpy matplotlib
```

### 2. Chạy giao diện Streamlit
Di chuyển vào thư mục `streamlit/` và thực thi:
```bash
streamlit run app.py
```

## 📂 Cấu trúc dự án
- `streamlit/app.py`: Giao diện chính của ứng dụng.
- `streamlit/src/detector.py`: Logic phát hiện biển báo toàn cảnh.
- `streamlit/src/model_handler.py`: Quản lý nạp và dự đoán mô hình.
- `streamlit/models/`: Chứa các file mô hình `.h5` và `.pkl`.
- `docs/`: Tài liệu hướng dẫn, bài học kinh nghiệm và điểm yếu của hệ thống.

---
**Phát triển bởi chuyên gia AI - Phiên bản v4.0 Full Integration**
