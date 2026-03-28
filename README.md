# Dự án: Nhận diện Biển báo Giao thông (GTSRB) - Phiên bản v4.0

Chào mừng Anh đến với phiên bản **v4.0 (Hybrid SVM+CNN)** của hệ thống nhận diện biển báo giao thông. Đây là bản nâng cấp mạnh mẽ nhất, kết hợp Deep Learning và Machine Learning truyền thống.

## 🚀 Tính năng mới trong v4.0

1.  **Mô hình Hybrid (Lai ghép)**:
    -   **Trích xuất đặc trưng**: Sử dụng mạng CNN (Convolutional Neural Network) để tự động học và trích xuất 256 đặc trưng sâu (Deep Features).
    -   **Phân loại**: Sử dụng Linear SVM để tìm ra ranh giới quyết định (Maximum Margin) tối ưu nhất, tăng độ chính xác vượt trội so với HOG+HSV.
2.  **Giao diện Premium v4.0**:
    -   Bố cục tập trung, thẻ kết quả phong cách Card hiện đại.
    -   Tích hợp minh bạch toán học (Mathematical Transparency) với biểu đồ Mermaid và LaTeX.
3.  **Hệ thống Modular**:
    -   `streamlit/app.py`: Giao diện chính.
    -   `streamlit/src/`: Chứa các module xử lý dữ liệu, mô hình và metadata riêng biệt.
    -   `streamlit/models/`: Chứa các tệp mô hình đã huấn luyện.

## 🛠 Hướng dẫn cài đặt

1.  **Cài đặt môi trường**:
    ```powershell
    pip install -r requirements.txt
    ```

2.  **Cấu hình Mô hình**:
    Đảm bảo thư mục `streamlit/models/` có đầy đủ 3 tệp sau:
    -   `cnn_feature_extractor.h5`
    -   `svm_scaler.pkl`
    -   `final_hybrid_svm.pkl`

3.  **Chạy ứng dụng**:
    ```powershell
    streamlit run streamlit/app.py
    ```

## 📂 Cấu trúc thư mục

```text
NhanDienBienBao/
├── streamlit/
│   ├── app.py              # File chạy chính
│   ├── src/                # Logic xử lý tách biệt
│   │   ├── data_processor.py
│   │   ├── model_handler.py
│   │   └── class_metadata.py
│   ├── assets/             # CSS styling
│   └── models/             # Các tệp mô hình v4.0
├── docs/                   # Tài liệu hướng dẫn & Học tập
└── requirements.txt        # Danh sách thư viện
```

## ✍️ Tác giả
Dự án được thực hiện và nâng cấp bởi Antigravity (Assistant AI) dành cho Anh.
