# 🚦 Traffic Sign Recognition System (SVM + Hybrid Features)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)

Hệ thống nhận diện biển báo giao thông (GTSRB) chuyên nghiệp sử dụng thuật toán **Support Vector Machine (SVM)** kết hợp kỹ thuật trích xuất đặc trưng lai ghép (**Hybrid Feature Extraction**) và minh bạch toán học (Mathematical Transparency).

---

## 🔬 Tổng quan về Khoa học dữ liệu (Data Science Overview)

Dự án này vượt qua giới hạn của "hộp đen" AI bằng cách cung cấp cái nhìn chi tiết về quy trình ra quyết định của mô hình.

### 1. Đặc trưng Lai ghép (Hybrid Features - 1812 dims)
Sự kết hợp giữa hình học và màu sắc giúp mô hình đạt độ chính xác cao ngay cả với thuật toán ML truyền thống:
- **HOG (Histogram of Oriented Gradients)**: 1764 chiều đặc trưng (pixels_per_cell=4x4), tập trung vào các đường biên và hình dạng vật thể.
- **HSV Histograms**: 48 chiều đặc trưng (16 bins mỗi kênh H, S, V), giúp phân biệt màu sắc biển báo (Cấm - Đỏ, Chú ý - Vàng, Chỉ dẫn - Xanh).

### 2. Thuật toán Học máy (Model Implementation)
- **Mô hình**: SVC (Support Vector Classification).
- **Kernel**: Radial Basis Function (RBF) - tối ưu cho không gian đặc trưng phi tuyến tính phức tạp.
- **Optimization**: GridSearchCV cho siêu tham số $C$ và $\gamma$.

### 3. Minh bạch Toán học (REQ008)
- **Visualization**: Trực quan hóa bản đồ Gradient (HOG Image) và Color Bins ngay trên UI.
- **Confidence Scopes**: Tính toán độ tin cậy dựa trên hàm quyết định (Decision Function) và Sigmoid mapping.

---

## 📂 Cấu trúc dự án (Scientific Structure)

```text
├── docs/               # Tài liệu khoa học (Issue Lists, Requirements, Learned Lessons)
├── models/             # Lưu trữ Weights vật lý (model.pkl, scaler.joblib)
├── src/                # Nhân lõi (Training pipeline, Feature Engineering)
├── streamlit/          # Lớp giao diện (Frontend Logic)
│   ├── src/            # Module hỗ trợ UI (Model Handler, Data Processor)
│   ├── assets/         # Tài nguyên tĩnh (CSS Styles)
│   └── app.py          # Entry point điều khiển ứng dụng
└── README.md           # Tài liệu hướng dẫn (File này)
```

---

## 🛠 Hướng dẫn Triển khai (Steps to Reproduce)

### 1. Khởi tạo môi trường
```bash
pip install -r requirements.txt
```

### 2. Đồng bộ hóa Scaler (Quan trọng)
Cần đảm bảo bộ chuẩn hóa khớp với mô hình 1812 chiều:
```bash
python streamlit/scripts/rebuild_scaler.py
```

### 3. Vận hành Web App (v4.1)
```bash
streamlit run streamlit/app.py
```

---

## 📊 Cơ sở lý thuyết nền tảng

Hệ thống tuân thủ các nguyên lý toán học:
- **HOG Calculation**: $G = \sqrt{G_x^2 + G_y^2}$
- **SVM Hyperplane**: $f(x) = \text{sgn}(\sum \alpha_i y_i K(x_i, x) + b)$

---
*Tài liệu này được cập nhật tự động và bám sát tiến độ phát triển hiện tại của dự án.*
