# Dự Án Nhận Diện Biển Báo Giao Thông (SVM + Streamlit)

Dự án này sử dụng mô hình Support Vector Machine (SVM) kết hợp với đặc trưng HOG để nhận diện 43 loại biển báo giao thông trong bộ dữ liệu GTSRB.

## 🚀 Tính năng chính
- Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) cho độ chính xác cao.
- Tối ưu hóa siêu tham số bằng GridSearchCV.
- Giao diện Web trực quan bằng Streamlit.

## 📂 Cấu trúc dự án
- `src/`: Chứa mã nguồn xử lý dữ liệu (`data_utils.py`) và huấn luyện (`train.py`).
- `streamlit/`: Chứa mã nguồn cho giao diện web (`app.py`) và dữ liệu.
- `models/`: Chứa các mô hình đã huấn luyện (`last_model.pkl`, `scaler.joblib`).
- `docs/`: Tài liệu dự án.

## 🛠 Hướng dẫn cài đặt và sử dụng

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Huấn luyện mô hình**:
   ```bash
   python src/train.py
   ```
   *Lưu ý: Quá trình này có thể tốn thời gian do số lượng ảnh lớn.*

3. **Chạy ứng dụng Web**:
   ```bash
   streamlit run streamlit/app.py
   ```

## 📜 Tài liệu kỹ thuật
Xem chi tiết quy trình triển khai tại [docs/requirements.md](docs/requirements.md).
