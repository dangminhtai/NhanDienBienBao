# Danh sách các vấn đề và giải pháp (Issue List)

| ID | Nguyên nhân | Hành động đã làm | Giải pháp |
|---|---|---|---|
| ISS001 | ValueError: X has 324 features, but SVC is expecting 1812 features | Phân tích `GTSRBv2.ipynb` để tìm logic hybrid features. | Đồng bộ hóa `data_utils.py` với HOG 4x4 + HSV Histogram 48 bins. |
| ISS002 | Thiếu bộ chuẩn hóa `scaler.joblib` tương thích với 1812 chiều | Viết script `src/rebuild_scaler.py` để tạo lại scaler từ dữ liệu cục bộ. | Tạo thành công `models/scaler_1812.joblib`. |
| ISS003 | Giao diện Streamlit cơ bản, chưa chuyên nghiệp | Thiết kế lại UI với Custom CSS, Premium Theme. | Nâng cấp `streamlit/app.py` v2.0. |
| ISS004 | Lỗi đường dẫn sau khi di chuyển tệp mô hình và dataset | Cập nhật `app.py` và `rebuild_scaler.py` bám sát cấu trúc mới. | Di chuyển logic load sang `streamlit/models/` và `dataset/`. |
| ISS005 | Cảnh báo Streamlit về nhãn trống và tham số ảnh cũ | Cập nhật tham số `width` và thêm label ẩn cho file uploader. | Khắc phục xong các cảnh báo trong app.py v2.2. |
