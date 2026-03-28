# Danh sách các vấn đề và giải pháp (Issue List)

| ID | Nguyên nhân | Hành động đã làm | Giải pháp |
|---|---|---|---|
| ISS001 | ValueError: X has 324 features, but SVC is expecting 1812 features | Phân tích `GTSRBv2.ipynb` để tìm logic hybrid features. | Đồng bộ hóa `data_utils.py` với HOG 4x4 + HSV Histogram 48 bins. |
| ISS002 | Thiếu bộ chuẩn hóa `scaler.joblib` tương thích với 1812 chiều | Viết script `src/rebuild_scaler.py` để tạo lại scaler từ dữ liệu cục bộ. | Tạo thành công `models/scaler_1812.joblib`. |
| ISS003 | Giao diện Streamlit cơ bản, chưa chuyên nghiệp | Thiết kế lại UI với Custom CSS, Premium Theme. | Nâng cấp `streamlit/app.py` v2.0. |
| ISS003.1 | AttributeError: module 'streamlit' has no attribute 'mermaid' | Thay thế `st.mermaid()` bằng `st.markdown()` kết hợp với khối mã ` ```mermaid `. | Sử dụng `st.markdown` để vẽ biểu đồ luồng Mermaid trong Streamlit. |
| ISS004 | Lỗi đường dẫn sau khi di chuyển tệp mô hình và dataset | Cập nhật `app.py` và `rebuild_scaler.py` bám sát cấu trúc mới. | Di chuyển logic load sang `streamlit/models/` và `dataset/`. |
| ISS005 | Cảnh báo Streamlit về nhãn trống và tham số ảnh cũ | Cập nhật tham số `width` và thêm label ẩn cho file uploader. | Khắc phục xong các cảnh báo trong app.py v2.2. |
| ISS006 | Chưa có cơ chế cache mô hình khiến app tải lại mỗi lần reload | Sử dụng `st.cache_resource` trong Streamlit. | Hoàn thành v4.0 |
| ISS007 | Thiếu độ tin cậy (Confidence score) trong kết quả | Tính toán từ `predict_proba` hoặc `decision_function`. | Hoàn thành v4.0 |
| ISS008 | Cấu trúc thư mục chồng chéo (`src/` vs `streamlit/src/`) | Triển khai logic nạp model/scaler linh hoạt. | Đang cải thiện |
| ISS009 | RuntimeError: Data is outside [0.0, 1.0] in HOG visualization | Thêm tham số `clamp=True` vào lệnh `st.image`. | Hoàn thành v4.1 |
| ISS010 | Mô hình bị Overfitting trên tập Train/Test (Đạt 91%+ nhưng sai trên ảnh lạ) | Do $C=100.0$ quá cao và thiếu Data Augmentation. | Đã chẩn đoán (Xem docs) |
