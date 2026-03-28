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
| ISS011 | InconsistentVersionWarning: scikit-learn mismatch (1.5.1 vs 1.6.1) | Phân tích log terminal. | Nâng cấp scikit-learn lên 1.6.1 nếu gặp lỗi logic. |
| ISS012 | Hiện tượng "Đốm ma" (False Positives) dày đặc khi quét Deep Scan | - **Lọc hình học (Solidity):** Biển báo là vật thể đặc. Tuy nhiên, một số biển có họa tiết trắng lớn (biển Cấm Đi Ngược Chiều) có thể làm giảm Solidity. Ngưỡng 0.35 - 0.4 là điểm cân bằng tốt.
- **Bài học về sự cân bằng (v4.8):** Cài đặt bộ lọc quá chặt sẻ dẫn đến Under-detection. Ngưỡng Laplacian 40-50 và Saturation 30 là "điểm ngọt".
- **Gỡ bỏ seen_boxes (v5.0):** Việc chặn trùng lặp bằng tọa độ thô trong vòng lặp quét có thể làm mất các ứng viên tốt hơn xuất hiện ở các tổ hợp HSV sau. Để NMS xử lý toàn bộ là phương án an toàn nhất.
- **Lọc Mật độ màu (High Saturation Ratio):** Biển báo thật luôn có sự đan xen màu sắc (đỏ-trắng, xanh-trắng). Một vùng ảnh có >95% là màu bão hòa cao thường là nhiễu quang học (đốm sáng) chứ không phải biển báo.
" cực mạnh cho các vật thể tự nhiên (lá cây, đất).
- **Morphology Closing (9x9):** Giúp nối liền các mảng màu bị chia cắt bởi text hoặc họa tiết trắng bên trong biển báo. | Hoàn thành v4.6 |
| ISS013 | Biển to bị sót (False Negatives), biển nhỏ bị nhầm | Tăng Morphology (9x9), nới lỏng Laplacian/Saturation/SVM về mức cân bằng. | Hoàn thành v4.8 |
| ISS014 | Đốm ma chiếm chỗ biển thật (Case 3), Không thấy biển (Case 2) | Gỡ bỏ seen_boxes, mở rộng lưới quét 4x4, thêm lọc mật độ màu >95%. | Hoàn thành v5.0 |
