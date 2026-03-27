# Bài học kinh nghiệm trong dự án (Learned Lessons)

## Kỹ thuật Feature Engineering (Hybrid)
- Trích xuất đặc trưng lai ghép (Hybrid) giữa hình học (HOG) và màu sắc (HSV) mang lại độ chi tiết cao hơn cho mô hình ML truyền thống như SVM.
- **HOG (Fine-grained)**: Sử dụng `pixels_per_cell=(4, 4)` thay vì 8x8 mặc định tăng gấp 4 lần số lượng đặc trưng (từ 324 lên 1764), giúp mô tả hình dạng biển báo tốt hơn.
- **HSV Histogram**: Thêm 48 chiều đặc trưng màu sắc giúp phân biệt các biển báo có hình dạng giống nhau nhưng màu khác nhau (ví dụ: biển báo cấm đỏ vs biển báo chỉ dẫn xanh).

## Tiền xử lý (Preprocessing)
- **CLAHE**: Cân bằng ánh sáng cục bộ cực kỳ quan trọng đối với tập dữ liệu GTSRB vì có nhiều ảnh bị tối hoặc lóa sáng.

## Triển khai (Deployment)
- Luôn kiểm tra tính tương thích giữa `StandardScaler` và `Model` (số lượng đặc trưng `n_features_in_`).
- Khi tích hợp mô hình từ bên ngoài, cần bám sát Notebook gốc để tái hiện đúng pipeline trích xuất đặc trưng.

- **Giải mã Hộp đen (Model Transparency)**: Việc kết hợp `st.latex` và `st.pyplot` giúp giao diện AI trở nên chuyên nghiệp và minh bạch hơn, giải quyết triệt để yêu cầu về "Toán học cho AI".
- **Tối ưu hóa (Optimization)**: Sử dụng `@st.cache_resource` giúp giảm thời gian nạp mô hình SVM (1812 đặc trưng) xuống mức tức thì sau lần nạp đầu tiên.
- **SVM Confidence**: Dù SVC không bật `probability=True`, ta vẫn có thể ước lượng độ tin cậy thông qua `decision_function` và hàm Sigmoid để cải thiện trải nghiệm người dùng.
