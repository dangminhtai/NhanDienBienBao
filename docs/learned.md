# Bài học kinh nghiệm trong dự án (Learned Lessons)

## Kỹ thuật Feature### Học được từ v4.0 (Hybrid SVM+CNN)
- **Kiến trúc Lai ghép**: Cách kết hợp CNN làm bộ trích xuất đặc trưng (Feature Extractor) và SVM làm bộ phân loại (Classifier) mang lại độ chính xác cao hơn và khả năng bao quát tốt hơn HOG.
- **Tiền xử lý CNN**: Quy trình chuẩn hóa ảnh đơn giản (Resize 32x32, normalize 0-1) nhưng cực kỳ quan trọng để khớp với dữ liệu huấn luyện.
- **Tích hợp Keras trong Streamlit**: Cần sử dụng `@st.cache_resource` để tránh việc tải lại mô hình CNN nặng nề mỗi khi người dùng tương tác.
- **Minh bạch toán học**: Việc giải thích cơ chế "Deep Features" giúp người dùng tin tưởng hơn vào kết quả của mô hình "Hộp đen".
ình dạng giống nhau nhưng màu khác nhau (ví dụ: biển báo cấm đỏ vs biển báo chỉ dẫn xanh).

## Tiền xử lý (Preprocessing)
- **CLAHE**: Cân bằng ánh sáng cục bộ cực kỳ quan trọng đối với tập dữ liệu GTSRB vì có nhiều ảnh bị tối hoặc lóa sáng.

## Triển khai (Deployment)
- Luôn kiểm tra tính tương thích giữa `StandardScaler` và `Model` (số lượng đặc trưng `n_features_in_`).
- Khi tích hợp mô hình từ bên ngoài, cần bám sát Notebook gốc để tái hiện đúng pipeline trích xuất đặc trưng.

- **Giải mã Hộp đen (Model Transparency)**: Việc kết hợp `st.latex` và `st.pyplot` giúp giao diện AI trở nên chuyên nghiệp và minh bạch hơn, giải quyết triệt để yêu cầu về "Toán học cho AI".
- **Tối ưu hóa (Optimization)**: Sử dụng `@st.cache_resource` giúp giảm thời gian nạp mô hình SVM (1812 đặc trưng) xuống mức tức thì sau lần nạp đầu tiên.
- **SVM Confidence**: Dù SVC không bật `probability=True`, ta vẫn có thể ước lượng độ tin cậy thông qua `decision_function` và hàm Sigmoid để cải thiện trải nghiệm người dùng.

### Học được từ v4.0-Detect (Full-Image Pipeline)
- **Kiến trúc Lai ghép Cấp độ 2**: Kết hợp giữa Scanning (HOG+SVM nhị phân) để tìm vùng ứng viên và Recognition (CNN+SVM đa lớp) để định danh. Đây là cách tiếp cận hiệu quả hơn so với việc quét CNN trên toàn bô ảnh.
- **Tham số HOG**: Cần khớp chính xác số lượng đặc trưng (ví dụ: 324) với kích thước ROI (32x32) để đảm bảo scaler hoạt động đúng.
- **NMS (Non-Maximum Suppression)**: Là thành phần bắt buộc trong bất kỳ pipeline Detection nào để dọn dẹp các hộp phát hiện dư thừa.
- **Streamlit Wide Layout**: Phù hợp cho việc hiển thị các kết quả phân tích phức tạp và nhiều ảnh cùng lúc.
