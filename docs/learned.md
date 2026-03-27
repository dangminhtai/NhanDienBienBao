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

## Phân tích hệ thống (System Analysis)
- **Tính minh bạch**: Một hệ thống AI tốt không chỉ đưa ra kết quả mà cần đi kèm với độ tin cậy (Confidence score) để người dùng có thể đánh giá mức độ chính xác.
- **Tối ưu hóa tài nguyên**: Sử dụng cache (`st.cache_resource`, `st.cache_data`) là tối quan trọng trong các ứng dụng Web AI để tránh lãng phí tài nguyên tính toán.
- **Cấu trúc SOLID**: Việc phân tách rõ ràng giữa logic giao diện (Streamlit) và logic xử lý (Service/Handler) giúp code dễ bảo trì hơn, nhưng cần tránh việc nhân bản code (redundancy).
