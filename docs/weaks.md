# Đánh giá điểm yếu của dự án (Weakness Assessment)

Dưới đây là các điểm yếu hiện tại của hệ thống nhận diện biển báo giao thông:

## 1. Cấu trúc thư mục (Project Structure)
- **Sự chồng chéo**: Có sự tồn tại của cả `src/` ở gốc và `streamlit/src/`. Điều này gây khó khăn cho việc quản lý mã nguồn và vi phạm nguyên lý DRY (Don't Repeat Yourself).
- **Phụ thuộc cứng**: Các script xử lý dữ liệu và huấn luyện chưa được module hóa hoàn toàn để có thể dùng chung giữa môi trường huấn luyện và triển khai web.

## 2. Mô hình & Thuật toán (Model & Algorithm)
- **Tính minh bạch (Transparency)**: Hiện tại app chỉ trả về kết quả dự đoán (Class ID), chưa có **độ tin cậy (Confidence score)**. Điều này khiến người dùng không biết mức độ chính xác của dự đoán đó.
- **Giới hạn của SVM**: SVM với đặc trưng HOG+HSV đạt kết quả tốt nhưng có thể gặp khó khăn với các biến thể về góc chụp, độ méo hoặc che khuất hơn là các mô hình Deep Learning (CNN).
- **Hardcoded Parameters**: Các thông số như `IMG_SIZE = (32, 32)` đang bị gắn cứng (hardcoded) trong nhiều file thay vì dùng một file config trung tâm.

## 3. Xử lý lỗi & Bảo mật (Error Handling & Security)
- **Kiểm soát đầu vào**: `extract_hybrid_features` chưa kiểm tra tính hợp lệ của ảnh đầu vào một cách chặt chẽ (ví dụ: ảnh bị lỗi, định dạng không hỗ trợ sâu).
- **Quản lý tài nguyên**: Việc tải mô hình (`joblib.load`) thực hiện mỗi khi app khởi động mà không có cơ chế cache hiệu quả của Streamlit (`st.cache_resource`) có thể gây tốn tài nguyên khi số lượng người dùng tăng.

## 4. Trải nghiệm người dùng (UX)
- **Thiếu tính năng xóa**: Người dùng không có nút "Clear" để thực hiện nhận diện ảnh mới một cách nhanh chóng mà không cần tải lại trang hoặc upload đè.
- **Phản hồi chậm**: Với các ảnh độ phân giải cao, việc xử lý HOG 4x4 trên CPU có thể gây trễ nhẹ nếu không được tối ưu hóa bằng vectorization hoặc xử lý song song.

## 5. Tài liệu & Quy trình (Documentation)
- **Thiếu Logging**: Hệ thống hoàn toàn không có file log để theo dõi các lỗi xảy ra trong quá trình chạy thực tế (Production).
