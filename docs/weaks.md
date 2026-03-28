# Đánh giá điểm yếu của dự án (Weakness Assessment)

Dưới đây là các điểm yếu hiện tại của hệ thống nhận diện biển báo giao thông:

## 1. Cấu trúc thư mục (Project Structure)
- **Sự chồng chéo**: Có sự tồn tại của cả `src/` ở gốc và `streamlit/src/`. Điều này gây khó khăn cho việc quản lý mã nguồn và vi phạm nguyên lý DRY (Don't Repeat Yourself).
- **Phụ thuộc cứng**: Các script xử lý dữ liệu và huấn luyện chưa được module hóa hoàn toàn để có thể dùng chung giữa môi trường huấn luyện và triển khai web.

## 2. Mô hình & Thuật toán (Model & Algorithm)
- **Tính minh bạch (Transparency) [ĐÃ GIẢI QUYẾT v4.0]**: Đã thêm phần "Cơ sở Toán học & Minh bạch dữ liệu" với HOG/HSV visualization và LaTeX math.
- **Giới hạn của SVM**: SVM với đặc trưng HOG+HSV đạt kết quả tốt nhưng có thể gặp khó khăn với các biến thể về góc chụp, độ méo hoặc che khuất hơn là các mô hình Deep Learning (CNN).
- **Hardcoded Parameters**: Các thông số như `IMG_SIZE = (32, 32)` đang bị gắn cứng (hardcoded) trong nhiều file thay vì dùng một file config trung tâm.

## 3. Xử lý lỗi & Bảo mật (Error Handling & Security)
- **Kiểm soát đầu vào**: `extract_hybrid_features` chưa kiểm tra tính hợp lệ của ảnh đầu vào một cách chặt chẽ (ví dụ: ảnh bị lỗi, định dạng không hỗ trợ sâu).
- **Quản lý tài nguyên [ĐÃ CẢI THIỆN v4.0]**: Đã áp dụng `@st.cache_resource` để nạp mô hình SVG 1812 chiều hiệu quả, tránh nạp lại nhiều lần.

## 4. Trải nghiệm người dùng (UX)
- **Thiếu tính năng xóa**: Người dùng không có nút "Clear" để thực hiện nhận diện ảnh mới một cách nhanh chóng mà không cần tải lại trang hoặc upload đè.
- **Phản hồi chậm**: Với các ảnh độ phân giải cao, việc xử lý HOG 4x4 trên CPU có thể gây trễ nhẹ nếu không được tối ưu hóa bằng vectorization hoặc xử lý song song.

### Điểm yếu v4.0 (Hybrid SVM+CNN)
- **Tốc độ Inference (Dự đoán)**: Do phải chạy cả mô hình CNN (để trích xuất đặc trưng) và SVM, thời gian phản hồi có thể chậm hơn so với HOG đơn thuần.
- **Dung lượng**: Cần lưu trữ cả tệp `.h5` của CNN, làm tăng tổng dung lượng thư mục `models/`.
- **Phụ thuộc Thư viện**: Cần cài đặt `tensorflow` (khá nặng) thay vì chỉ dùng `scikit-image` và `scikit-learn`.
- **Nhạy cảm với Tiền xử lý**: Nếu ảnh đầu vào không được resize chính xác 32x32 và normalize 0-1, kết quả trích xuất đặc trưng sẽ sai lệch hoàn toàn.

## 5. Tài liệu & Quy trình (Documentation)
- **Thiếu Logging**: Hệ thống hoàn toàn không có file log để theo dõi các lỗi xảy ra trong quá trình chạy thực tế (Production).
