# Yêu cầu dự án (Requirements)

| ID | Yêu cầu | Chi tiết | Trạng thái |
|---|---|---|---|
| REQ001 | Nhận diện biển báo | Sử dụng tập dữ liệu GTSRB (43 lớp). | Hoàn thành |
| REQ002 | Mô hình SVM | Sử dụng SVM với đặc trưng lai ghép (GTSRBv2 logic). | Hoàn thành |
| REQ003 | Hybrid Features | HOG (4x4) + HSV Histogram = 1812 đặc trưng. | Hoàn thành |
| REQ004 | Tiền xử lý | Áp dụng CLAHE cho ảnh đầu vào. | Hoàn thành |
| REQ005 | Giao diện Web | Triển khai bằng Streamlit với giao diện Premium. | Hoàn thành |
| REQ006 | Hiệu suất | Response time < 2s cho mỗi lần dự đoán. | Hoàn thành |
| REQ007 | Github Policy | Không tự động đẩy code lên Github trừ khi có yêu cầu. | Hoàn thành |
| REQ008 | Minh bạch toán học | Giải thích quy trình dự đoán từ cơ sở toán học đến số liệu thực tế. | Hoàn thành |
| REQ009 | Sơ đồ Tổng quan | Hiển thị Bản đồ Dòng chảy (map2.md) trực tiếp trên giao diện UI. | Hoàn thành |

## Quy trình triển khai Step-by-step
1. **Thiết lập môi trường**: Cài đặt dependencies từ `requirements.txt`.
2. **Module trung tâm**: `src/data_utils.py` xử lý logic trích xuất đặc trưng 1812 chiều.
3. **Scaler đồng bộ**: Chạy `src/rebuild_scaler.py` để tạo bộ chuẩn hóa tương thích mô hình.
4. **App Deployment**: Khởi chạy `streamlit run streamlit/app.py`.
