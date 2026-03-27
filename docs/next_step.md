# Đề xuất cải tiến dự án (Next Steps)

Sau khi hoàn thiện phiên bản v2.0 sử dụng SVM và Hybrid Features, em đề xuất các hướng phát triển tiếp theo:

## 1. Nâng cấp Mô hình Deep Learning (CNN)
- Mặc dù SVM hoạt động tốt với đặc trưng Hybrid, nhưng **CNN (Convolutional Neural Networks)** như ResNet hoặc EfficientNet có thể tự học đặc trưng và đạt độ chính xác >99% trên GTSRB.
- Tích hợp TensorFlow/Keras hoặc PyTorch vào dự án.

## 2. Tối ưu hóa hiệu năng (Model Quantization)
- Nén mô hình SVM hoặc CNN để chạy mượt mà ngay cả trên các thiết bị tài nguyên thấp hoặc môi trường Edge Computing.

## 3. Hệ thống nhận diện video thời gian thực
- Mở rộng ứng dụng để nhận diện biển báo trực tiếp từ Webcam hoặc tệp Video sử dụng **OpenCV** và **YOLOv8** để localize (xác định vị trí) biển báo trước khi phân loại.

## 4. Báo cáo thống kê
- Thêm tính năng lưu lại lịch sử nhận diện và xuất báo cáo CSV/PDF cho người dùng.
