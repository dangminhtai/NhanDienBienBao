# Các bước tiếp theo (Next Steps) - Lộ trình v4.1+

## Cải tiến Thuật toán (Algorithm Improvements)
- **YOLO Integration**: Chuyển đổi từ SVM-HOG scanning sang YOLOv8/v11 để đạt tốc độ real-time và độ chính xác cao hơn trong tác vụ Detection.
- **Data Augmentation**: Bổ sung các phép biến đổi về ánh sáng và nhiễu (GTSDB thường có ảnh chất lượng thấp hơn GTSRB) để detector hoạt động ổn định hơn.
- **Ensemble Model**: Kết hợp nhiều kiến trúc CNN (MobileNet, ResNet) làm feature extractor để tăng cường độ tin cậy.
- [ ] Thêm tính năng "Vẽ Heatmap" cho các vùng màu đang được quét để trực quan hóa Deep Scan.
- [ ] Huấn luyện lại SVM Binary với nhiều ảnh "Hard Negatives" (lá cây, bóng đổ) để tăng độ nhạy tự nhiên.
- [ ] Tích hợp YOLOv11 để thay thế hoàn toàn bộ quét HSV cổ điển (Long term).

## Cải tiến Ứng dụng (App Improvements)
- **WebCam Real-time**: Sử dụng `streamlit-webrtc` để hỗ trợ nhận diện trực tiếp qua camera thay vì chỉ tải ảnh lên.
- **History Tracking**: Lưu trữ lịch sử nhận diện và xuất báo cáo CSV/PDF.
- **Multilingual UI**: Hỗ trợ đa ngôn ngữ hoàn chỉnh (Vietnamese, English, German).

## Triển khai (Deployment)
- **Dockerization**: Đóng gói ứng dụng vào Docker container để dễ dàng triển khai lên Cloud (AWS, Azure, Heroku).
- **Optimization**: Chuyển đổi mô hình sang định dạng ONNX hoặc TensorFlow Lite để chạy nhanh hơn trên các thiết bị tài nguyên yếu.
