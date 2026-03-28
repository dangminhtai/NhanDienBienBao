### Đề xuất cải thiện (Next Steps) - Phiên bản v4.1+

Sau khi triển khai thành công mô hình Lai ghép (Hybrid) v4.0, các bước tiếp theo để nâng cấp hệ thống bao gồm:

1.  **Chuyển đổi sang WebCam (Real-time Prediction)**:
    -   Tích hợp tính năng nhận diện biển báo trực tiếp từ Camera bằng OpenCV.
    -   Sử dụng thư viện `streamlit-webrtc` để hỗ trợ trình duyệt.
2.  **Tối ưu hóa Mô hình (Model Quantization)**:
    -   Chuyển đổi mô hình CNN sang định dạng `.tflite` (TensorFlow Lite) để giảm dung lượng và tăng tốc độ trích xuất đặc trưng trên CPU.
3.  **Hệ thống Lọc nhiễu (Pre-filtering)**:
    -   Thêm một lớp phân loại nhị phân (Binary Classifier) trước khi đưa vào SVM để xác định xem vùng ảnh có thực sự là "Biển báo" hay không.
4.  **Logging & Monitoring**:
    -   Sử dụng SQLite hoặc tệp log để lưu lại các lịch sử dự đoán (kèm ảnh) để phân tích các trường hợp nhầm lẫn (Misclassifications) sau này.
5.  **Ứng dụng thực tế**:
    -   Xây dựng thêm tính năng đọc tên biển báo bằng giọng nói (Text-to-Speech) để hỗ trợ lái xe rảnh tay.
