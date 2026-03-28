# Điểm yếu hiện tại của dự án (Current Weaknesses) - v4.0 Full

## Hiệu năng (Performance)
- **Tốc độ Detection**: Quy trình quét HOG trên nhiều vùng đề xuất (proposed boxes) vẫn còn chậm hơn so với các kiến trúc One-stage detector hiện đại như YOLO.
- **FPS**: Chưa đạt mức real-time trên CPU thông thường khi xử lý ảnh độ phân giải cao.

## Độ chính xác (Accuracy)
- **Cận thị HSV**: Sự phụ thuộc vào lọc màu HSV khiến mô hình dễ bị đánh lừa bởi các vùng có màu đỏ/xanh tương tự (biển quảng cáo, đèn giao thông) nếu SVM Binary không đủ mạnh.
- **Ánh sáng khắc nghiệt**: Trong điều kiện thiếu sáng trầm trọng hoặc lóa nắng, mặt nạ HSV có thể bị đứt gãy, dẫn đến việc bỏ sót biển báo.

## Kỹ thuật (Technical)
- **NMS**: Ngưỡng NMS cố định có thể loại bỏ các biển báo nằm rất gần nhau.
- **Dependency**: Yêu cầu cài đặt bộ thư viện nặng (TensorFlow, OpenCV, Scikit-learn) làm tăng kích thước bundle triển khai.
