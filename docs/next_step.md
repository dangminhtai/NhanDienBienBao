# Tiếp theo thực hiện (Next Steps)

1. **[REQ005] Xuất báo cáo Batch Mode:**
    - Cho phép anh tải file CSV chứa danh sách: `Tên file | Nhãn | Độ tin cậy | Tọa độ Box`.
    - Điều này rất quan trọng để anh làm báo cáo cuối kỳ.

2. **[OPT001] Tăng tốc Batch Mode:**
    - Hiện tại quét ~300 ảnh mất khoảng 40-50s. Nếu dùng đa luồng (Multi-threading) có thể giảm xuống còn 10-15s.

3. **[FEAT001] Tích hợp YOLO Lite:**
    - Nếu anh muốn độ chính xác 100% cho các biển báo trùng lặp hoặc sát nhau, em có thể chuẩn bị một bản hướng dẫn training YOLO nhẹ để thay thế hoàn toàn bộ quét HSV hiện tại.

4. **[UI001] Dark Mode Optimization:**
    - Tinh chỉnh CSS để giao diện trông "ngầu" hơn nữa khi anh bật Dark Mode của trình duyệt.
