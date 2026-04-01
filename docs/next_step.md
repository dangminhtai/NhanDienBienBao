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

5. **[DOC001] Tích hợp tài liệu vào Dashboard:**
    - Thêm một tab "Kiến thức toán học" ngay trên ứng dụng Streamlit để anh có thể trình chiếu báo cáo (mục 4.2.1) trực tiếp cho mọi người xem mà không cần mở file .md.

6. **[DAT001] Tự động hóa Negative Mining:** 
    - Phát triển script tự động lưu lại những vùng dự đoán sai (False Positives) trong ứng dụng thực tế để đưa vào tập huấn luyện tiếp theo của SVM, giúp mô hình ngày càng thông minh hơn.

7. **[HOG001] Đa tỷ lệ (Pyramid) hoặc ROI Scaling:**
    - Hiện tại HOG trích xuất trên ROI 32x32. Thử nghiệm với HOG đa tỷ lệ hoặc Pyramid ảnh có thể hỗ trợ phát hiện tốt hơn các biển báo cực lớn (ở gần) và cực nhỏ (ở xa).

8. **[SVM001] Tối ưu SVM chuyên sâu (Progressive GridSearch):** 
    - Sau khi tìm được vùng tham số tốt, thực hiện tìm kiếm "mịn" hơn (ví dụ: $1 < C < 1.05$) để tinh chỉnh lề quyết định đạt độ chính xác tối ưu.

9. **[EVAL001] Precision-Recall Curve:** 
    - Vẽ biểu đồ đường cong Precision-Recall để anh có thể trực quan hóa khả năng hoạt động của mô hình ở các ngưỡng xác suất (Probability threshold) khác nhau trên Dashboard.
