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

10. **[PROPOSAL001] Hình thái học Thích nghi (Adaptive Morphology):** 
    - Hiện tại Kernel Morph (9x9) là cố định. Cần phát triển logic tự động tính toán kích thước Kernel dựa trên Res (độ phân giải) của ảnh và khoảng cách dự kiến để vá lỗ hổng tốt hơn cho biển báo ở xa.

11. **[COLOR001] Kiểm nghiệm Normalized RGB cho vùng có bóng đổ (Shadow Detection):** 
    - Hiện dự án dùng HSV là chính. Có thể tích hợp thêm lớp kiểm tra Normalized RGB để tăng độ chính xác tại các vùng biển báo bị bóng đổ đè lên, nơi Hue vẫn giữ nguyên nhưng độ rực rỡ (S) bị thay đổi mạnh.

12. **[MSER001] Tích hợp bộ dò MSER cho phát hiện xa (High Resolution):** 
    - Với các biển báo cực nhỏ ở xa, bộ lọc HSV có thể bỏ sót. Thử nghiệm MSER để phát hiện các "vùng cực trị" ổn định có thể tăng recall đáng kể cho ảnh độ phân giải cao.

13. **[UI001] Giao diện Tương tác nâng cao (UI/UX):** 
    - Phát triển nút gạt (Toggle) chế độ Sáng/Tối trực quan hơn và tích hợp thêm các báo cáo PDF tự động sau khi Batch Processing kết thúc để người dùng dễ dàng lưu trữ kết quả.

14. **[UI002] Tối ưu hóa không gian hiển thị (Space Optimization):** 
    - Áp dụng cơ chế `st.expander` hoặc `st.popover` cho các biểu đồ kỹ thuật số phụ trợ để giao diện chính tập trung hoàn toàn vào kết quả hình ảnh, giúp người dùng không có chuyên môn vẫn dễ dàng tiếp cận.

15. **[UI003] Cơ chế xử lý lỗi đồ họa (Robust Rendering):** 
    - Xây dựng lớp bọc (wrapper) cho các thành phần HTML/JS tùy chỉnh để bắt lỗi cú pháp (Syntax error) và hiển thị thông báo thay thế (fallback) thay vì làm treo hoặc hiện thông báo lỗi của trình duyệt lên Dashboard.

16. **[AI001] Chế độ Chồng lớp Kích hoạt (Activation Overlays):** 
    - Tích hợp tính năng chồng lớp Feature Maps trực tiếp lên ảnh gốc (như mô hình Grad-CAM) để người dùng thấy rõ AI đang "chú ý" vào vùng pixel nào trên biển báo thực tế.

17. **[AI002] Trực quan hóa 3D (3D Tensor Projection):** 
    - Thử nghiệm trình diễn các khối dữ liệu Conv (ví dụ: 14x14x64) dưới dạng khối 3D tương tác, giúp người xem hình dung rõ ràng về khái niệm "Volume" trong Deep Learning.

18. **[AI003] Phân tích cụm Đặc trưng (Feature Cluster Analysis):** 
    - Sử dụng thuật toán t-SNE hoặc PCA để trực quan hóa vector 256 chiều lên không gian 2D/3D, giúp người dùng thấy được cách AI "gom nhóm" các biển báo cùng loại lại gần nhau trong không gian đặc trưng.

19. **[AI004] Thử nghiệm Kích hoạt Nơ-ron (Neuron Probing):** 
    - Cho phép người dùng "vẽ" hoặc sửa đổi trực tiếp các điểm trên bản đồ 5x5 để xem các Gene (256 biến) thay đổi thế nào trong thời gian thực, giúp hiểu rõ mối quan hệ nhân quả (Causality) trong mạng CNN.

20. **[UI004] Trực quan hóa Phép chuẩn hóa (Standardization Viz):** 
    - Bổ sung một mô hình đồ họa (ví dụ: Biểu đồ phân phối Gauss) để người dùng thấy rõ dữ liệu bị co giãn thế nào khi đi từ CNN qua StandardScaler trước khi vào SVM.

21. **[AI005] Trực quan hóa Siêu mặt phẳng SVM (SVM Decision Boundary):** 
    - Trình diễn cách SVM sử dụng bộ mã 256 chiều đã chuẩn hóa để vạch ra các "Siêu mặt phẳng" phân tách các loại biển báo, giúp người dùng hiểu tại sao một biển báo lại được gán cho một nhãn cụ thể.

22. **[AI006] Chiếu Tọa độ Thực (Real-time t-SNE Projection):** 
    - Xây dựng một module chạy ngầm để tính toán tọa độ t-SNE/UMAP thực tế cho vector 256 chiều, giúp người dùng thấy vị trí chính xác của ảnh mình vừa tải lên trong mối tương quan với toàn bộ 34.000 điểm dữ liệu huấn luyện.

23. **[AI007] Hiệu chuẩn Thống kê (Platt Scaling):** 
    - Nghiên cứu tích hợp phương pháp Platt Scaling để chuyển đổi điểm số `decision_function` của SVM thành xác suất thực tế một cách chuẩn xác hơn so với hàm Softmax đơn giản, tăng độ tin cậy cho biểu đồ phân tích Step 4.

24. **[UI006] So sánh Kernel Tương tác (Interactive Kernel Lab):** 
    - Cho phép người dùng tạm thời thay đổi Kernel (ví dụ từ Linear sang RBF) để xem ranh giới quyết định thay đổi thế nào trên cùng một tập dữ liệu, giúp hiểu sâu về sức mạnh của từng loại Kernel.

25. **[UI007] Chế độ Phân loại Đối tượng (Audience Switch):** 
    - Xây dựng một công tắc cho phép chuyển đổi giao diện giữa 'Người học' (Dùng ngôn ngữ sư phạm) và 'Lập trình viên' (Dùng ngôn ngữ kỹ thuật, show mã nguồn, file, dòng code) để tối ưu hóa trải nghiệm cho nhiều đối tượng khác nhau.

26. **[UI008] Tùy chỉnh Siêu tham số (Interactive Hyperparameter Tuning):** 
    - Cho phép người dùng trượt (Slider) để thay đổi giá trị $C$ hoặc $\gamma$ trên UI và xem ranh giới Kernel trong Step 4.2 co giãn theo thời gian thực, giúp hiểu rõ hiện tượng Bias/Variance.

27. **[UI009] Hoạt họa Bẻ cong Không gian (Space Bending Animation):** 
    - Phát triển một đoạn hoạt họa (GIF/Lottie) minh họa sống động quá trình các điểm dữ liệu từ 1D được 'nhấc bổng' lên 2D để SVM kẻ ranh giới, giúp tăng tính tương tác và dễ hiểu cho Bước 4.2.

28. **[UI010] Cán cân Logic Tương tác (Interactive Logic Scale):** 
    - Cho phép người dùng rê chuột vào từng thanh trên cán cân ở Bước 4.3 để xem vùng pixel tương ứng trong ảnh gốc đã đóng góp vào quyết định đó như thế nào (kết hợp với Integrated Gradients), giúp tạo ra sự kết nối trực quan tối đa.

22. **[UI005] Khám phá Phân bổ (Distribution Explorer):** 
    - Cho phép người chọn 1 Gene và xem biểu đồ phân bổ (Histogram) của 34.000 điểm dữ liệu mẫu của Gene đó, giúp thấy rõ tại sao Mean và Std lại có giá trị như vậy.
