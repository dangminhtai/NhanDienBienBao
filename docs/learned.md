# Bài học kinh nghiệm trong dự án (Learned Lessons)

## Học được từ v4.0 (Hybrid SVM+CNN)
- **Kiến trúc Lai ghép**: Cách kết hợp CNN làm bộ trích xuất đặc trưng (Feature Extractor) và SVM làm bộ phân loại (Classifier) mang lại độ chính xác cao hơn và khả năng bao quát tốt hơn HOG.
- **Tiền xử lý CNN**: Quy trình chuẩn hóa ảnh đơn giản (Resize 32x32, normalize 0-1) nhưng cực kỳ quan trọng để khớp với dữ liệu huấn luyện.
- **Tầm quan trọng của NMS**: IoU threshold đóng vai trò cực kỳ quan trọng trong việc giữ lại các ô chuẩn xác nhất.
- **Trực quan hóa "Nội tạng" thuật toán**: Khi trình bày cho người dùng không rành code, không được nhảy vọt từ Mặt nạ trắng đen sang Ảnh màu khoanh khung ngay lập tức. Phải có bước đệm: Vẽ khung trực tiếp lên mặt nạ đen trắng để chứng minh bản chất của thuật toán `findContours` là làm việc trên các "hòn đảo" pixel, sau đó mới dùng tọa độ đó để mapping ngược lại ảnh gốc. Điều này giúp xóa tan "Hộp đen" hoàn toàn! 🕵️‍♂️🔥
- **Tích hợp Keras trong Streamlit**: Cần sử dụng `@st.cache_resource` để tránh việc tải lại mô hình CNN nặng nề mỗi khi người dùng tương tác.
- **Bảo toàn độ phân giải (v5.2):** Loại bỏ logic downscaling (800px) giúp hệ thống "nhìn thấy" các biển báo nhỏ (20-30px) ở xa cực kỳ hiệu quả. Tuy nhiên đánh đổi bằng thời gian xử lý ảnh 4K tăng lên (~1-2s).
- **Unicode OpenCV (v6.0):** `cv2.putText` không hỗ trợ Unicode. Luôn luôn dùng PIL để vẽ chữ tiếng Việt trước khi hiển thị trên Streamlit.
- **Trực quan hóa (v6.1):** Việc thêm ảnh "Meta" giúp kiểm chứng độ chính xác của mô hình ngay lập tức bằng mắt thường, tăng tính minh bạch toán học.
- **Morphology Closing (9x9):** Giúp nối liền các mảng màu bị chia cắt bởi text hoặc họa tiết trắng bên trong biển báo.
- **Bài học về sự cân bằng (v4.8):** Cài đặt bộ lọc quá chặt (Laplacian 100, SVM 0.5) sẽ dẫn đến Under-detection. Ngưỡng Laplacian 40-50 và Saturation 30 là "điểm ngọt" để vừa giữ biển vừa diệt nhiễu.
- **Minh bạch toán học**: Việc giải thích cơ chế "Deep Features" giúp người dùng tin tưởng hơn vào kết quả của mô hình "Hộp đen".

## Tiền xử lý (Preprocessing)
- **CLAHE**: Cân bằng ánh sáng cục bộ cực kỳ quan trọng đối với tập dữ liệu GTSRB vì có nhiều ảnh bị tối hoặc lóa sáng.
- **Lọc hình học (Solidity):** Biển báo là vật thể đặc. Việc kiểm tra tỷ lệ giữa diện tích contour và diện tích bounding box (Solidity > 0.45) là cách cực kỳ hiệu quả để loại bỏ các đốm nhiễu lốm đốm.
- **Độ nét (Laplacian Variance):** Biển báo kim loại thường sắc nét. Các vùng mờ bokeh hoặc lá cây mờ có phương sai Laplacian thấp. Ngưỡng 60-80 giúp loại bỏ "đốm ma" rất tốt.
- **Chuẩn hóa SVM Score:** SVM Linear trả về khoảng cách không giới hạn. Dùng hàm Sigmoid (Platt scaling đơn giản) giúp chuyển về dải 0-100% thân thiện với người dùng.

## Triển khai (Deployment)
- Luôn kiểm tra tính tương thích giữa `StandardScaler` và `Model` (số lượng đặc trưng `n_features_in_`).
- Khi tích hợp mô hình từ bên ngoài, cần bám sát Notebook gốc để tái hiện đúng pipeline trích xuất đặc trưng.
- **Giải mã Hộp đen (Model Transparency)**: Việc kết hợp `st.latex` và `st.pyplot` giúp giao diện AI trở nên chuyên nghiệp và minh bạch hơn, giải quyết triệt để yêu cầu về "Toán học cho AI".
- **Tối ưu hóa (Optimization)**: Sử dụng `@st.cache_resource` giúp giảm thời gian nạp mô hình SVM (1812 đặc trưng) xuống mức tức thì sau lần nạp đầu tiên.
- **SVM Confidence**: Dù SVC không bật `probability=True`, ta vẫn có thể ước lượng độ tin cậy thông qua `decision_function` và hàm Sigmoid để cải thiện trải nghiệm người dùng.

### Học được từ v4.0-Detect (Full-Image Pipeline)
- **Kiến trúc Lai ghép Cấp độ 2**: Kết hợp giữa Scanning (HOG+SVM nhị phân) để tìm vùng ứng viên và Recognition (CNN+SVM đa lớp) để định danh. Đây là cách tiếp cận hiệu quả hơn so với việc quét CNN trên toàn bô ảnh.
- **Tham số HOG**: Cần khớp chính xác số lượng đặc trưng (ví dụ: 324) với kích thước ROI (32x32) để đảm bảo scaler hoạt động đúng.
- **NMS (Non-Maximum Suppression)**: Là thành phần bắt buộc trong bất kỳ pipeline Detection nào để dọn dẹp các hộp phát hiện dư thừa.
- **Streamlit Wide Layout**: Phù hợp cho việc hiển thị các kết quả phân tích phức tạp và nhiều ảnh cùng lúc.

## Tài liệu hóa Kỹ thuật (Technical Documentation)
- **Truy vết Toán học (Mathematical Tracking)**: Việc tài liệu hóa trực tiếp từ mã triển khai thực tế (như notebook) giúp báo cáo có chiều sâu và tính thuyết phục cao.
- **Biểu diễn Công thức**: Sử dụng LaTeX để biểu diễn các khái niệm như Gradient, RBF Kernel, hay IoU giúp chuẩn hóa ngôn ngữ toán học trong báo cáo AI chuyên nghiệp.
- **Kết nối Lý thuyết và Thực thi**: Luôn liên kết các hằng số cụ thể trong code (như ngưỡng lọc Hue 0-10) với ý nghĩa vật lý/toán học của chúng (tính tuần hoàn của không gian màu).

## Khai thác Dữ liệu (Data Mining)
- **Hard Negative Mining (HNM)**: Việc chọn mẫu âm (Negative) có chiến thuật (dựa trên IoU < 0.1) quan trọng hơn việc lấy mẫu âm ngẫu nhiên tràn lan. Điều này giúp SVM có ranh giới quyết định (Decision Boundary) sắc nét hơn.
- **Tỷ lệ Mẫu (Sampling Ratio)**: Duy trì tỷ lệ Negative gấp 2 lần Positive là một kinh nghiệm tốt cho bài toán Detection để giảm thiểu False Positives ngay từ tầng lọc đầu tiên.

## Đặc trưng HOG (HOG Features)
- **Chuẩn hóa $L_2$-Hys**: Cực kỳ hiệu quả cho tập dữ liệu có độ tương phản thay đổi mạnh như GTSDB. Hạn chế giá trị 0.2 giúp tránh "Saturation" của Gradient.
- **Tính toán số chiều**: Việc khớp tham số `pixels_per_cell` và `cells_per_block` với kích thước ảnh ROI ($32 \times 32 \rightarrow 324$ features) là bước cơ bản để đảm bảo SVM nhận đúng đầu vào.
- **Chiến thuật Grayscale**: Dù lọc màu quan trọng ban đầu, nhưng HOG chỉ cần Gradient cường độ để xác định hình dáng (tròn, tam giác, hình chữ nhật).

## Phương pháp Sư phạm trong Tài liệu (Educational Documentation)
- **Giải mã Hộp đen (Breaking the Black Box)**: Việc cung cấp các ví dụ số học thực tế (numerical examples) giúp người đọc nắm bắt bản chất thuật toán nhanh hơn rất nhiều so với chỉ đưa ra công thức trừu tượng.
- **Minh họa từng bước**: Chia nhỏ quy trình (Gradient -> Magnitude -> Binning -> Normalization) kèm ví dụ cho từng bước giúp xây dựng niềm tin vào tính chính xác của hệ thống AI.

## Tài liệu chuyên nghiệp (Standard Documentation)
- **Tính Phổ quát (Universality)**: Tài liệu kỹ thuật tốt không nên quá phụ thuộc vào tên file nội bộ mà cần diễn đạt quy trình dưới dạng kiến thức chuẩn để người dùng khác có thể tham khảo cho các dự án tương tự.
- **Tránh "Hộp đen" trong Siêu tham số**: Giải thích ý nghĩa vật lý/vận hành của các tham số trừu tượng (như C, Gamma) giúp người học hiểu được *tại sao* mô hình hoạt động hiệu quả thay vì chỉ biết *cách* chạy code.

## Đánh giá Mô hình (Model Evaluation)
- **Cân bằng Precision-Recall**: Trong bài toán phát hiện biển báo, Recall (độ phủ) thường quan trọng hơn một chút so với Precision để tránh bỏ sót các biển báo nguy hiểm. Tuy nhiên, Precision thấp sẽ gây ra quá nhiều "báo động giả" trên Dashboard.
- **Vai trò của Confusion Matrix**: Đây là công cụ tốt nhất để phân tích "điểm yếu" của mô hình, từ đó tìm ra các mẫu Hard Negatives cần thu thập thêm.
- **F1-Score**: Dùng làm chỉ số vàng (Golden metric) khi phải chọn lựa giữa các phiên đầu huấn luyện khác nhau.

## Siêu tham số (Hyperparameters)
- **Tầm quan trọng của C**: Một hằng số nhỏ ($C=1$) giúp mô hình linh hoạt hơn với nhiễu ảnh thực tế của GTSDB so với việc ép mô hình học thuộc lòng bằng $C$ quá lớn.
- **RBF và Gamma**: Kernel rbf là lựa chọn tối ưu cho dữ liệu phi tuyến tính như HOG. Việc chọn đúng $\gamma$ giúp ranh giới quyết định (Decision Boundary) bao quát hơn, chống Overfitting hiệu quả.
- **GridSearchCV**: Đây là công cụ khoa học nhất để tìm tham số, thay vì thử-sai (Trial & Error) cảm tính. Việc kết hợp Cross-Validation 5-folds đảm bảo tính ổn định của mô hình trên mọi góc nhìn.

## Quản lý Module và Packages (Python Architecture)
- **Tầm quan trọng của `__init__.py` (v4.0.1)**: Mặc dù Python 3 hỗ trợ Implicit Namespace Packages, nhưng trong các ứng dụng Web như Streamlit (có cơ chế Hot Reload), việc thiếu `__init__.py` có thể gây ra lỗi `KeyError: 'module.name'` do bộ nạp module làm mất vết package trong `sys.modules`. Luôn đảm bảo các thư mục `components`, `views`, `src` đều là các Package chính quy.
- **Xử lý Import Error**: Khi gặp lỗi `KeyError` ngay tại dòng `from...import`, nguyên nhân thường nằm ở việc bộ nạp (Importer) bị xung đột bộ nhớ đệm. Giải pháp triệt để là chuẩn hóa cấu trúc Package và khởi tạo lại môi trường runtime.
