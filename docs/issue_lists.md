# 🐞 Danh sách Lỗi (Issue Tracking)

Dưới đây là danh sách các lỗi kỹ thuật phát hiện được trong quá trình phát triển hệ thống X-Ray CNN.

| ID | Tên Lỗi | Chi tiết | Trạng thái |
|---|---|---|---|
| ISSUE#001 | State Reset on Interaction | Khi kéo Slider ở Bước 2.1.2 hoặc 2.1.3, toàn bộ kết quả phân tích bị mất, bắt người dùng phải "Nhấn nút" lại từ đầu. | ✅ Đã Fix |
| ISSUE#002 | Module Collision | Xung đột tên 'streamlit' với thư viện hệ thống khi import các thành phần anatomy. | ✅ Đã Fix |
| ISSUE#003 | Math Mismatch (ReLU) | Kết quả tính tay và kết quả mô hình không khớp do thiếu bước ReLU Activation. | ✅ Đã Fix |
| ISSUE#004 | KeyError Module Import | Lỗi KeyError: 'components.anatomy.step2_conv_layers' khi import tại single_predict.py. | ✅ Đã Fix |

---
*Ghi chú: Issue #001 sẽ được fix bằng giải pháp st.session_state.*
