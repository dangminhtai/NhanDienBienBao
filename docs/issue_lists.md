# 🐞 Danh sách Lỗi (Issue Tracking)

Dưới đây là danh sách các lỗi kỹ thuật phát hiện được trong quá trình phát triển hệ thống X-Ray CNN.

| ID | Tên Lỗi | Chi tiết | Trạng thái |
|---|---|---|---|
| ISSUE#001 | State Reset on Interaction | Khi kéo Slider ở Bước 2.1.2 hoặc 2.1.3, toàn bộ kết quả phân tích bị mất, bắt người dùng phải "Nhấn nút" lại từ đầu. | ✅ Đã Fix |
| ISSUE#002 | Module Collision | Xung đột tên 'streamlit' với thư viện hệ thống khi import các thành phần anatomy. | ✅ Đã Fix |
| ISSUE#003 | Math Mismatch (ReLU) | Kết quả tính tay và kết quả mô hình không khớp do thiếu bước ReLU Activation. | ✅ Đã Fix |
| ISSUE#004 | KeyError Module Import | Lỗi KeyError: 'components.anatomy.step2_conv_layers' khi import tại single_predict.py. | ✅ Đã Fix |
| ISSUE#005 | Video Upload Limit | Mặc định Streamlit giới hạn upload 200MB, gây lỗi khi tải video clip dài. | ✅ Đã Fix |
| ISSUE#006 | Video Directory Mismatch | Đường dẫn thư mục videos bị lặp lại 'streamlit/streamlit/videos'. | ✅ Đã Fix |
| ISSUE#008 | Video Mode UI Missing | Giao diện Video không hiện ra do tham số check app_mode bị lệch tên (đã thêm - Beta). | ✅ Đã Fix |
| ISSUE#009 | NameError 'st' | Thiếu import streamlit as st trong math_visualizer.py sau khi refactor. | ✅ Đã Fix |
| ISSUE#010 | CMS Hot Reload | File content.json thay đổi nhưng UI không cập nhật do Singleton cache. | ✅ Đã Fix |

---
*Ghi chú: Issue #008 sửa lại string comparison trong app.py.*
