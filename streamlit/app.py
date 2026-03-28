import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import các module nội bộ từ thư mục src/
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import load_hybrid_system, predict_hybrid
from src.class_metadata import get_class_names

def load_css(file_name):
    """Nạp file CSS từ đường dẫn cục bộ."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Recognition v4.0",
    page_icon="🚦",
    layout="centered"
)

# --- NẠP CSS NGOÀI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
load_css(os.path.join(current_dir, 'assets', 'style.css'))

def main():
    # Sidebar
    st.sidebar.title("🚦 Hệ thống Nhận diện")
    st.sidebar.markdown("""
        <div style="padding:10px; border:1px solid #eee; border-radius:10px">
        <b>Phiên bản:</b> 4.0 (Hybrid)<br>
        <b>Kiến trúc:</b> CNN + SVM Linear<br>
        <b>Đặc trưng:</b> Deep Features (256 dims)
        </div>
    """, unsafe_allow_html=True)

    st.title("🚦 Nhận diện Biển báo Giao thông")
    st.write("Giải pháp lai ghép tối ưu: Sử dụng sức mạnh trích xuất đặc trưng của **CNN** và khả năng phân loại chính xác của **SVM**.")

    # 1. Load Hybrid System
    cnn_extractor, scaler, svm_model = load_hybrid_system()
    class_names = get_class_names()

    if cnn_extractor is None or scaler is None or svm_model is None:
        st.warning("⚠️ Hệ thống mô hình đang được thiết lập hoặc thiếu thành phần. Vui lòng kiểm tra thông báo lỗi bên trên.")
        return

    # 2. Giao diện Upload
    uploaded_file = st.file_uploader("📥 Chọn ảnh biển báo...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        # Căn giữa ảnh bằng cột
        row_col1, row_col2, row_col3 = st.columns([1, 2, 1])
        with row_col2:
            st.image(image, caption='Ảnh đã tải lên', width=300)
        
        # Nút dự đoán
        if st.button("🔍 BẮT ĐẦU NHẬN DIỆN"):
            with st.spinner('CNN đang trích xuất đặc trưng sâu...'):
                # Tiền xử lý
                img_batch = preprocess_image_for_cnn(image)
                
                # Dự đoán lai ghép
                prediction_id, confidence = predict_hybrid(img_batch, cnn_extractor, scaler, svm_model)
                result_name = class_names.get(prediction_id, "Không xác định")

                # Hiển thị kết quả phong cách card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="result-title">KẾT QUẢ PHÂN TÍCH HYBRID</div>
                        <div class="result-name">{result_name}</div>
                        <div class="label-id">NHÃN: #{prediction_id} | ĐỘ TIN CẬY (MARGIN): {confidence:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

                # --- PHẦN MINH BẠCH TOÁN HỌC (Update cho v4.0) ---
                st.divider()
                with st.expander("📊 CƠ SỞ TOÁN HỌC VÀ QUY TRÌNH LAI GHÉP (v4.0)"):
                    
                    st.subheader("1. Tại sao dùng Hybrid (C-SVM)?")
                    st.write("""
                    Hệ thống này kết hợp hai "ông vua" trong học máy:
                    - **CNN (Convolutional Neural Network)**: Chuyên gia trong việc hiểu cấu trúc ảnh, thay thế cho HOG truyền thống.
                    - **SVM (Support Vector Machine)**: Chuyên gia trong việc tìm ranh giới phân loại tối ưu (Maximum Margin).
                    """)

                    st.subheader("2. Quy trình xử lý dữ liệu (Inference Pipeline)")
                    st.markdown("""
                    ```mermaid
                    graph LR
                        A[Ảnh gốc] --> B[Resize 32x32]
                        B --> C[CNN Feature Extractor]
                        C --> D[Deep Features - 256 dims]
                        D --> E[Standard Scaler]
                        E --> F[SVM Linear Classifier]
                        F --> G[Kết quả cuối cùng]
                    ```
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("**CNN Feature Extraction**")
                        st.write("""
                        Lớp `feature_dense` (256 node) của CNN đóng vai trò là bộ trích xuất đặc trưng tự động. 
                        Nó học được các mẫu hình học phức tạp mà HOG có thể bỏ sót.
                        """)
                    
                    with col2:
                        st.info("**SVM classification**")
                        st.latex(r"w^T \cdot \phi(x) + b = 0")
                        st.write(f"SVM sử dụng Kernel tuyến tính để phân loại vector 256 chiều này. Độ tin cậy được tính bằng khoảng cách tới siêu mặt phẳng quyết định.")

                    st.subheader("3. Dữ liệu số (Numerical Trace)")
                    # Lấy thử feature vector từ CNN để hiển thị
                    deep_feats = cnn_extractor.predict(img_batch, verbose=0)
                    st.write(f"Vector đặc trưng sâu (Trích đoạn 20/256 chiều):")
                    st.code(deep_feats[0][:20])
                    st.write(f"Giá trị Margin lớn nhất tìm thấy: **{confidence:.4f}**")

if __name__ == "__main__":
    main()
