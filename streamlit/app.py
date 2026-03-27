import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import các module nội bộ từ thư mục src/
from src.data_processor import extract_hybrid_features, get_hsv_histograms
from src.model_handler import load_model_and_scaler, get_prediction
from src.class_metadata import get_class_names

def load_css(file_name):
    """Nạp file CSS từ đường dẫn cục bộ."""
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
        <b>Phiên bản:</b> 4.0 (Math Tracking)<br>
        <b>Model:</b> SVM (1812 Features)
        </div>
    """, unsafe_allow_html=True)

    st.title("🚦 Nhận diện Biển báo Giao thông")
    st.write("Tải lên ảnh biển báo để AI phân tích tự động bằng SVM.")

    # 1. Load Model & Metadata
    model, scaler = load_model_and_scaler(current_dir)
    class_names = get_class_names()

    if model is None or scaler is None:
        st.error("❌ Không tìm thấy mô hình hoặc bộ chuẩn hóa tại thư mục `models/`.")
        return

    # 2. Giao diện Upload
    uploaded_file = st.file_uploader("📥 Chọn ảnh biển báo...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        # Căn giữa ảnh bằng cột
        row_col1, row_col2, row_col3 = st.columns([1, 2, 1])
        with row_col2:
            st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
        
        # Nút dự đoán
        if st.button("🔍 BẮT ĐẦU NHẬN DIỆN"):
            with st.spinner('Đang phân tích đặc trưng Hybrid...'):
                img_rgb = np.array(image.convert('RGB'))
                
                # Trích xuất đặc trưng kèm visualization
                features, hog_img = extract_hybrid_features([img_rgb], visualize=True)
                
                # Dự đoán kèm độ tin cậy
                prediction_id, confidence = get_prediction(model, scaler, features)
                result_name = class_names.get(prediction_id, "Không xác định")

                # Hiển thị kết quả phong cách card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="result-title">KẾT QUẢ PHÂN TÍCH</div>
                        <div class="result-name">{result_name}</div>
                        <div class="label-id">NHÃN: #{prediction_id} | ĐỘ TIN CẬY: {confidence*100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

                # --- PHẦN MINH BẠCH TOÁN HỌC (REQ008) ---
                st.divider()
                with st.expander("📊 CƠ SỞ TOÁN HỌC & MINH BẠCH DỮ LIỆU"):
                    st.subheader("1. Trực quan hóa Đặc trưng (Feature Visualization)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("**HOG (Histogram of Oriented Gradients)**")
                        st.image(hog_img, caption="Bản đồ Gradient (Cạnh & Hình dạng)", use_container_width=True, clamp=True)
                        st.write("Mô hình tập trung vào các đường biên và góc cạnh của biển báo.")
                    
                    with col2:
                        st.info("**HSV (Color Histograms)**")
                        hsv_feats = get_hsv_histograms(img_rgb)
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.bar(range(16), hsv_feats[0:16], color='red', alpha=0.6, label='Hue')
                        ax.bar(range(16), hsv_feats[16:32], color='green', alpha=0.4, label='Sat')
                        ax.bar(range(16), hsv_feats[32:48], color='blue', alpha=0.2, label='Val')
                        ax.set_title("HSV Color Bins (48 dims)")
                        st.pyplot(fig)
                        st.write("Phân bố màu sắc giúp phân biệt biển báo Đỏ (cấm) và Xanh (chỉ dẫn).")

                    st.subheader("2. Giải thích Thuật toán (Algorithms Explained)")
                    
                    tab1, tab2 = st.tabs(["Cơ chế HOG", "Cơ chế SVM"])
                    
                    with tab1:
                        st.latex(r"G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)")
                        st.write("""
                        HOG chia ảnh thành các ô (cells) 4x4 pixels. Với mỗi ô, nó tính toán độ lớn gradient và hướng của nó.
                        Các hướng này được gom vào 9 bins (0-180°). 
                        - **Kích thước vector**: 32x32 pixels / 4x4 cells = 8x8 blocks. 
                        - Mỗi block 2x2 cells có $4 \times 9 = 36$ đặc trưng.
                        - Tổng cộng: $7 \times 7 \times 36 = 1764$ đặc trưng HOG.
                        """)

                    with tab2:
                        st.latex(r"f(x) = \text{sgn}\left( \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b \right)")
                        st.write("""
                        SVM tìm kiếm một siêu phẳng (hyperplane) tối ưu trong không gian 1812 chiều để phân tách các lớp biển báo.
                        Hàm Kernel được sử dụng là **RBF (Radial Basis Function)**:
                        """)
                        st.latex(r"K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)")
                        st.write("Độ tin cậy được tính dựa trên khoảng cách từ vector đặc trưng tới siêu phẳng quyết định.")

                    st.subheader("3. Dữ liệu số (Numerical Trace)")
                    st.write(f"Vector đặc trưng đầu vào (Trích đoạn 20/1812 chiều):")
                    st.code(features[0][:20])

if __name__ == "__main__":
    main()
