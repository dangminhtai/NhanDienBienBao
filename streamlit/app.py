import streamlit as st
import os
import numpy as np
from PIL import Image

# Import các module nội bộ từ thư mục src/
from src.data_processor import extract_hybrid_features
from src.model_handler import load_model_and_scaler, get_prediction
from src.class_metadata import get_class_names

def load_css(file_name):
    """Nạp file CSS từ đường dẫn cục bộ."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Recognition v3.1",
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
        <b>Phiên bản:</b> 3.1 (CSS Externalized)<br>
        <b>Model:</b> SVM (1812 Features)
        </div>
    """, unsafe_allow_html=True)

    st.title("🚦 Nhận diện Biển báo Giao thông")
    st.write("Tải lên ảnh biển báo để AI phân tích tự động bằng SVM.")

    # 1. Load Model & Metadata
    model, scaler = load_model_and_scaler(current_dir)
    class_names = get_class_names()

    if model is None:
        st.error("❌ Không tìm thấy mô hình tại `streamlit/models/last_model.pkl`.")
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
                
                # Trích xuất đặc trưng
                features = extract_hybrid_features([img_rgb])
                
                # Dự đoán
                prediction_id = get_prediction(model, scaler, features)
                result_name = class_names.get(prediction_id, "Không xác định")

                # Hiển thị kết quả phong cách card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="result-title">KẾT QUẢ PHÂN TÍCH</div>
                        <div class="result-name">{result_name}</div>
                        <div class="label-id">NHÃN: #{prediction_id}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

if __name__ == "__main__":
    main()
