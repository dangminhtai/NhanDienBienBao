import streamlit as st
import os
from PIL import Image

# Import Helper Functions
from components.ui_helpers import load_css
from src.model_handler import load_hybrid_system
from src.class_metadata import get_class_names

# Import Views
from views.single_predict import render_single_predict_view
from views.batch_process import render_batch_process_view
from views.full_image_detect import render_full_image_view
from views.math_visualizer import render_math_section

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Recognition v4.0",
    page_icon="🚦",
    layout="wide" 
)

# --- NẠP CSS NGOÀI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
load_css(os.path.join(current_dir, 'assets', 'style.css'))

def main():
    # Sidebar
    st.sidebar.title("🚦 Hệ thống Nhận diện")
    
    app_mode = st.sidebar.radio("Chọn chế độ hoạt động:", 
                                ["Dự đoán nhanh (Single Sign)", "Phát hiện & Nhận diện (Full Image)", "Quét Thư mục (Batch Mode)"])
    
    st.sidebar.markdown("---")
    
    # Tham số Detection (Chỉ hiện khi ở chế độ Full Image hoặc Batch Mode)
    det_params = {}
    auto_tune = False
    show_debug = False
    
    if app_mode in ["Phát hiện & Nhận diện (Full Image)", "Quét Thư mục (Batch Mode)"]:
        st.sidebar.subheader("🛠️ Cấu hình Detection")
        det_params['min_s'] = st.sidebar.slider("Độ bão hòa tối thiểu (Saturation)", 0, 255, 80)
        det_params['min_v'] = st.sidebar.slider("Độ sáng tối thiểu (Value)", 0, 255, 40)
        det_params['min_size'] = st.sidebar.slider("Kích thước tối thiểu (Px)", 10, 200, 30)
        det_params['nms_thresh'] = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.3)
        
        st.sidebar.markdown("---")
        st.sidebar.write("### 📐 Tùy chỉnh Hình học (Bước 4)")
        det_params['min_ratio'], det_params['max_ratio'] = st.sidebar.slider(
            "Dải tỷ lệ khung hình (Rộng/Cao)", 
            0.1, 2.0, (0.6, 1.4), 0.1,
            help="Biển báo chuẩn thường là 1.0 (Hình vuông). Loại bỏ các vật thể quá dài hoặc quá dẹt."
        )
        det_params['min_solidity'] = st.sidebar.slider(
            "Độ đặc tối thiểu (Solidity)", 
            0.1, 1.0, 0.3, 0.05,
            help="Tỷ lệ diện tích màu trắng so với diện tích ô vuông. Biển báo kim loại thường rất 'đặc'."
        )
        
        st.sidebar.markdown("---")
        st.sidebar.write("### 📸 Tùy chỉnh Chất lượng (Bước 5)")
        det_params['min_laplacian'] = st.sidebar.slider(
            "Độ nét Laplacian (0 = Tắt)", 
            0, 4000, 40, 5,
            help="Lọc các vùng bị mờ nhòe. Kéo về 0 để Tắt bước này nếu ảnh luôn sắc nét."
        )
        
        auto_tune = st.sidebar.checkbox("Chế độ Hyper-Scan (Siêu quét 16 vòng)", value=False)
        show_debug = st.sidebar.checkbox("Hiện mặt nạ quét màu (Debug Mask)", value=False)
    
    st.sidebar.markdown(f"""
        <div style="padding:10px; border:1px solid #eee; border-radius:10px">
        <b>Phiên bản:</b> 4.0 (Hybrid)<br>
        <b>Kiến trúc:</b> CNN + SVM Linear<br>
        <b>Khoa học:</b> SOLID Refactored
        </div>
    """, unsafe_allow_html=True)

    st.title("🚦 Nhận diện Biển báo Giao thông")
    
    # 1. Load Recognition Models (Dùng chung cho cả 3 chế độ)
    cnn_extractor, rec_scaler, svm_model = load_hybrid_system()
    class_names = get_class_names()

    if cnn_extractor is None or rec_scaler is None or svm_model is None:
        st.warning("⚠️ Hệ thống nhận diện đang thiếu thành phần. Vui lòng kiểm tra models/")
        return

    # 2. Rẻ nhánh sang các Views theo kiến trúc SOLID Router
    if app_mode == "Quét Thư mục (Batch Mode)":
        render_batch_process_view(current_dir, cnn_extractor, rec_scaler, svm_model, class_names, det_params, auto_tune)
        
    else:
        # Chế độ ảnh đơn (Upload: dùng chung cho Single và Full Image)
        uploaded_file = st.file_uploader("📥 Chọn ảnh...", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            
            if app_mode == "Dự đoán nhanh (Single Sign)":
                render_single_predict_view(image, app_mode, cnn_extractor, rec_scaler, svm_model, class_names, current_dir)

            elif app_mode == "Phát hiện & Nhận diện (Full Image)":
                render_full_image_view(
                    image, class_names, current_dir, det_params, 
                    auto_tune, show_debug, cnn_extractor, rec_scaler, svm_model
                )
                
    # --- PHẦN MINH BẠCH TOÁN HỌC ---
    # Section này luôn hiện ở cuối các chế độ
    render_math_section(app_mode)

if __name__ == "__main__":
    main()
