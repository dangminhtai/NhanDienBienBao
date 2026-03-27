import streamlit as st
import os
import numpy as np
from PIL import Image
import joblib
import sys

# Thêm thư mục gốc vào path để import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import preprocess_image, get_class_names

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Nhận diện Biển báo Giao thông (V2.2)",
    page_icon="🚦",
    layout="centered"
)

# Custom CSS cho giao diện hiện đại và cân đối
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 300px;
        border-radius: 30px;
        height: 3.5em;
        background-color: #28a745;
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        transition: 0.3s all ease-in-out;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        color: white;
    }
    .prediction-container {
        padding: 30px;
        border-radius: 20px;
        background-color: white;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 30px;
        border-bottom: 8px solid #28a745;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .class-name {
        font-size: 32px;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .class-id {
        font-size: 16px;
        color: #6c757d;
        font-weight: 500;
        letter-spacing: 2px;
    }
    .class-id b {
        color: #28a745;
        font-size: 20px;
    }
    .sidebar-info {
        padding: 15px;
        background-color: #ffffff;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #e1e4e8;
    }
    .header-style {
        text-align: center;
        color: #1e3d59;
        font-weight: 800;
        margin-bottom: 30px;
    }
    /* Căn giữa ảnh st.image */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

def load_models():
    """Tải mô hình và bộ chuẩn hóa từ thư mục streamlit/models/."""
    # Vị trí file app.py hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'last_model.pkl')
    scaler_path = os.path.join(current_dir, 'models', 'scaler_1812.joblib')
    
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    return model, scaler

def main():
    st.sidebar.markdown("""
        <div class="sidebar-info">
        <h3 style='margin-top:0'>🚦 Thông tin hệ thống</h3>
        <p><b>Phiên bản:</b> 2.2 (Path Updated)</p>
        <p><b>Công nghệ:</b> SVM + Hybrid Features</p>
        <p><b>Số chiều:</b> 1812</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='header-style'>🚦 Hệ thống Nhận diện Biển báo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>Tải lên ảnh biển báo để AI phân tích tự động</p>", unsafe_allow_html=True)

    # Tải mô hình
    model, scaler = load_models()
    class_names = get_class_names()

    if model is None:
        st.error(f"❌ Không tìm thấy tệp mô hình tại thư mục `streamlit/models/`.")
        return

    # Upload ảnh - Cần label không rỗng
    uploaded_file = st.file_uploader("📥 Chọn tệp hình ảnh biển báo", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Căn giữa ảnh hiển thị
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', width=400)
        
        # Nút dự đoán căn giữa bằng cột
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 PHÂN TÍCH NGAY")
            
        if predict_button:
            with st.spinner('Đang trích xuất 1812 đặc trưng...'):
                from src.data_utils import extract_hybrid_features
                img_array = np.array(image.convert('RGB'))
                features = extract_hybrid_features([img_array])
                
                if features is not None:
                    features = features.reshape(1, -1)
                    if scaler is not None and scaler.n_features_in_ == 1812:
                        features_input = scaler.transform(features)
                    else:
                        features_input = features
                    
                    prediction = model.predict(features_input)[0]
                    result_name = class_names.get(prediction, "Không xác định")
                    
                    st.markdown(f"""
                        <div class="prediction-container">
                            <div class="class-id">KẾT QUẢ PHÂN TÍCH</div>
                            <div class="class-name">{result_name}</div>
                            <div class="class-id">MÃ NHÃN BIỂN BÁO: <b>#{prediction}</b></div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.error("Lỗi khi xử lý hình ảnh.")

if __name__ == "__main__":
    main()
