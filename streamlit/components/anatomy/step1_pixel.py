import streamlit as st
import numpy as np
import os

def render_step1_pixel_tracking(image, current_dir):
    """Giải phẫu Bước 1: Tiền xử lý & Ma trận ndarray."""
    with st.expander("Bước 1: Tiền Xử Lý Hình Ảnh", expanded=True):
        # Đọc lời giải thích từ file step1.md
        step1_path = os.path.join(current_dir, "docs", "predict", "steps", "step1.md")
        if os.path.exists(step1_path):
            with open(step1_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        
        # Xử lý ma trận
        img_32x32 = image.resize((32, 32))
        raw_ndarray = np.array(img_32x32)
        
        col_i1, col_i2 = st.columns([0.4, 0.6])
        with col_i1:
            st.image(img_32x32, width=150, caption='Ảnh 32x32 (Pixel thực)') 
            
        with col_i2:
            sample_rows = raw_ndarray[0:3, :, 0] 
            st.code(f"""
# Trạng thái: Dữ liệu pixel gốc (uint8: 0-255)
# Hiển thị: 3 dòng đầu tiên (Full 32 cột) - Kênh Đỏ

{sample_rows}
            """, language="python")
        
        return raw_ndarray

def get_pixel_data(image):
    """Trích xuất ma trận pixel 32x32 thầm lặng."""
    img_32x32 = image.resize((32, 32))
    return np.array(img_32x32)
