import streamlit as st
import os

def render_cnn_overview(raw_ndarray, current_dir):
    """Bản đồ tổng quan và Chuẩn hóa (Normalization)."""
    # 1. Hiển thị nội dung plan2.md
    plan2_path = os.path.join(current_dir, "docs", "predict", "plans", "plan2.md")
    if os.path.exists(plan2_path):
        with open(plan2_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
            
    # 2. Tracking chuẩn hóa (Normalization)
    st.markdown("### 🛠️ 2.1: Phép chia Chuẩn hóa (Dữ liệu vào CNN)")
    normalized_rows = (raw_ndarray[0:3, :, 0]) / 255.0
    st.code(f"""
# Công thức: raw_ndarray / 255.0
# Kết quả: 3 dòng đầu (Full 32 cột) - Kênh màu Đỏ

{normalized_rows}
    """, language="python")
    st.caption("Dữ liệu lúc này đã là số thực [0, 1]. Đây là 'thức ăn' chuẩn cho các Nơ-ron.")

    # 3. Bản đồ dòng chảy (Architecture Map)
    st.markdown("---")
    st.markdown("### 🗺️ 2.0: Bản đồ Dòng chảy Đặc trưng (Full Flow)")
    
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap; padding: 20px; background: #f8f9fa; border-radius: 15px; border: 1px solid #dee2e6;">
        <div style="background: #fff9c4; padding: 10px; border: 2px solid #fbc02d; border-radius: 8px; text-align: center; min-width: 100px;">
            <b style="color: #f57f17;">INPUT</b><br/><small>32x32x3</small>
        </div>
        <div style="color: #adb5bd;">➡️</div>
        <div style="background: #e3f2fd; padding: 10px; border: 2px solid #2196f3; border-radius: 8px; text-align: center; min-width: 110px;">
            <b style="color: #1565c0;">KHỐI 1</b><br/><small>14x14x64</small>
        </div>
        <div style="color: #adb5bd;">➡️</div>
        <div style="background: #f3e5f5; padding: 10px; border: 2px solid #9c27b0; border-radius: 8px; text-align: center; min-width: 110px;">
            <b style="color: #7b1fa2;">KHỐI 2</b><br/><small>5x5x64</small>
        </div>
        <div style="color: #adb5bd;">➡️</div>
        <div style="background: #efebe9; padding: 10px; border: 2px solid #795548; border-radius: 8px; text-align: center; min-width: 100px;">
            <b style="color: #4e342e;">FLATTEN</b><br/><small>1.600 số</small>
        </div>
        <div style="color: #adb5bd;">➡️</div>
        <div style="background: #c8e6c9; padding: 10px; border: 2px solid #388e3c; border-radius: 8px; text-align: center; min-width: 110px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <b style="color: #1b5e20;">DENSE 256</b><br/><small>Mã Gene</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **Hành trình "Gạn đục khơi trong":**
    1. **Khối 1 & 2:** Biến đổi Pixel thành khối đặc trưng.
    2. **Flatten:** Xếp chồng 1.600 con số trung gian thành hàng dài.
    3. **Dense:** "Đúc" lại thành đúng **256 từ khóa then chốt**.
    """)
