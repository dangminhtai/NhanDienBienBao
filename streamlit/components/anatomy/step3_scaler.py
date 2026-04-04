import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def render_step3_standardization(deep_features, scaler, current_dir):
    """Mổ xẻ Bước 3.1: Chuẩn hóa Đặc trưng (Standardization)."""
    st.markdown("---")
    st.header("⚖️ Bước 3: Cân bằng Đặc trưng (Standardization)")
    
    # 1. Hướng dẫn lý thuyết từ file step3.md (Nếu có)
    step3_path = os.path.join(current_dir, "docs", "predict", "steps", "step3.md")
    if os.path.exists(step3_path):
        with open(step3_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # 2. Thực hiện chuẩn hóa thực tế
    # deep_features shape: (1, 256)
    scaled_features = scaler.transform(deep_features)
    
    # 3. Trực quan hóa So sánh (Trước vs Sau)
    st.markdown("### 📊 So sánh Dữ liệu: Trước vs Sau Cân bằng")
    
    col_viz, col_math = st.columns([0.6, 0.4])
    
    with col_viz:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Raw features (CNN Output)
        ax.plot(deep_features.flatten(), label="Trước (CNN Raw)", color='gray', alpha=0.5, linewidth=1)
        # Scaled features (SVM Input)
        ax.plot(scaled_features.flatten(), label="Sau (Standardized)", color='#FF4B4B', linewidth=1.5)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_title("Biến đổi 256 Mã Gene về cùng Hệ quy chiếu")
        ax.legend()
        st.pyplot(fig)
        st.caption("Anh có thể thấy đường màu đỏ (Sau) đã được kéo về quanh mức 0, giúp SVM cá cân bằng hơn.")

    with col_math:
        st.markdown("**🔢 Tính toán thực tế (3 Gene ngẫu nhiên):**")
        st.latex(rf"z = \frac{{x - \mu}}{{\sigma}}")
        
        # Chọn 3 chỉ số ngẫu nhiên
        np.random.seed(42)
        random_indices = np.random.choice(256, 3, replace=False)
        
        calc_data = []
        for idx in random_indices:
            x_val = deep_features[0, idx]
            mu_val = scaler.mean_[idx]
            sigma_val = np.sqrt(scaler.var_[idx])
            z_val = scaled_features[0, idx]
            
            calc_data.append({
                "Gene #": idx,
                "Gốc (x)": f"{x_val:.4f}",
                "TB (μ)": f"{mu_val:.4f}",
                "Độ lệch (σ)": f"{sigma_val:.4f}",
                "Kết quả (z)": f"{z_val:.4f}"
            })
        
        st.table(pd.DataFrame(calc_data))
        st.success("✅ Mọi Gene bây giờ đều có 'tiếng nói' ngang nhau!")

    # --- PHẦN GIẢI THÍCH MỚI CHO NGƯỜI XEM ---
    with st.expander("💡 Giải mã: Con số Trung bình (μ) và Độ lệch (σ) từ đâu mà có?", expanded=False):
        st.markdown("""
        Để có được bảng tính trên, hệ thống đã thực hiện qua 2 giai đoạn:
        
        1. **Trong Quá khứ (Huấn luyện - Training):** 
            - Máy tính quét qua **34.000 tấm ảnh** biển báo mẫu.
            - Với mỗi Gene (trong 256 Gene), nó lập một bảng điểm cho toàn bộ 34.000 tấm ảnh đó.
            - **Trung bình (μ):** Là điểm số trung bình của 'xã hội' biển báo đối với Gene đó.
            - **Độ lệch (σ):** Là mức độ 'nhảy số' (biến thiên) của cả lớp.
            - Các con số này được **Đóng băng (Freeze)** và lưu vào bộ nhớ như một **'Cây thước đo chuẩn'**.
        
        2. **Ở Hiện tại (Dự đoán - Predict):**
            - Khi anh đưa 1 tấm ảnh mới vào, nó 'đẻ' ra giá trị **Gốc (x)**.
            - Máy tính **KHÔNG** tính lại trung bình. Nó lấy ngay **'Cây thước đo chuẩn'** ở bước 1 ra để so sánh.
            - Phép tính $z = (x - \mu) / \sigma$ chính là cách để xem: "So với 34.000 ảnh cũ, ảnh mới này đang đứng ở đâu? Cao hơn hay thấp hơn bao nhiêu?"
        
        **Kết luận:** Nhờ vậy, SVM luôn có một hệ quy chiếu công bằng để so sánh mọi tấm ảnh với nhau!
        """)
