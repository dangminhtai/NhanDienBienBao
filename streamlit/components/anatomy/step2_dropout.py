import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_dropout_layer(pool1_out):
    """Mô phỏng Tầng Dropout (Lưới bảo vệ chống học vẹt)."""
    st.markdown("### 🚿 2.1.6: Giải phẫu Lưới bảo vệ (Dropout - 25%)")
    
    st.info("""
    💡 **Bản chất của Dropout:** Trong quá trình Huấn luyện, AI sẽ ngẫu nhiên "tắt" (cho bằng 0) một tỷ lệ nơ-ron (ở đây là 25%). 
    Việc này ép các nơ-ron còn lại phải học cách tự đứng vững, không được "dựa dẫm" vào nhau, giúp mô hình bớt "học vẹt".
    """)

    # 1. Chọn Filter để xem (Đã qua Pooling nên là 14x14)
    f_idx = st.select_slider("🔍 Chọn Filter muốn xem hiệu ứng (0-63):", options=list(range(64)), value=0, key="dropout_f_idx")
    
    # 2. Chế độ mô phỏng
    col_mode, col_info = st.columns([0.4, 0.6])
    with col_mode:
        simulate_train = st.toggle("🧪 Mô phỏng Chế độ Huấn luyện (Training Mode)", value=False)
    
    # Lấy dữ liệu 14x14 từ Filter cụ thể
    data = pool1_out[0, :, :, f_idx]
    
    if simulate_train:
        # Tạo mask ngẫu nhiên (Fixed seed để ổn định khi scroll)
        np.random.seed(42 + f_idx)
        mask = np.random.rand(*data.shape) > 0.25
        dropout_data = data * mask
        num_dropped = np.sum(~mask)
        with col_info:
            st.warning(f"⚠️ Đã ngắt kết nối **{num_dropped}** điểm dữ liệu (vùng tối/đen trên hình).")
    else:
        dropout_data = data
        with col_info:
            st.success("✅ Chế độ Dự đoán (Inference): 100% tín hiệu được truyền qua.")

    # 3. Hiển thị Trực quan
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(dropout_data, cmap='viridis')
    ax.set_title(f"Bản đồ 14x14 (Filter #{f_idx}) - " + ("Training" if simulate_train else "Inference"))
    ax.axis('off')
    
    # Nếu mô phỏng, vẽ các dấu X đỏ lên điểm bị dropout
    if simulate_train:
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if not mask[y, x]:
                    ax.text(x, y, '×', color='red', ha='center', va='center', fontsize=8, alpha=0.6)

    st.pyplot(fig)
    
    st.markdown("---")
    st.caption("Đây là bước 'luyện quân' khắc nghiệt, giúp AI của chúng ta bền bỉ hơn khi gặp các biển báo bị rách hoặc mờ trong thực tế.")
