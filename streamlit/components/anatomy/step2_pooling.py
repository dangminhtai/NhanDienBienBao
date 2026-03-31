import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def render_pooling_layer(cnn_extractor, img_batch, fmaps2):
    """Mổ xẻ Tầng MaxPooling2D #1 (28x28 -> 14x14)."""
    st.markdown("### 🧹 2.2c: Giải phẫu Gạn lọc (MAX-POOL 14x14)")
    
    # 1. Trích xuất dữ liệu
    layer_pool1 = [l for l in cnn_extractor.layers if "pool" in l.name.lower()][0]
    modelp = tf.keras.Model(inputs=cnn_extractor.input, outputs=layer_pool1.output)
    outp = modelp.predict(img_batch, verbose=0)
    
    # 2. Tìm vùng nóng trên ma trận 28x28
    map2d = outp[0, :, :, 0]
    y_m, x_m = np.unravel_index(np.argmax(map2d), map2d.shape)
    y_i, x_i = y_m * 2, x_m * 2
    
    # 3. Lấy patch 2x2 từ fmaps2 (28x28)
    patch2x2 = fmaps2[0, y_i:y_i+2, x_i:x_i+2, 0]
    
    col_p1, col_img, col_p2 = st.columns([1, 0.8, 1])
    with col_p1:
        st.write(f"**Ma trận 2x2**"); st.write(patch2x2)
    with col_img:
        fig_mini, ax_mini = plt.subplots(figsize=(2,2))
        ax_mini.imshow(patch2x2, cmap='viridis'); ax_mini.axis('off')
        st.pyplot(fig_mini)
    with col_p2:
        st.success(f"**Kết quả (Max): {outp[0, y_m, x_m, 0]:.4f}**")
    
    st.latex(r"H_{out} = \frac{28}{2} = 14")
    st.info("💡 **Thành quả KHỐI 1:** Ảnh 32x32 đã được 'cô đặc' thành khối 14x14x64.")
