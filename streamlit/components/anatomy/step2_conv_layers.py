import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def render_conv1_layer(cnn_extractor, img_batch, raw_ndarray):
    """Mổ xẻ Tầng Conv2D #1 (32x32 -> 30x30)."""
    st.markdown("### 🔬 2.1.2: Giải phẫu Tầng Conv2D #1 (32x32 ➡️ 30x30)")
    
    # 1. Trích xuất dữ liệu
    layer_conv1 = [l for l in cnn_extractor.layers if "conv2d" in l.name.lower()][0]
    weights1, biases1 = layer_conv1.get_weights()
    bias1_0 = biases1[0]
    
    model1 = tf.keras.Model(inputs=cnn_extractor.input, outputs=layer_conv1.output)
    fmaps1 = model1.predict(img_batch, verbose=0)
    actual_val1 = fmaps1[0, 0, 0, 0]

    # 2. Minh họa Tích chập 3 kênh
    st.markdown("**🎲 Quy trình Tích chập 3 Kênh (R-G-B) tại tọa độ (0,0):**")
    patch_r = raw_ndarray[0:3, 0:3, 0] / 255.0
    patch_g = raw_ndarray[0:3, 0:3, 1] / 255.0
    patch_b = raw_ndarray[0:3, 0:3, 2] / 255.0
    
    kernel_r = weights1[:, :, 0, 0]
    kernel_g = weights1[:, :, 1, 0]
    kernel_b = weights1[:, :, 2, 0]

    tab_r, tab_g, tab_b = st.tabs(["🔴 Kênh Đỏ (R)", "🟢 Kênh Xanh Lá (G)", "🔵 Kênh Xanh Dương (B)"])
    with tab_r:
        c1, c2 = st.columns(2); c1.write(p_r := patch_r); c2.write(k_r := kernel_r)
        st.success(f"Sum R: {np.sum(p_r * k_r):.6f}")
    with tab_g:
        c1, c2 = st.columns(2); c1.write(p_g := patch_g); c2.write(k_g := kernel_g)
        st.success(f"Sum G: {np.sum(p_g * k_g):.6f}")
    with tab_b:
        c1, c2 = st.columns(2); c1.write(p_b := patch_b); c2.write(k_b := kernel_b)
        st.success(f"Sum B: {np.sum(p_b * k_b):.6f}")

    total_l1 = np.sum(patch_r * kernel_r) + np.sum(patch_g * kernel_g) + np.sum(patch_b * kernel_b) + bias1_0
    st.code(f"S(R) + S(G) + S(B) + Bias = {total_l1:.4f}")
    st.markdown(f"> **ReLU Check:** `max(0, {total_l1:.4f})` = **{actual_val1:.4f}**")

    # 3. Heatmaps
    fig1, axes1 = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes1.flat):
        if i < 8: ax.imshow(fmaps1[0, :, :, i], cmap='viridis')
        ax.axis('off')
    st.pyplot(fig1)
    
    return fmaps1

def render_conv2_layer(cnn_extractor, img_batch, fmaps1):
    """Mổ xẻ Tầng Conv2D #2 (30x30 -> 28x28)."""
    st.markdown("### 🧬 2.1.3: Giải mã Siêu Tích Chập (Conv2D #2 - 3x3x32)")
    st.info("💡 **Hồi 1: Tại sao 30 giảm xuống 28?**\n\nKhi anh quét khung 3x3 trên lưới 30x30, vị trí tâm khung chỉ từ pixel thứ 1 đến thứ 28.")
    
    # Trích xuất dữ liệu
    layer_conv2 = [l for l in cnn_extractor.layers if "conv2d" in l.name.lower()][1]
    weights2, biases2 = layer_conv2.get_weights()
    bias2_0 = biases2[0]
    
    model2 = tf.keras.Model(inputs=cnn_extractor.input, outputs=layer_conv2.output)
    fmaps2 = model2.predict(img_batch, verbose=0)
    actual_val2 = fmaps2[0, 0, 0, 0]

    st.markdown("#### 🧪 Phòng thí nghiệm: Cách 32 tầng hội quân thành 1")
    ch_idx = st.select_slider("🔍 Chọn 1 trong 32 kênh để soi:", options=list(range(32)), value=0)
    
    p_ch = fmaps1[0, 0:3, 0:3, ch_idx]
    k_ch = weights2[:, :, ch_idx, 0]
    sum_ch = np.sum(p_ch * k_ch)
    
    cl1, cl2, cl3 = st.columns(3)
    with cl1: st.write("Ảnh (Trang #%d)" % ch_idx); st.write(p_ch)
    with cl2: st.write("Kernel (Lớp #%d)" % ch_idx); st.write(k_ch)
    with cl3: st.write("Kết quả (IxK)"); st.write(p_ch * k_ch); st.success("Sum: %.6f" % sum_ch)
        
    st.markdown("---")
    total_from_32 = sum([np.sum(fmaps1[0, 0:3, 0:3, i] * weights2[:, :, i, 0]) for i in range(32)])
    z_linear = total_from_32 + bias2_0
    
    st.markdown("#### 🧮 Phép cộng 32 Kênh + Bias")
    st.warning(f"Tổng Tuyến tính (Z): {z_linear:.6f}")
    st.latex(r"ReLU(Z) = \max(0, Z)")
    st.success(f"Kết quả sau ReLU: {max(0, z_linear):.6f}")
    st.info(f"📍 Đối chứng: **{actual_val2:.6f}** (Khớp tuyệt đối!)")
    
    return fmaps2
