import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def render_conv1_layer(cnn_extractor, img_batch, raw_ndarray):
    """Mổ xẻ Tầng Conv2D #1 (32x32 -> 30x30)."""
    st.markdown("### 🔬 2.1.2: Giải phẫu Tầng Conv2D #1 (32x32 ➡️ 30x30)")
    
    # 1. Công thức kích thước
    st.latex(r"H_{out} = H_{in} - K + 1 = 32 - 3 + 1 = 30")
    st.latex(r"W_{out} = W_{in} - K + 1 = 32 - 3 + 1 = 30")
    st.caption("💡 Vì không dùng Padding (Valid), mỗi cạnh bị mất 2 pixel do khung 3x3 không thể quét ra ngoài viền.")

    # 2. Trích xuất dữ liệu
    layer_conv1 = [l for l in cnn_extractor.layers if "conv2d" in l.name.lower()][0]
    weights1, biases1 = layer_conv1.get_weights()
    
    # 3. Thanh trượt chọn Filter
    f_idx = st.select_slider(
        "🔍 Chọn 1 trong 32 bộ lọc (Filter) của Tầng 1 để soi chi tiết:",
        options=list(range(32)),
        value=0,
        key="conv1_filter_slider"
    )

    bias1_f = biases1[f_idx]
    model1 = tf.keras.Model(inputs=cnn_extractor.input, outputs=layer_conv1.output)
    fmaps1 = model1.predict(img_batch, verbose=0)
    actual_val1 = fmaps1[0, 0, 0, f_idx]

    # 4. Minh họa Tích chập 3 kênh
    st.markdown(f"#### 🎲 Quy trình Tích chập 3 Kênh (R-G-B) của Filter #{f_idx}")
    st.caption(f"Tại tọa độ (0,0), Filter #{f_idx} sẽ nhân ma trận với 3 lớp màu của ảnh gốc cùng lúc.")
    
    patch_r = raw_ndarray[0:3, 0:3, 0] / 255.0
    patch_g = raw_ndarray[0:3, 0:3, 1] / 255.0
    patch_b = raw_ndarray[0:3, 0:3, 2] / 255.0
    
    kernel_r = weights1[:, :, 0, f_idx]
    kernel_g = weights1[:, :, 1, f_idx]
    kernel_b = weights1[:, :, 2, f_idx]

    tab_r, tab_g, tab_b = st.tabs(["🔴 Kênh Đỏ (R)", "🟢 Kênh Xanh Lá (G)", "🔵 Kênh Xanh Dương (B)"])
    with tab_r:
        c1, c2 = st.columns(2); c1.write(p_r := patch_r); c2.write(k_r := kernel_r)
        sum_r = np.sum(p_r * k_r)
        st.success(f"Sum R: {sum_r:.6f}")
    with tab_g:
        c1, c2 = st.columns(2); c1.write(p_g := patch_g); c2.write(k_g := kernel_g)
        sum_g = np.sum(p_g * k_g)
        st.success(f"Sum G: {sum_g:.6f}")
    with tab_b:
        c1, c2 = st.columns(2); c1.write(p_b := patch_b); c2.write(k_b := kernel_b)
        sum_b = np.sum(p_b * k_b)
        st.success(f"Sum B: {sum_b:.6f}")

    total_l1 = sum_r + sum_g + sum_b + bias1_f
    st.markdown("---")
    st.markdown("#### 🧮 Phép toán Tổng lực (3 Kênh + Bias)")
    st.code(f"{sum_r:.6f} (R) + {sum_g:.6f} (G) + {sum_b:.6f} (B) + {bias1_f:.6f} (Bias) = {total_l1:.6f}")
    st.warning(f"➡️ **Tổng Tuyến tính (Z): {total_l1:.6f}**")
    
    st.latex(r"ReLU(Z) = \max(0, Z)")
    st.success(f"➡️ **Kết quả sau ReLU: {max(0, total_l1):.6f}**")
    st.info(f"📍 **Đối chứng:** Giá trị thực tế tại ô (0,0,f={f_idx}) là: **{actual_val1:.6f}** (Khớp tuyệt đối!)")

    # 5. Heatmaps (Show top 8 filters for context)
    st.markdown(f"**🖼️ Bản đồ đặc trưng (Top 8 / 32):**")
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
