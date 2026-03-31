import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def render_pooling_layer(cnn_extractor, img_batch, fmaps2, pool1_out):
    """Mổ xẻ Tầng MaxPooling2D #1 (28x28 -> 14x14)."""
    st.markdown("### 🧹 2.1.5: Giải phẫu Gạn lọc (MaxPooling_1 - 14x14)")
    st.info("💡 **Góc nhìn thực tế:** Maxwelling sẽ quét qua từng vùng 2x2 và chỉ giữ lại 'người mạnh nhất' (giá trị lớn nhất).")

    # 1. Chọn Filter để soi
    f_idx = st.select_slider("🔍 Bước 1: Chọn Filter muốn gạn lọc (0-63):", options=list(range(64)), value=0, key="pool_f_idx")

    # 2. Chọn vùng tọa độ để quét (Stride = 2)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        y_idx = st.slider("📍 Chọn tọa độ Y (Hàng):", 0, 13, 0)
    with col_s2:
        x_idx = st.slider("📍 Chọn tọa độ X (Cột):", 0, 13, 0)

    # 3. Trích xuất dữ liệu (ĐÃ CÓ TỪ CACHE)
    # Tọa độ thực tế trên map 28x28
    y_start, x_start = y_idx * 2, x_idx * 2
    
    # Lấy patch 2x2 từ fmaps2 (28x28)
    patch2x2 = fmaps2[0, y_start:y_start+2, x_start:x_start+2, f_idx]
    max_val = np.max(patch2x2)
    
    # 4. Hiển thị Trực quan
    col_txt, col_viz = st.columns([0.4, 0.6])
    
    with col_txt:
        st.markdown(f"**Ma trận 2x2 tại ({y_start}, {x_start}) - Filter #{f_idx}**")
        st.write(patch2x2)
        st.markdown(f"**Toán học:**")
        st.latex(rf"max \begin{{pmatrix}} {patch2x2[0,0]:.4f} & {patch2x2[0,1]:.4f} \\ {patch2x2[1,0]:.4f} & {patch2x2[1,1]:.4f} \end{{pmatrix}} = {max_val:.4f}")
        if max_val == 0:
            st.error("⚠️ Vùng này toàn số 0. Anh thử kéo thanh trượt tìm vùng sáng hơn nhé!")
        else:
            st.success(f"✅ Đã gạn lọc được giá trị: {max_val:.4f}")

    with col_viz:
        # Vẽ bản đồ 28x28 và ô vuông đánh dấu
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        map_data = fmaps2[0, :, :, f_idx]
        ax_p.imshow(map_data, cmap='viridis')
        
        # Vẽ hình chữ nhật đánh dấu vùng 2x2 (Tọa độ matplotlib ngược với numpy: x, y)
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_start-0.5, y_start-0.5), 2, 2, linewidth=2, edgecolor='red', facecolor='none')
        ax_p.add_patch(rect)
        ax_p.set_title(f"Bản đồ 28x28 (Filter #{f_idx})")
        ax_p.axis('off')
        st.pyplot(fig_p)

    st.markdown("---")
    st.latex(r"H_{out} = \frac{28}{2} = 14 \quad | \quad W_{out} = \frac{28}{2} = 14")
    st.info(f"💡 **Thành quả:** Sau khi quét hết 14x14 vùng như trên, ta thu được bản đồ cô đặc kích thước 14x14.")
