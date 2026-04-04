import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def render_dense_juicer(cnn_extractor, flatten_out, deep_features):
    """Mổ xẻ Bước 2.3: Nén đặc trưng (Flatten & Dense)."""
    st.markdown("---")
    st.header("🧬 2.3: BƯỚC NÉN (Định danh)")

    # --- 2.3.1: Flatten ---
    st.markdown("### 📏 2.3.1: Giải phẫu San phẳng (Flatten)")
    col_flat_txt, col_flat_viz = st.columns([0.5, 0.5])
    
    with col_flat_txt:
        st.info(rf"""
        💡 **Hành động:** AI duỗi thẳng khối dữ liệu 3D (5x5x64) thành một hàng duy nhất.
        - **Toán học:** $5 \times 5 \times 64 = 1.600$ con số thô.
        """)
        flatten_1d = flatten_out.flatten()
        st.write("**Mẫu 10 số đầu tiên sau khi duỗi:**")
        st.code(flatten_1d[:10])

    with col_flat_viz:
        fig_f, ax_f = plt.subplots(figsize=(6, 2))
        ax_f.plot(flatten_1d, color='gray', alpha=0.5, linewidth=0.5)
        ax_f.set_title("Vector 1.600 chiều (Dữ liệu Thô)")
        ax_f.axis('off')
        st.pyplot(fig_f)

    # --- 2.3.2: Dense (feature_dense) ---
    st.markdown("### 🧬 2.3.2: Máy ép Đặc trưng (Dense - 256 Mã Gene)")
    
    # --- PHẦN GIẢI THÍCH "BÌNH DÂN" ---
    st.success("""
    💡 **Cơ chế 'Bầu chọn có trọng số':** 
    Thay vì giữ 1.600 số thô, AI tạo ra 256 'Ông đại biểu' (Gene). 
    Mỗi ông đại biểu sẽ nhân 1.600 số thô với một bộ trọng số riêng để tính ra điểm số của mình.
    """)

    # --- TRỰC QUAN HÓA CHI TIẾT (UN-BLACKBOX) ---
    try:
        dense_layer = cnn_extractor.layers[-1]
        weights, biases = dense_layer.get_weights()
        
        # 1. Cho người dùng chọn Gene để xem "Bầu chọn"
        gene_idx = st.slider("🔍 Soi chi tiết cách tạo ra Gene số (0-255):", 0, 255, 0)
        
        # 2. Tính toán sự đóng góp (Dữ liệu vào * Trọng số)
        gene_weights = weights[:, gene_idx]
        flatten_1d = flatten_out.flatten()
        contributions = flatten_1d * gene_weights # Từng thành phần một: x_i * w_i
        
        col_calc, col_top = st.columns([0.5, 0.5])
        
        with col_calc:
            st.markdown(f"**Toán học của Gene #{gene_idx}:**")
            st.latex(rf"Intensity = \sum_{{i=1}}^{{1600}} (X_i \times W_i) + Bias")
            
            # Hiển thị 5 phép tính mẫu đầu tiên
            st.write("**Ví dụ 5 'ông cử tri' đầu tiên:**")
            examples = []
            for i in range(5):
                examples.append({
                    "Cử tri (X)": f"{flatten_1d[i]:.4f}",
                    "Trọng số (W)": f"{gene_weights[i]:.4f}",
                    "Kết quả (X*W)": f"{contributions[i]:.4f}"
                })
            st.table(pd.DataFrame(examples))

        with col_top:
            st.markdown(f"**Lược đồ đóng góp (Contribution Map):**")
            # Vẽ biểu đồ các đóng góp
            fig_cont, ax_cont = plt.subplots(figsize=(6, 3))
            ax_cont.bar(range(len(contributions)), contributions, color='green', alpha=0.7)
            ax_cont.set_title(f"Phân bổ điểm số cho Gene #{gene_idx}")
            ax_cont.set_xlabel("1.600 vị trí dữ liệu")
            ax_cont.set_ylabel("Điểm cộng dồn")
            st.pyplot(fig_cont)
            st.caption("Cột càng cao nghĩa là điểm dữ liệu đó đóng góp càng nhiều vào Gene này.")

        # 3. Tổng kết ra kết quả cuối cùng
        final_sum = np.sum(contributions) + biases[gene_idx]
        st.markdown(f"➡️ **Tổng điểm thu được cho Gene #{gene_idx}:** `{final_sum:.4f}`")
        if final_sum > 0:
            st.success(f"🌟 Gene #{gene_idx} được kích hoạt mạnh mẽ!")
        else:
            st.warning(f"💤 Gene #{gene_idx} hầu như không có phản ứng.")

    except Exception as e:
        st.error(f"⚠️ Lỗi giải mã: {str(e)}")

    # --- KHÔI PHỤC BIỂU ĐỒ TỔNG QUAN 256 INDEX ---
    st.markdown("### 📊 Tổng quan Bộ Mã Gene (256 Đặc trưng)")
    features_1d = deep_features.flatten()
    df_features = pd.DataFrame(features_1d, columns=["Cường độ"])
    st.bar_chart(df_features, use_container_width=True, height=200)

    st.markdown("---")
    st.info("🎯 **Bàn giao:** Toàn bộ 256 mã này sẽ được gửi sang **Bộ phân loại SVM** để đưa ra quyết định cuối cùng.")
