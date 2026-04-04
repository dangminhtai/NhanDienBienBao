import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_block2_quickview(fmaps3, fmaps4, pool2_out):
    """Trực quan nhanh Khối Conv 2 (Trung cấp - 14x14 -> 5x5)."""
    st.markdown("---")
    st.header("🧱 2.2: KHỐI CONV 2 (Trung cấp - Soi chi tiết)")
    
    st.info("""
    💡 **Tại sao gọi là Trung cấp?** Ở tầng này, các bản đồ đặc trưng đã bị thu nhỏ đáng kể (từ 14x14 về 5x5). 
    AI không còn nhìn vào từng pixel lẻ tẻ mà bắt đầu nhận diện các **cụm kết cấu** và **tổ hợp hình khối** phức tạp hơn của biển báo.
    """)

    # Tạo 3 cột để hiển thị nhanh 3 giai đoạn của Khối 2
    col_c3, col_c4, col_p2 = st.columns(3)
    
    # Lấy ngẫu nhiên hoặc filter đầu tiên để demo
    f_idx = 0
    
    with col_c3:
        st.markdown("**🔍 2.2.1: Conv2D_3**")
        st.caption("(Kích thước: 12x12)")
        fig3, ax3 = plt.subplots()
        ax3.imshow(fmaps3[0, :, :, f_idx], cmap='viridis')
        ax3.axis('off')
        st.pyplot(fig3)
        st.markdown("Lọc ra các tổ hợp cạnh cấp 2.")

    with col_c4:
        st.markdown("**🔍 2.2.3: Conv2D_4**")
        st.caption("(Kích thước: 10x10)")
        fig4, ax4 = plt.subplots()
        ax4.imshow(fmaps4[0, :, :, f_idx], cmap='viridis')
        ax4.axis('off')
        st.pyplot(fig4)
        st.markdown("Tăng cường các vùng đặc trưng.")

    with col_p2:
        st.markdown("**🧹 2.2.5: Pool_2**")
        st.caption("(Kích thước: 5x5)")
        figp2, axp2 = plt.subplots()
        axp2.imshow(pool2_out[0, :, :, f_idx], cmap='viridis')
        axp2.axis('off')
        st.pyplot(figp2)
        st.markdown("Cô đặc về 25 điểm dữ liệu.")

    st.success("🎯 **Kết quả:** Sau Khối 2, dữ liệu đã cực kỳ cô đọng, sẵn sàng để 'vắt nước' thành vector định danh.")
