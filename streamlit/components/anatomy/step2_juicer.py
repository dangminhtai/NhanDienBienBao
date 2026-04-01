import streamlit as st
import pandas as pd
import tensorflow as tf

def render_dense_juicer(deep_features):
    """Máy ép đặc trưng (1.600 -> 256)."""
    st.markdown("### 📊 2.3: Giải phẫu Máy ép Đặc trưng (1.600 ➡️ 256)")
    
    # ... (Phễu nén giữ nguyên) ...
    
    # 3. Hiển thị (Dùng Dữ liệu từ Cache)
    st.markdown("💡 **Mã Gene DNA:** 256 chỉ số định danh duy nhất.")
    
    # Đảm bảo dữ liệu là mảng 1D để hiển thị và vẽ biểu đồ
    features_1d = deep_features.flatten()
    st.code(f"# 10 'Mã Gene' đầu tiên:\n{features_1d[:10]}")
    
    import pandas as pd
    df_features = pd.DataFrame(features_1d, columns=["Cường độ"])
    st.bar_chart(df_features, use_container_width=True, height=200)
    st.success("✅ CNN đã bàn giao bộ Mã DNA cho SVM!")
