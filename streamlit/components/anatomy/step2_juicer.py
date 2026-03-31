import streamlit as st
import pandas as pd
import tensorflow as tf

def render_dense_juicer(cnn_extractor, img_batch):
    """Máy ép đặc trưng (1.600 -> 256)."""
    st.markdown("### 📊 2.3: Giải phẫu Máy ép Đặc trưng (1.600 ➡️ 256)")
    
    # 1. Tóm tắt Khối Conv 2
    st.markdown("""
    <div style="background: #fff3e0; padding: 10px; border-left: 5px solid #ff9800; border-radius: 5px;">
        <b>🔄 Khối Conv 2:</b> Dữ liệu tiếp tục đi sâu để còn lại <b>5x5x64</b> (1.600 số).
    </div>
    <br/>
    """, unsafe_allow_html=True)
    
    # 2. Phễu nén HTML
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="display: inline-block; width: 140px; height: 40px; background: #90caf9; border-radius: 5px; line-height: 40px; border: 2px solid #1e88e5;"><b>1.600 Số thô</b></div>
        <div style="font-size: 30px; color: #1e88e5;">🔽</div>
        <div style="display: flex; justify-content: center; align-items: center;">
             <div style="width: 0; height: 0; border-left: 80px solid transparent; border-right: 80px solid transparent; border-top: 100px solid #bbdefb; position: relative;">
                <div style="position: absolute; top: -80px; left: -40px; width: 80px; text-align: center; color: #0d47a1; font-weight: bold;">PHẾU NÉN</div>
             </div>
        </div>
        <div style="font-size: 30px; color: #4caf50;">🔽</div>
        <div style="display: inline-block; width: 140px; height: 50px; background: #a5d6a7; border-radius: 25px; line-height: 50px; border: 2px solid #388e3c; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"><b>🧬 256 MÃ GENE</b></div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Trích xuất & Hiển thị
    deep_features = cnn_extractor.predict(img_batch, verbose=0)[0]
    st.markdown("💡 **Mã Gene DNA:** 256 chỉ số định danh duy nhất.")
    st.code(f"# 10 'Mã Gene' đầu tiên:\n{deep_features[:10]}")
    
    df_features = pd.DataFrame(deep_features, columns=["Cường độ"])
    st.bar_chart(df_features, use_container_width=True, height=200)
    st.success("✅ CNN đã bàn giao bộ Mã DNA cho SVM!")
