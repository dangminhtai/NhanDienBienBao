import streamlit as st
import os
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import predict_hybrid

def render_single_predict_view(image, app_mode, cnn_extractor, rec_scaler, svm_model, class_names, current_dir):
    """Render giao diện Dự đoán nhanh (Single Sign)."""
    st.header("🔍 Chế độ Dự đoán nhanh (Single Sign)")
    row_col1, row_col2, row_col3 = st.columns([1, 1, 1])
    with row_col2:
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    if st.button("🔍 BẮT ĐẦU NHẬN DIỆN"):
        with st.spinner('Đang phân tích...'):
            img_batch = preprocess_image_for_cnn(image)
            prediction_id, confidence = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
            result_name = class_names.get(prediction_id, "Không xác định")

            st.markdown(f"""
                <div class="prediction-card">
                    <div class="result-title">KẾT QUẢ PHÂN TÍCH HYBRID</div>
                    <div class="result-name">{result_name}</div>
                    <div class="label-id">NHÃN: #{prediction_id} | ĐỘ TIN CẬY: {confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # ---------------------------------------------------------
            # TRỰC QUAN HÓA HÌNH ẢNH TRONG QUÁ TRÌNH PREPROCESS (BƯỚC 1)
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("### 🖼️ Giải Phẫu Trực Quan Hệ Thống")
            
            with st.expander("📌 Bước 1: Tiền Xử Lý Hình Ảnh (32x32 Pixel & Ma Trận ndarray)", expanded=True):
                # Đọc lời giải thích từ file step1.md
                step1_path = os.path.join(current_dir, "docs", "predict", "steps", "step1.md")
                if os.path.exists(step1_path):
                    with open(step1_path, "r", encoding="utf-8") as f:
                        st.markdown(f.read())
                
                # Hiển thị trực quan việc chuyển đổi ndarray
                st.markdown("> **🔎 Góc nhìn Cỗ máy (Dữ liệu Ma trận ndarray):**")
                
                # Tạo bản sao ma trận gốc 0-255 để trực quan hóa
                import numpy as np
                from PIL import Image
                img_32x32 = image.resize((32, 32))
                raw_ndarray = np.array(img_32x32) # Ma trận gốc 0-255 (uint8)
                
                col_i1, col_i2 = st.columns([0.4, 0.6])
                with col_i1:
                    st.image(img_32x32, width=150, caption='Ảnh 32x32 (Pixel thực)') 
                    # ĐÃ GỠ BỎ LABEL NHẰM TRÁNH DATA LEAK TRONGinference
                    
                with col_i2:
                    # Trích xuất 1 phần ma trận 5x5 của kênh màu R để minh họa (y hệt Notebook)
                    sample_matrix = raw_ndarray[0:5, 0:5, 0] # Lấy Kênh R, 5 dòng 5 cột đầu
                    st.code(f"# ndarray {raw_ndarray.shape}\n# Kiểu dữ liệu: {raw_ndarray.dtype}\n{sample_matrix}", language="python")
                    st.caption("Ví dụ ma trận 5x5 (Kênh R). Các giá trị nguyên [0, 255] bám sát Notebook.")

            with st.expander("📌 Bước 2: X-Quang Hệ thần kinh CNN (Deep Tracking - Plan 2 Sâu)"):
                # Hiển thị nội dung plan2.md để dẫn dắt kỹ thuật
                plan2_path = os.path.join(current_dir, "docs", "predict", "plans", "plan2.md")
                if os.path.exists(plan2_path):
                    with open(plan2_path, "r", encoding="utf-8") as f:
                        st.markdown(f.read())
                
                # ---------------------------------------------------------
                # 2.1: TRACKING CHUẨN HÓA (NORMALIZATION)
                # ---------------------------------------------------------
                st.markdown("### 🛠️ 2.1: Phép chia Chuẩn hóa (Dữ liệu vào CNN)")
                normalized_matrix = raw_ndarray[0:5, 0:5, 0] / 255.0
                st.code(f"# Công thức: matrix / 255.0\n# Kết quả 5 dòng đầu (Kênh R):\n{normalized_matrix}", language="python")
                st.caption("Dữ liệu lúc này đã là số thực [0, 1]. Đây là 'thức ăn' chuẩn cho các Nơ-ron.")

                # ---------------------------------------------------------
                # 2.2: DEEP TRACKING - X-QUANG TẦNG CONVOLUTION
                # ---------------------------------------------------------
                st.markdown("### 🔬 2.2: X-Quang Tầng soi (Feature Maps)")
                st.caption("Dưới đây là 8 'mắt nhìn' đầu tiên của CNN. Anh sẽ thấy nó bóc tách biên giới và màu sắc biển báo:")
                
                import matplotlib.pyplot as plt
                import tensorflow as tf

                # Tạo mô hình phụ để trích xuất Feature Maps (Tầng Conv đầu tiên)
                try:
                    # Lấy layer Conv2D đầu tiên (thường là index 0 hoặc 1)
                    conv_layer = cnn_extractor.layers[0] 
                    debug_model = tf.keras.Model(inputs=cnn_extractor.input, outputs=conv_layer.output)
                    feature_maps = debug_model.predict(img_batch, verbose=0) # Shape (1, 30, 30, 32)
                    
                    # Vẽ 8 feature maps đầu tiên
                    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
                    for i, ax in enumerate(axes.flat):
                        if i < 8:
                            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
                        ax.axis('off')
                    st.pyplot(fig)
                    st.caption("8 Bản đồ nhiệt (Heatmaps) đầu tiên: Chỗ tím/xanh là tối, chỗ vàng rực là nơi CNN 'dính' đặc điểm mạnh.")
                except Exception as e:
                    st.warning(f"Không thể hiển thị Heatmap: {str(e)}")

                # ---------------------------------------------------------
                # 2.2b: TRACKING MA TRẬN TRUNG GIAN (POOLING)
                # ---------------------------------------------------------
                st.markdown("### 🧹 2.2b: Tracking Ma trận sau Gạn lọc (Pooling)")
                try:
                    # Lấy layer Pooling cuối cùng (thường là layer trước Flatten)
                    # Trong architecture của user, MaxPooling2D cuối là layer index 6
                    pool_layer = cnn_extractor.layers[6] 
                    pool_model = tf.keras.Model(inputs=cnn_extractor.input, outputs=pool_layer.output)
                    pool_output = pool_model.predict(img_batch, verbose=0) # Shape (1, 5, 5, 64)
                    
                    st.code(f"# Shape sau Pooling: {pool_output.shape}\n# Ma trận 5x5 của Lớp đặc trưng số 0:\n{pool_output[0, :, :, 0]}", language="python")
                    st.caption("Anh thấy không? Lúc này không còn là pixel màu nữa, mà là 'Cường độ đặc trưng'.")
                except:
                    st.caption("Dữ liệu nén quá sâu (5x5). Đã sẵn sàng để Trải phẳng.")

                # ---------------------------------------------------------
                # 2.3: GIẢI PHẪU PHÉP NÉN (FLATTEN TO DENSE)
                # ---------------------------------------------------------
                st.markdown("### 📊 2.3: Giải phẫu Phép nén (Flatten ➡️ 256)")
                st.caption("1.600 con số trung gian (5x5x64) được duỗi thẳng và nén về 256 Ý chính:")
                
                # Trình trích xuất đặc trưng thực tế
                deep_features = cnn_extractor.predict(img_batch, verbose=0)[0]
                
                st.code(f"# Vector định danh (256,)\n# Mẫu 10 giá trị đầu tiên:\n{deep_features[:10]}", language="python")

                import pandas as pd
                df_features = pd.DataFrame(deep_features, columns=["Giá trị Đặc trưng"])
                st.bar_chart(df_features, use_container_width=True, height=200)
                st.success(f"✅ Đã 'soi thấu' hoàn toàn Bước 2. Cỗ máy đã hiểu đây là cái gì!")

            # --- THÊM ẢNH META MINH HỌA ---
            meta_path = os.path.join(current_dir, "dataset", "Meta", f"{prediction_id}.png")
            if os.path.exists(meta_path):
                col_m1, col_m2, col_m3 = st.columns([1, 0.5, 1])
                with col_m2:
                    st.image(meta_path, caption=f"Ảnh mẫu (Chuẩn)", use_container_width=True)
            st.balloons()
