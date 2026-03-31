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

            with st.expander("📌 Bước 2: Truy vết Biến đổi Kỹ thuật (CNN Tracking - Plan 2)"):
                # Hiển thị nội dung plan2.md để dẫn dắt kỹ thuật
                plan2_path = os.path.join(current_dir, "docs", "predict", "plans", "plan2.md")
                if os.path.exists(plan2_path):
                    with open(plan2_path, "r", encoding="utf-8") as f:
                        st.markdown(f.read())
                
                # ---------------------------------------------------------
                # 2.1: TRACKING CHUẨN HÓA (NORMALIZATION)
                # ---------------------------------------------------------
                st.markdown("### 🛠️ 2.1: Phép chia Chuẩn hóa (0-255 ➡️ 0-1)")
                normalized_matrix = raw_ndarray[0:5, 0:5, 0] / 255.0
                st.code(f"# Công thức: matrix / 255.0\n# Kết quả mẫu (5x5 Kênh R):\n{normalized_matrix}", language="python")
                st.caption("Dữ liệu đã được đưa về đoạn [0, 1] để Keras xử lý tối ưu hơn.")

                # ---------------------------------------------------------
                # 2.2: TRACKING SHAPE CNN LAYERS
                # ---------------------------------------------------------
                st.markdown("### 🔬 2.2: Dây chuyền Tích chập (CNN Layers Tracking)")
                
                # Bảng tracking bám sát code .py
                tracking_data = [
                    {"Hàm (Function)": "Input Data", "Shape": "(32, 32, 3)", "Dữ liệu": "3072 pixel rỗ hột."},
                    {"Hàm (Function)": "Conv 1 & 2 (Filters: 64)", "Shape": "(28, 28, 64)", "Dữ liệu": "Quét đặc trưng biên/màu."},
                    {"Hàm (Function)": "MaxPooling 1 (2x2)", "Shape": "(14, 14, 64)", "Dữ liệu": "Gạn lọc thông tin (Nén)."},
                    {"Hàm (Function)": "Conv 3 & 4 (Filters: 64)", "Shape": "(10, 10, 64)", "Dữ liệu": "Quét đặc trưng sâu."},
                    {"Hàm (Function)": "MaxPooling 2 (2x2)", "Shape": "(5, 5, 64)", "Dữ liệu": "Cô đặc tối đa."},
                ]
                st.table(tracking_data)
                
                # ---------------------------------------------------------
                # 2.3: TRACKING KẾT QUẢ VECTOR (RESULT)
                # ---------------------------------------------------------
                st.markdown("### 📊 2.3: Chốt hạ Đặc trưng (Dense 256 Tracking)")
                
                # Trình trích xuất đặc trưng thực tế
                deep_features = cnn_extractor.predict(img_batch, verbose=0)[0]
                
                st.code(f"# Shape cuối: (256,)\n# Mẫu 10 giá trị đầu tiên:\n{deep_features[:10]}", language="python")

                import pandas as pd
                df_features = pd.DataFrame(deep_features, columns=["Giá trị Đặc trưng"])
                st.bar_chart(df_features, use_container_width=True, height=200)
                st.success(f"✅ Hoàn tất mổ xẻ Bước 2: Từ 3072 pixel ➡️ {len(deep_features)} đặc trưng tinh túy.")

            # --- THÊM ẢNH META MINH HỌA ---
            meta_path = os.path.join(current_dir, "dataset", "Meta", f"{prediction_id}.png")
            if os.path.exists(meta_path):
                col_m1, col_m2, col_m3 = st.columns([1, 0.5, 1])
                with col_m2:
                    st.image(meta_path, caption=f"Ảnh mẫu (Chuẩn)", use_container_width=True)
            st.balloons()
