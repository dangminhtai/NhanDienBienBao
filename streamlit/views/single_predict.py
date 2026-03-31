import streamlit as st
import os
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import predict_hybrid

# Import các tiểu mô-đun giải phẫu (SOLID Components)
from components.anatomy.step1_pixel import render_step1_pixel_tracking
from components.anatomy.step2_cnn_overview import render_cnn_overview
from components.anatomy.step2_conv_layers import render_conv1_layer, render_conv2_layer
from components.anatomy.step2_pooling import render_pooling_layer
from components.anatomy.step2_juicer import render_dense_juicer

def render_single_predict_view(image, app_mode, cnn_extractor, rec_scaler, svm_model, class_names, current_dir):
    """Render giao diện Dự đoán nhanh (Single Sign) - SOLID Edition."""
    st.header("🔍 Chế độ Dự đoán nhanh (Single Sign)")
    
    col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
    with col_img2:
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    if st.button("🔍 BẮT ĐẦU NHẬN DIỆN"):
        with st.spinner('Đang phân tích...'):
            img_batch = preprocess_image_for_cnn(image)
            prediction_id, confidence = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
            result_name = class_names.get(prediction_id, "Không xác định")

            # Hiển thị thẻ kết quả
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="result-title">KẾT QUẢ PHÂN TÍCH HYBRID</div>
                    <div class="result-name">{result_name}</div>
                    <div class="label-id">NHÃN: #{prediction_id} | ĐỘ TIN CẬY: {confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🖼️ Giải Phẫu Trực Quan Hệ Thống")
            
            # --- BƯỚC 1: TIỀN XỬ LÝ ---
            raw_ndarray = render_step1_pixel_tracking(image, current_dir)

            # --- BƯỚC 2: X-QUANG CNN ---
            with st.expander("📌 Bước 2: X-Quang Hệ thần kinh CNN (Deep Tracking)"):
                # 2.0 & 2.1: Overview & Normalization
                render_cnn_overview(raw_ndarray, current_dir)
                
                # 2.2: Conv Layer 1
                fmaps1 = render_conv1_layer(cnn_extractor, img_batch, raw_ndarray)
                
                # 2.2b: Conv Layer 2 (Deep Dive)
                fmaps2 = render_conv2_layer(cnn_extractor, img_batch, fmaps1)
                
                # 2.2c: MaxPooling
                render_pooling_layer(cnn_extractor, img_batch, fmaps2)
                
                # 2.3: Feature Juicer
                render_dense_juicer(cnn_extractor, img_batch)

            # --- THÊM ẢNH META MINH HỌA ---
            meta_path = os.path.join(current_dir, "dataset", "Meta", f"{prediction_id}.png")
            if os.path.exists(meta_path):
                col_m1, col_m2, col_m3 = st.columns([1, 0.5, 1])
                with col_m2:
                    st.image(meta_path, caption="Ảnh mẫu chuẩn", use_container_width=True)
            
            st.balloons()
