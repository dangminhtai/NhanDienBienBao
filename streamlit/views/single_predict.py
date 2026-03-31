import streamlit as st
import os
import tensorflow as tf
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import predict_hybrid, extract_all_fmaps

# Import các tiểu mô-đun giải phẫu (SOLID Components)
from components.anatomy.step1_pixel import render_step1_pixel_tracking
from components.anatomy.step2_cnn_overview import render_cnn_overview
from components.anatomy.step2_conv_layers import render_conv1_layer, render_relu_activation, render_conv2_layer
from components.anatomy.step2_pooling import render_pooling_layer
from components.anatomy.step2_juicer import render_dense_juicer

def render_single_predict_view(image, app_mode, cnn_extractor, rec_scaler, svm_model, class_names, current_dir):
    """Render giao diện Dự đoán nhanh (Single Sign) - SOLID Edition."""
    st.header("🔍 Chế độ Dự đoán nhanh (Single Sign)")
    
    # Khởi tạo session state để lưu kết quả (Tránh mất dữ liệu khi kéo Slider)
    if 'predict_results' not in st.session_state:
        st.session_state['predict_results'] = None

    col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
    with col_img2:
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    # Nút bấm chính
    if st.button("🔍 BẮT ĐẦU NHẬN DIỆN"):
        with st.spinner('Đang phân tích chuyên sâu (Deep Tracking)...'):
            # 1. Dự đoán Hybrid & Trích xuất Feature Maps (Cache một lần duy nhất)
            img_batch = preprocess_image_for_cnn(image)
            prediction_id, confidence = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
            fmaps_cache = extract_all_fmaps(cnn_extractor, img_batch)
            
            # 2. Lấy dữ liệu pixel
            from components.anatomy.step1_pixel import get_pixel_data
            raw_ndarray = get_pixel_data(image)
            
            # 3. Lưu vào Session State (Toàn bộ dữ liệu AI đã sẵn sàng)
            st.session_state['predict_results'] = {
                'prediction_id': prediction_id,
                'confidence': confidence,
                'img_batch': img_batch,
                'raw_ndarray': raw_ndarray,
                'fmaps_cache': fmaps_cache
            }

    # HIỂN THỊ KẾT QUẢ (Hoàn toàn Instant nhờ Cache)
    if st.session_state['predict_results'] is not None:
        res = st.session_state['predict_results']
        prediction_id = res['prediction_id']
        confidence = res['confidence']
        img_batch = res['img_batch']
        raw_ndarray = res['raw_ndarray']
        fmaps_cache = res.get('fmaps_cache')
        
        result_name = class_names.get(prediction_id, "Không xác định")

        # Hiển thị thẻ kết quả Premium
        st.markdown(f"""
            <div class="prediction-card">
                <div class="result-title">KẾT QUẢ PHÂN TÍCH HYBRID</div>
                <div class="result-name">{result_name}</div>
                <div class="label-id">NHÃN: #{prediction_id} | ĐỘ TIN CẬY: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🖼️ Giải Phẫu Trực Quan Hệ Thống")
        
        # --- BƯỚC 1: TIỀN XỬ LÝ (Tracking Pixel) ---
        render_step1_pixel_tracking(image, current_dir)

        # --- BƯỚC 2: X-QUANG CNN ---
        if fmaps_cache:
            with st.expander("📌 Bước 2: X-Quang Hệ thần kinh CNN (Deep Tracking)", expanded=True):
                # 2.0 & 2.1: Overview & Normalization
                render_cnn_overview(raw_ndarray, current_dir)
                
                # 2.1.2: Conv Layer 1 (Laboratory - Dùng fmaps1 từ Cache)
                z_val1, actual_val1 = render_conv1_layer(cnn_extractor, img_batch, raw_ndarray, fmaps_cache['fmaps1'])
                
                # 2.1.3: ReLU Activation #1 (Dùng fmaps1 từ Cache)
                render_relu_activation(z_val1, actual_val1, fmaps_cache['fmaps1'])
                
                # 2.1.4: Conv Layer 2 (Dùng fmaps1 và fmaps2 từ Cache)
                fmaps2 = render_conv2_layer(cnn_extractor, img_batch, fmaps_cache['fmaps1'], fmaps_cache['fmaps2'])
                
                # 2.1.5: MaxPooling (Dùng fmaps2 và pool1_out từ Cache)
                render_pooling_layer(cnn_extractor, img_batch, fmaps_cache['fmaps2'], fmaps_cache['pool1_out'])
                
                # 2.3: Feature Juicer (Dùng deep_features từ Cache)
                render_dense_juicer(fmaps_cache['deep_features'])
        else:
            st.warning("⚠️ Hệ thống vừa cập nhật tính năng Turbo Cache. Vui lòng nhấn nút **[🔍 BẮT ĐẦU NHẬN DIỆN]** phía trên một lần nữa để kích hoạt bộ nhớ đệm siêu tốc!")

        # --- THÊM ẢNH META MINH HỌA ---
        meta_path = os.path.join(current_dir, "dataset", "Meta", f"{prediction_id}.png")
        if os.path.exists(meta_path):
            col_m1, col_m2, col_m3 = st.columns([1, 0.5, 1])
            with col_m2:
                st.image(meta_path, caption="Ảnh mẫu chuẩn", use_container_width=True)
        
        if 'balloons_shown' not in st.session_state:
            st.balloons()
            st.session_state['balloons_shown'] = True
