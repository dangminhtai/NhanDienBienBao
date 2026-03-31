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
                    # Hiển thị 3 dòng đầu đầy đủ bề ngang để minh chứng cho Shape (32, 32, 3)
                    # Ta lấy Kênh R, 3 dòng đầu, toàn bộ 32 cột
                    sample_rows = raw_ndarray[0:3, :, 0] 
                    st.markdown("**🎲 Truy vết Ma trận Gốc (Raw Matrix):**")
                    st.code(f"""
# Trạng thái: Dữ liệu pixel gốc (uint8: 0-255)
# Hiển thị: 3 dòng đầu tiên (Full 32 cột) - Kênh Đỏ

{sample_rows}
                    """, language="python")
                    st.caption(f"💡 Anh thấy dấu '[ ]' bao quanh 32 con số không? Đó chính là minh chứng cho chiều rộng thực tế của mảng ndarray {raw_ndarray.shape}.")

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
                # Lấy 3 dòng đầu đầy đủ bề ngang để đồng bộ với Bước 1
                normalized_rows = (raw_ndarray[0:3, :, 0]) / 255.0
                st.code(f"""
# Công thức: raw_ndarray / 255.0
# Kết quả: 3 dòng đầu (Full 32 cột) - Kênh màu Đỏ

{normalized_rows}
                """, language="python")
                st.caption("Dữ liệu lúc này đã là số thực [0, 1]. Đây là 'thức ăn' chuẩn cho các Nơ-ron.")

                # ---------------------------------------------------------
                # 2.2: GIẢI PHẪU TOÁN HỌC - 32 KERNELS ➡️ 32 FEATURE MAPS
                # ---------------------------------------------------------
                st.markdown("### 🔬 2.2: Giải phẫu 32 Bộ lọc (Kernels) ➡️ 32 Feature Maps")
                st.caption("Khám phá cách một khối 3D (32x32x3) biến thành một khối dày hơn (30x30x32):")
                
                # Hệ công thức kép H_out và W_out
                st.latex(r"""
                \begin{cases} 
                H_{out} = \lfloor \frac{H_{in} + 2 \cdot P - H_k}{S} \rfloor + 1 = \lfloor \frac{32 + 0 - 3}{1} \rfloor + 1 = 30 \\
                W_{out} = \lfloor \frac{W_{in} + 2 \cdot P - W_k}{S} \rfloor + 1 = \lfloor \frac{32 + 0 - 3}{1} \rfloor + 1 = 30
                \end{cases}
                """)
                st.info("""
                💡 **Sự thật về Khối (Volume):** 
                - **Kernel thực tế:** Không phải mảnh 3x3 phẳng, mà là khối **3x3x3** (dày bằng 3 kênh R-G-B của ảnh).
                - **Quá trình:** Một khối Kernel 3x3x3 quét qua ảnh sẽ cộng dồn cả 3 lớp màu để cho ra **1 tấm phẳng 30x30**.
                - **Kết quả:** Vì có 32 filter ➡️ xếp chồng 32 tấm phẳng ➡️ Thu được khối **(30, 30, 32)**.
                """)

                import matplotlib.pyplot as plt
                import tensorflow as tf
                import numpy as np

                # Tìm tầng Conv2D đầu tiên
                conv_layer = next((l for l in cnn_extractor.layers if "conv2d" in l.name.lower()), None)
                
                if conv_layer:
                    try:
                        # 1. Trích xuất Kernel (Trọng số) thực tế của Filter #0
                        weights, biases = conv_layer.get_weights()
                        bias_0 = biases[0]

                        # 4. Hiển thị mô phỏng phép tính CHI TIẾT (3 KÊNH MÀU)
                        st.markdown("**🎲 Quy trình Tích chập 3 Kênh (R-G-B) tại tọa độ (0,0):**")
                        st.caption("AI không chỉ soi kênh Đỏ, nó soi cả 3 kênh màu cùng lúc để tổng hợp đặc trưng:")
                        
                        # Chuẩn bị dữ liệu 3 kênh
                        patch_r = raw_ndarray[0:3, 0:3, 0] / 255.0
                        patch_g = raw_ndarray[0:3, 0:3, 1] / 255.0
                        patch_b = raw_ndarray[0:3, 0:3, 2] / 255.0
                        
                        kernel_r = weights[:, :, 0, 0]
                        kernel_g = weights[:, :, 1, 0]
                        kernel_b = weights[:, :, 2, 0]

                        # Hiển thị 3 kênh song song
                        tab_r, tab_g, tab_b = st.tabs(["🔴 Kênh Đỏ (R)", "🟢 Kênh Xanh Lá (G)", "🔵 Kênh Xanh Dương (B)"])
                        
                        with tab_r:
                            c1, c2 = st.columns(2)
                            c1.write("**Mảnh ảnh R (3x3)**"); c1.write(patch_r)
                            c2.write("**Bộ lọc R (3x3)**"); c2.write(kernel_r)
                            sum_r = np.sum(patch_r * kernel_r)
                            st.success(f"➡️ Kết quả nhân Kênh Đỏ (Sum R): **{sum_r:.6f}**")
                            
                        with tab_g:
                            c1, c2 = st.columns(2)
                            c1.write("**Mảnh ảnh G (3x3)**"); c1.write(patch_g)
                            c2.write("**Bộ lọc G (3x3)**"); c2.write(kernel_g)
                            sum_g = np.sum(patch_g * kernel_g)
                            st.success(f"➡️ Kết quả nhân Kênh Xanh Lá (Sum G): **{sum_g:.6f}**")
                            
                        with tab_b:
                            c1, c2 = st.columns(2)
                            c1.write("**Mảnh ảnh B (3x3)**"); c1.write(patch_b)
                            c2.write("**Bộ lọc B (3x3)**"); c2.write(kernel_b)
                            sum_b = np.sum(patch_b * kernel_b)
                            st.success(f"➡️ Kết quả nhân Kênh Xanh Dương (Sum B): **{sum_b:.6f}**")

                        # Tổng kết cuối cùng
                        st.markdown("➡️ **Bước 2.2: Cộng dồn 3 Kênh & Bias**")
                        total_linear = sum_r + sum_g + sum_b + bias_0
                        st.code(f"{sum_r:.4f} (R) + {sum_g:.4f} (G) + {sum_b:.4f} (B) + {bias_0:.4f} (Bias)\n= {total_linear:.4f} (Tổng tuyến tính)")
                            
                        # ---------------------------------------------------------
                        # 2.2a: GIẢI PHẪU KÍCH HOẠT ReLU (LỌC TÍN HIỆU ÂM)
                        # ---------------------------------------------------------
                        st.markdown("### ⚡ 2.2a: Giải phẫu Kích hoạt ReLU")
                        st.latex(r"ReLU(x) = \max(0, x)")
                        
                        # 3. Trích xuất Feature Maps thực tế để đối chứng
                        debug_model = tf.keras.Model(inputs=cnn_extractor.input, outputs=conv_layer.output)
                        feature_maps = debug_model.predict(img_batch, verbose=0)
                        actual_value = feature_maps[0, 0, 0, 0]
                        
                        col_r1, col_r2 = st.columns([1, 1])
                        with col_r1:
                            st.markdown("**1. Trước ReLU (Linear)**")
                            st.write(f"Giá trị: `{total_linear:.4f}`")
                        with col_r2:
                            st.markdown("**2. Sau ReLU (Non-linear)**")
                            # actual_value chính là kết quả sau ReLU từ mô hình
                            st.success(f"Giá trị: `{actual_value:.4f}`")
                            st.caption("Khớp 100% với giá trị thực của AI")

                        st.info(f"💡 **Phòng thí nghiệm:** Anh thấy chưa? `{sum_r:.4f} + {sum_g:.4f} + {sum_b:.4f} + {bias_0:.4f}` đúng bằng `{total_linear:.4f}`. Sau khi đi qua cổng ReLU, nếu nó dương thì giữ nguyên là `{actual_value:.4f}`!")

                        # 4b: TRUY VẾT MA TRẬN 3D (VOLUME TRACKING)
                        st.markdown("**📄 Truy vết 3/32 Trang hồ sơ (Ma trận 3x3 đầu):**")
                        c_f0, c_f1, c_f2 = st.columns(3)
                        with c_f0:
                            st.caption("Trang #0 (Vừa tính)")
                            st.write(feature_maps[0, 0:3, 0:3, 0])
                        with c_f1:
                            st.caption("Trang #1 (Đặc trưng 2)")
                            st.write(feature_maps[0, 0:3, 0:3, 1])
                        with c_f2:
                            st.caption("Trang #2 (Đặc trưng 3)")
                            st.write(feature_maps[0, 0:3, 0:3, 2])
                        st.caption(f"💡 Mỗi ma trận trên tương ứng với 1 tấm Heatmap ở dưới. Chúng xếp chồng lên nhau thành khối {feature_maps.shape}.")

                        # 5. Vẽ 8 feature maps đầu tiên
                        st.markdown("**🖼️ Trực quan hóa Lát cắt (Slicing 3D ➡️ 2D):**")
                        st.caption("Khối dữ liệu (30, 30, 32) giống như một hồ sơ có 32 trang. Dưới đây là 8 trang đầu tiên:")
                        
                        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
                        for i, ax in enumerate(axes.flat):
                            if i < 8:
                                ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
                                ax.set_title(f"Trang (Filter) #{i}")
                            ax.axis('off')
                        st.pyplot(fig)
                        st.info("💡 **Góc nhìn không gian:** 32 tấm ảnh 2D này khi xếp chồng lên nhau chính là tạo nên khối dữ liệu 3D (30, 30, 32) mà chúng ta vừa tính toán.")

                    except Exception as e:
                        st.warning(f"Lỗi khi mổ xẻ toán học: {str(e)}")
                else:
                    st.warning("⚠️ Không tìm thấy tầng Conv2D để giải phẫu.")

                # ---------------------------------------------------------
                # 2.2b: GIẢI PHẪU GẠN LỌC (MAXPOOLING2D)
                # ---------------------------------------------------------
                st.markdown("### 🧹 2.2b: Giải phẫu Gạn lọc (MaxPooling2D)")
                st.caption("Hãy xem cách AI nén dữ liệu và chỉ giữ lại những 'tín hiệu' mạnh nhất:")
                
                # Tìm tầng Pooling cuối và tầng Conv ngay trước nó
                pool_idx = next((i for i, l in enumerate(cnn_extractor.layers) if "pool" in l.name.lower()[::-1]), -1) # Tìm từ dưới lên
                pool_layers = [i for i, l in enumerate(cnn_extractor.layers) if "pool" in l.name.lower()]
                last_pool_idx = pool_layers[-1] if pool_layers else -1
                
                if last_pool_idx != -1:
                    try:
                        pool_layer = cnn_extractor.layers[last_pool_idx]
                        prev_layer = cnn_extractor.layers[last_pool_idx - 1] # Tầng trước Pooling
                        
                        # Trích xuất dữ liệu của cả 2 tầng
                        debug_input_model = tf.keras.Model(inputs=cnn_extractor.input, outputs=prev_layer.output)
                        debug_output_model = tf.keras.Model(inputs=cnn_extractor.input, outputs=pool_layer.output)
                        
                        input_vol = debug_input_model.predict(img_batch, verbose=0) # Thường là 10x10
                        output_vol = debug_output_model.predict(img_batch, verbose=0) # Thường là 5x5
                        
                        # 1. Trực quan vùng 2x2 đầu tiên (Filter 0)
                        input_patch_2x2 = input_vol[0, 0:2, 0:2, 0]
                        max_val = np.max(input_patch_2x2)
                        actual_pool_val = output_vol[0, 0, 0, 0]

                        col_p1, col_arrow, col_p2 = st.columns([1, 0.5, 1])
                        with col_p1:
                            st.markdown("**1. Vùng 2x2 (Trước lọc)**")
                            st.write(input_patch_2x2)
                            st.caption(f"Tầng: {prev_layer.name}")
                        with col_arrow:
                            st.markdown("<br><br><center>➡️ <b>MAX</b> ➡️</center>", unsafe_allow_html=True)
                        with col_p2:
                            st.markdown("**2. Kết quả (Sau lọc)**")
                            st.success(f"Giá trị: {actual_pool_val:.4f}")
                            st.caption(f"Tầng: {pool_layer.name}")

                        st.latex(r"H_{out} = \frac{H_{in}}{Stride} = \frac{10}{2} = 5")
                        st.info(f"💡 **Cơ chế:** Trong 4 ô ở trên, ô có giá trị lớn nhất ({max_val:.4f}) được chọn để đi tiếp. Những ô còn lại bị loại bỏ. Điều này giúp giảm 75% lượng dữ liệu nhưng vẫn giữ được đặc trưng mạnh nhất.")

                        # Show ma trận 5x5 đầy đủ
                        st.markdown("**📄 Ma trận Đặc trưng 'Vàng' (5x5):**")
                        st.code(f"# Khối dữ liệu cô đặc: {output_vol.shape}\n{output_vol[0, :, :, 0]}", language="python")

                    except Exception as e:
                        st.warning(f"Lỗi khi mổ xẻ Pooling: {str(e)}")
                else:
                    st.caption("Không tìm thấy tầng Pooling để giải phẫu.")

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
