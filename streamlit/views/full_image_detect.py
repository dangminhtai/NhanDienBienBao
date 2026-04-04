import streamlit as st
import os
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import load_detection_system, predict_hybrid
from src.detector import TrafficSignDetector
from src.content_manager import get_ui

ui = get_ui()
from components.ui_helpers import draw_vietnamese_text

@st.cache_resource
def get_cached_detector(current_dir):
    return TrafficSignDetector(
        model_path=os.path.join(current_dir, "models", "detect_model.pkl"),
        scaler_path=os.path.join(current_dir, "models", "detect_scaler.pkl")
    )

def render_full_image_view(image, class_names, current_dir, det_params, auto_tune, show_debug, cnn_extractor, rec_scaler, svm_model):
    det_model, det_scaler = load_detection_system()
    if det_model is not None and det_scaler is not None:
        detector = get_cached_detector(current_dir)
        # --- TRỰC QUAN HÓA BƯỚC 1 (CLAHE) ---
        st.write(f"### {ui.get('full_image_detect.step1_title', '📸 Bước 1')}")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Phẫu thuật 1: Chuyển hệ BGR sang LAB và rút lấy kênh Độ Sáng (Lightness)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Phẫu thuật 2: CHỈ can thiệp vào kênh L cường độ sáng bằng CLAHE lưới 8x8
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Phẫu thuật 3: Bơm kênh sáng mới (cl) vào lại mảng màu (a, b) cũ
        limg = cv2.merge((cl, a, b))
        img_clahe_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        img_clahe_pil = Image.fromarray(cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(image, caption="1. Ảnh gốc (RGB)", use_container_width=True)
        with col2:
            st.image(l, caption="2. Rút kênh Sáng (L)", use_container_width=True, clamp=True)
        with col3:
            st.image(cl, caption="3. Cân bằng Histogram (CLAHE)", use_container_width=True, clamp=True)
        with col4:
            st.image(img_clahe_pil, caption="4. Trả lại mảng Màu (A,B)", use_container_width=True)

        st.write(ui.get("full_image_detect.step1_math", "#### 📐 Toán học Lõi"))
        
        # Bắt đầu vẽ Biểu đồ Histogram Real-time
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        
        # Tính toán Histogram
        hist_l = cv2.calcHist([l], [0], None, [256], [0, 256])
        hist_cl = cv2.calcHist([cl], [0], None, [256], [0, 256])
        
        # Vẽ biểu đồ kênh sáng Gốc
        ax[0].plot(hist_l, color='gray')
        ax[0].fill_between(range(256), hist_l.flatten(), color='gray', alpha=0.3)
        ax[0].set_title('Mật độ sáng (Gốc) - Phân bố hẹp', fontsize=10)
        ax[0].set_xlim([0, 256])
        
        # Vẽ biểu đồ kênh sáng CLAHE
        ax[1].plot(hist_cl, color='blue')
        ax[1].fill_between(range(256), hist_cl.flatten(), color='blue', alpha=0.3)
        ax[1].set_title('Mật độ sáng (CLAHE) - Giãn phổ & Cắt ngọn (ClipLimit)', fontsize=10)
        ax[1].set_xlim([0, 256])
        
        # Căn chỉnh để gọn gàng
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("#### 📐 Công thức nội suy (Thực tế):")
        st.latex(r"CDF(x) = \frac{\text{Số phần tử } \le x}{\text{Tổng số phần tử}}")
        st.latex(r"\text{new\_value} = \text{round}(CDF(x) \times 255)")
        
        st.info("""
**Tại sao lại có màn biến đổi này?** 
Biển báo nằm trong bóng râm sẽ có điểm ảnh bị "túm tụm" lại ở vùng tối (giống như biểu đồ Gốc bên trái, chóp nhọn nằm lệch về gần số 0). Máy móc không thể nhìn thấy chữ bên trong góc tối đó.
Bằng thuật toán CLAHE lưới 8x8, máy sẽ tính **Hàm phân bố Tích phân (CDF)** bình dân như trên để đùn dãn các giá trị pixel ra xa nhau. Lúc này chi tiết biển báo sẽ rực sáng lên mà không làm mất đi các sắc xám gốc (Nhìn vào Biểu đồ bên phải, phổ ánh sáng đã được trải dài đều từ 0 đến 255).
        """)
        st.divider()
        
        # --- TRỰC QUAN HÓA BƯỚC 2 (HSV FILTERING) ---
        st.write(f"### {ui.get('full_image_detect.step2_title', '📸 Bước 2')}")
        
        hsv = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2HSV)
        min_s, min_v = det_params['min_s'], det_params['min_v']
        
        # Đỏ
        lower_red1 = np.array([0, min_s, min_v])
        upper_red1 = np.array([15, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([155, min_s, min_v])
        upper_red2 = np.array([179, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Xanh dương
        lower_blue = np.array([100, min_s, min_v])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Vàng
        lower_yellow = np.array([17, min_s, min_v])
        upper_yellow = np.array([33, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Tổng hợp Mask Đen trắng
        mask_combined = cv2.bitwise_or(mask_red, mask_blue)
        mask_combined = cv2.bitwise_or(mask_combined, mask_yellow)
        
        mask_colored_rgb = np.zeros_like(img_bgr)
        mask_colored_rgb[mask_blue > 0] = [0, 150, 255]
        mask_colored_rgb[mask_yellow > 0] = [255, 255, 0]
        mask_colored_rgb[mask_red > 0] = [255, 0, 0]
        
        s_channel = hsv[:,:,1]
        
        c2_1, c2_2, c2_3, c2_4 = st.columns(4)
        with c2_1:
            st.image(img_clahe_pil, caption="1. Liền mạch kết quả (Từ Bước 1)", use_container_width=True)
        with c2_2:
            st.image(s_channel, caption="2. Chiết xuất Độ Bão Hòa (Kênh S)", use_container_width=True, clamp=True)
        with c2_3:
            st.image(mask_colored_rgb, caption="3. Di quét rải 3 dải màu định vị", use_container_width=True)
        with c2_4:
            st.image(mask_combined, caption="4. Đập khuôn Mặt nạ Tổng hợp Trắng/Đen", use_container_width=True, clamp=True)
            
        st.info("Loại bỏ vùng ảnh không liên quan như bầu trời, mặc đường cây cối, cần điều chỉnh S và V của từng ảnh sao cho phù hợp")
        st.divider()

        # --- TRỰC QUAN HÓA BƯỚC 3 (MORPHOLOGY & CONTOURS) ---
        st.write(f"### {ui.get('full_image_detect.step3_title', '📸 Bước 3')}")
        
        kernel_close = np.ones((9,9), np.uint8)
        mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = np.ones((3,3), np.uint8)
        mask_final = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask_boxed = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
        preview_contours = img_clahe_bgr.copy()
        
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(mask_boxed, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.rectangle(preview_contours, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
        
        preview_contours_pil = Image.fromarray(cv2.cvtColor(preview_contours, cv2.COLOR_BGR2RGB))

        c3_1, c3_2, c3_3, c3_4, c3_5 = st.columns(5)
        with c3_1:
            st.image(mask_combined, caption="1. Mặt nạ gốc", use_container_width=True, clamp=True)
        with c3_2:
            st.image(mask_closed, caption="2. Vá lỗ hổng", use_container_width=True, clamp=True)
        with c3_3:
            st.image(mask_final, caption="3. Tẩy hạt bụi", use_container_width=True, clamp=True)
        with c3_4:
            st.image(mask_boxed, caption="4. Bản chất khoanh Box", use_container_width=True, clamp=True)
        with c3_5:
            st.image(preview_contours_pil, caption="5. Trích xuất ứng viên", use_container_width=True)
            
        st.info(ui.get("full_image_detect.step3_desc", "Biển báo thường có chữ..."))
        st.divider()

        # --- TRỰC QUAN HÓA BƯỚC 4 (GEOMETRY FILTERING) ---
        st.write(f"### {ui.get('full_image_detect.step4_title', '📸 Bước 4')}")
        
        geo_passed_img = img_clahe_bgr.copy()
        geo_rejected_img = img_clahe_bgr.copy()
        
        geo_passed_count = 0
        geo_rejected_count = 0
        
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box)
            area_cnt = cv2.contourArea(cnt)
            solidity = area_cnt / float(w_box * h_box)
            
            is_size_ok = w_box >= det_params['min_size'] and h_box >= det_params['min_size']
            is_ratio_ok = det_params['min_ratio'] < aspect_ratio < det_params['max_ratio']
            is_solid_ok = solidity > det_params['min_solidity']
            
            if is_size_ok and is_ratio_ok and is_solid_ok:
                cv2.rectangle(geo_passed_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.rectangle(geo_rejected_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                geo_passed_count += 1
            else:
                cv2.rectangle(geo_rejected_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                geo_rejected_count += 1

        c4_1, c4_2 = st.columns(2)
        with c4_1:
            st.image(geo_rejected_img, channels="BGR", caption=f"1. Tổng quát {geo_passed_count + geo_rejected_count} nghi can", use_container_width=True)
        with c4_2:
            st.image(geo_passed_img, channels="BGR", caption=f"2. {geo_passed_count} ứng viên vượt qua vòng Hình học", use_container_width=True)
            
        st.info(f"Vòng 'Phỏng vấn hình dáng'. Điều kiện Tỷ lệ khắt khe ({det_params['min_ratio']} - {det_params['max_ratio']}) và Độ đặc > {det_params['min_solidity']*100:.0f}%.")
        st.divider()

        passed_rois = []
        if det_params['min_laplacian'] > 0:
            # --- TRỰC QUAN HÓA BƯỚC 5 (LAPLACIAN FOCUS FILTER) ---
            st.write(f"### {ui.get('full_image_detect.step5_title', '📸 Bước 5')}")
            
            focus_passed_img = img_clahe_bgr.copy()
            focus_rejected_img = img_clahe_bgr.copy()
            
            focus_passed_count = 0
            focus_rejected_count = 0
            rejected_rois = []
            
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = w_box / float(h_box)
                area_cnt = cv2.contourArea(cnt)
                solidity = area_cnt / float(w_box * h_box)
                
                is_size_ok = w_box >= det_params['min_size'] and h_box >= det_params['min_size']
                is_ratio_ok = det_params['min_ratio'] < aspect_ratio < det_params['max_ratio']
                is_solid_ok = solidity > det_params['min_solidity']
                
                if is_size_ok and is_ratio_ok and is_solid_ok:
                    roi = img_clahe_bgr[y:y+h_box, x:x+w_box]
                    if roi.size == 0: continue
                        
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                    
                    score_text = f"{laplacian_var:.0f}"
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    
                    if laplacian_var < det_params['min_laplacian']:
                        cv2.rectangle(focus_rejected_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                        focus_rejected_count += 1
                        if len(rejected_rois) < 5:
                            rejected_rois.append({'img': roi_rgb, 'score': score_text})
                    else:
                        cv2.rectangle(focus_passed_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.rectangle(focus_rejected_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        focus_passed_count += 1
                        passed_rois.append({'img': roi_rgb, 'score': score_text, 'bgr': roi, 'rect': (x, y, w_box, h_box)})

            c5_1, c5_2 = st.columns(2)
            with c5_1:
                st.image(focus_rejected_img, channels="BGR", caption=f"1. Khám thị lực ({focus_passed_count + focus_rejected_count} nghi can)", use_container_width=True)
            with c5_2:
                st.image(focus_passed_img, channels="BGR", caption=f"2. {focus_passed_count} Ứng viên Đạt chuẩn (Sắc nét)", use_container_width=True)
                
            if passed_rois or rejected_rois:
                st.write("#### 🔎 Soi cận cảnh: Ứng viên Mờ tịt (Trượt) vs Sắc nét (Đậu)")
                c_roi_1, c_roi_2 = st.columns(2)
                with c_roi_1:
                    if rejected_rois:
                        st.caption(f"Trượt (Điểm < {det_params['min_laplacian']})")
                        cols_r = st.columns(len(rejected_rois))
                        for i, roi_dict in enumerate(rejected_rois):
                            with cols_r[i]:
                                st.image(roi_dict['img'], caption=f"❌ {roi_dict['score']}")
                with c_roi_2:
                    if passed_rois:
                        st.caption(f"Đậu (Điểm >= {det_params['min_laplacian']})")
                        display_passed = passed_rois[:5]
                        cols_p = st.columns(len(display_passed))
                        for i, roi_dict in enumerate(display_passed):
                            with cols_p[i]:
                                st.image(roi_dict['img'], caption=f"✅ {roi_dict['score']}")                            
            st.info(f"Phép toán **Laplacian** phát hiện độ lệch mức xám. Ngưỡng hiện tại: **{det_params['min_laplacian']}**.")
            st.divider()
        else:
            st.info("💡 **Bước 5 (Sát hạch Độ nét)** đã bị TẮT. Mô hình sẽ giữ cả những biển báo bị nhòe để gửi trực tiếp cho AI.")
            st.divider()
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = w_box / float(h_box)
                solidity = cv2.contourArea(cnt) / float(w_box * h_box)
                is_size_ok = w_box >= det_params['min_size'] and h_box >= det_params['min_size']
                is_ratio_ok = det_params['min_ratio'] < aspect_ratio < det_params['max_ratio']
                is_solid_ok = solidity > det_params['min_solidity']
                if is_size_ok and is_ratio_ok and is_solid_ok:
                    roi = img_clahe_bgr[y:y+h_box, x:x+w_box]
                    if roi.size > 0:
                        passed_rois.append({'img': cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), 'score': "N/A", 'bgr': roi, 'rect': (x, y, w_box, h_box)})

        # --- TRỰC QUAN HÓA BƯỚC 6 (AI CLASSIFICATION) ---
        st.write("### 📸 Bước 6: Trí tuệ Nhân tạo thẩm định (HOG + SVM)")
        
        if passed_rois:
            svm_passed = []
            svm_rejected = []
            
            svm_passed_img = img_clahe_bgr.copy()
            svm_rejected_img = img_clahe_bgr.copy()
            
            for roi_dict in passed_rois:
                roi_bgr = roi_dict['bgr']
                roi_rgb = roi_dict['img']
                x, y, w_box, h_box = roi_dict['rect']
                
                roi_resized = cv2.resize(roi_bgr, (32, 32))
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                
                from skimage.feature import hog
                params = detector.hog_params.copy()
                params['visualize'] = True
                hog_feature, hog_image = hog(roi_gray, **params)
                
                if np.max(hog_image) > 0:
                    hog_image = hog_image / np.max(hog_image)
                hog_image_disp = (hog_image * 255).astype(np.uint8)
                
                feat_scaled = detector.scaler.transform(hog_feature.reshape(1, -1))
                prediction = detector.model.predict(feat_scaled)[0]
                confidence = detector.model.decision_function(feat_scaled)[0]
                
                score_text = f"{confidence:.1f}"
                text_y = max(y - 5, 15)
                
                if prediction == 1:
                    cv2.rectangle(svm_passed_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(svm_passed_img, score_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(svm_rejected_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(svm_rejected_img, score_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    svm_passed.append({'img': roi_rgb, 'hog_img': hog_image_disp, 'score': score_text, 'rect': (x, y, w_box, h_box), 'conf': confidence})
                else:
                    cv2.rectangle(svm_rejected_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                    cv2.putText(svm_rejected_img, score_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    svm_rejected.append({'img': roi_rgb, 'hog_img': hog_image_disp, 'score': score_text})
                    
            c6_1, c6_2 = st.columns(2)
            with c6_1:
                st.image(svm_rejected_img, channels="BGR", caption=f"1. AI Đánh rớt {len(svm_rejected)} ứng viên mạo danh (HOG hỗn loạn)", use_container_width=True)
            with c6_2:
                st.image(svm_passed_img, channels="BGR", caption=f"2. {len(svm_passed)} BIỂN BÁO CHÍNH THỨC 🎯", use_container_width=True)
                
            if svm_passed or svm_rejected:
                st.write("#### 🔎 Phòng chụp X-Quang AI (HOG Histogram): Trí tuệ Nhân tạo đang nhìn thấy gì?")
                c_ai_1, c_ai_2 = st.columns(2)
                with c_ai_1:
                    if svm_rejected:
                        st.caption(f"Trượt (Điểm Âm): Khung xương HOG lộn xộn")
                        cols_ra = st.columns(min(len(svm_rejected), 5))
                        for i, roi_dict in enumerate(svm_rejected[:5]):
                            with cols_ra[i]:
                                st.image(roi_dict['img'], use_container_width=True)
                                st.image(roi_dict['hog_img'], caption=f"❌ {roi_dict['score']}", use_container_width=True, clamp=True)
                with c_ai_2:
                    if svm_passed:
                        st.caption(f"Đạt (Điểm Dương): Khung xương HOG đỉnh cao")
                        cols_pa = st.columns(min(len(svm_passed), 5))
                        for i, roi_dict in enumerate(svm_passed[:5]):
                            with cols_pa[i]:
                                st.image(roi_dict['img'], use_container_width=True)
                                st.image(roi_dict['hog_img'], caption=f"✅ {roi_dict['score']}", use_container_width=True, clamp=True)
            
            st.info("Bước 6: Đây là bài kiểm tra cốt lõi nhất. Giám khảo Trí tuệ Nhân tạo được đào tạo bằng HOG.")
        else:
            st.warning("⚠️ Không có ứng viên nào lọt qua Bước 5.")
        
        st.divider()

        # --- TRỰC QUAN HÓA BƯỚC 7 (NON-MAXIMUM SUPPRESSION) ---
        st.write("### 🧹 Bước 7: Dọn dẹp chồng lấn (Non-Maximum Suppression - NMS)")
        
        if passed_rois and len(svm_passed) > 0:
            enable_nms_demo = st.checkbox("🛠️ Bật Mô phỏng Rác giả lập (Chuyên dùng để Thuyết trình/Dạy học NMS)", value=False, help="Bơm thêm khung rác để NMS dọn dẹp")
            
            sim_boxes = [item['rect'] for item in svm_passed]
            sim_probs = [float(item['conf']) for item in svm_passed]
            
            is_simulated = False
            if enable_nms_demo and len(sim_boxes) <= 2:
                is_simulated = True
                target_box = sim_boxes[0]
                target_prob = sim_probs[0]
                x, y, w, h = target_box
                
                for _ in range(5):
                    # import numpy already there
                    dx = np.random.randint(-15, 15)
                    dy = np.random.randint(-15, 15)
                    dw = np.random.randint(-10, 20)
                    dh = np.random.randint(-10, 20)
                    sim_boxes.append((x + dx, y + dy, w + dw, h + dh))
                    sim_probs.append(target_prob - np.random.uniform(0.1, 0.8))
            
            sorted_indices = np.argsort(sim_probs)[::-1]
            king_idx = sorted_indices[0]
            kx, ky, kw, kh = sim_boxes[king_idx]
            king_area = kw * kh
            
            table_data = []
            nms_thresh = det_params.get('nms_thresh', 0.3)
            
            for rank, idx in enumerate(sorted_indices):
                bx, by, bw, bh = sim_boxes[idx]
                prob = sim_probs[idx]
                
                if rank == 0:
                    iou = 1.0
                    status = "👑 VUA (Giữ lại)"
                else:
                    xx1 = max(kx, bx)
                    yy1 = max(ky, by)
                    xx2 = min(kx + kw, bx + bw)
                    yy2 = min(ky + kh, by + bh)
                    inter_w = max(0, xx2 - xx1)
                    inter_h = max(0, yy2 - yy1)
                    inter_area = inter_w * inter_h
                    union_area = king_area + (bw * bh) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > nms_thresh:
                        status = f"❌ XÓA (IoU {iou:.2f} > {nms_thresh})"
                    else:
                        status = "✅ GIỮ (Không đè lấp)"
                        
                table_data.append({
                    'Hạng': f"#{rank+1}",
                    'Điểm AI (Conf)': f"{prob:.2f}",
                    'Tọa độ (x,y,w,h)': f"({bx}, {by}, {bw}, {bh})",
                    'Độ đè lấp với VUA (IoU)': f"{iou:.2f}" if rank != 0 else "-",
                    'Phán quyết NMS': status
                })
                
            df_nms = pd.DataFrame(table_data)

            x, y, w, h = sim_boxes[king_idx]
            pad = 80
            cx1 = max(0, x - pad)
            cy1 = max(0, y - pad)
            cx2 = min(img_clahe_bgr.shape[1], x + w + pad)
            cy2 = min(img_clahe_bgr.shape[0], y + h + pad)
            
            zoom_before = img_clahe_bgr[cy1:cy2, cx1:cx2].copy()
            zoom_after = zoom_before.copy()
            
            final_boxes, keep_indices = detector.nms(sim_boxes, sim_probs, threshold=nms_thresh, return_indices=True)
            
            for idx, (bx, by, bw, bh) in enumerate(sim_boxes):
                nx, ny = bx - cx1, by - cy1
                score_txt = f"{sim_probs[idx]:.2f}"
                cv2.rectangle(zoom_before, (nx, ny), (nx + bw, ny + bh), (0, 255, 255), 2)
                cv2.putText(zoom_before, score_txt, (nx, max(ny-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
            for idx in keep_indices:
                bx, by, bw, bh = sim_boxes[idx]
                nx, ny = bx - cx1, by - cy1
                score_txt = f"{sim_probs[idx]:.2f}"
                cv2.rectangle(zoom_after, (nx, ny), (nx + bw, ny + bh), (0, 255, 0), 3)
                cv2.putText(zoom_after, f"VUA: {score_txt}", (nx, max(ny-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            st.write("#### 📊 Bảng thống kê thuật toán NMS So Đọ (Ai đè vùng của Vua sẽ bị chém)")
            if is_simulated:
                st.caption("*(Kích hoạt tính năng giả lập mồi rác thêm).*")
            st.dataframe(df_nms, use_container_width=True)
                
            c7_1, c7_2 = st.columns(2)
            with c7_1:
                st.image(zoom_before, channels="BGR", caption=f"1. Trước NMS ({len(sim_boxes)} khung)", use_container_width=True, clamp=True)
            with c7_2:
                st.image(zoom_after, channels="BGR", caption=f"2. Sau NMS ({len(final_boxes)} VUA)", use_container_width=True, clamp=True)
                
            st.info("Bước 7: Thuật toán NMS có nhiệm vụ dọn chồng lấp IoU.")
        else:
            st.warning("⚠️ Không có biểu báo hợp lệ để dọn dẹp.")
            
        st.divider()
        
        # --- BẮT ĐẦU QUÉT TOÀN CẢNH (Thực thi Final Detect) ---
        if st.button("🚀 BẮT ĐẦU QUÉT TOÀN CẢNH"):
            start_time = time.time()
            with st.spinner('Đang thực hiện quét đa sắc...'):
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                display_pil = image.copy()
                boxes, mask, stats = detector.detect(
                    img_bgr, 
                    min_s=det_params['min_s'], 
                    min_v=det_params['min_v'], 
                    min_size=det_params['min_size'],
                    nms_threshold=det_params['nms_thresh'],
                    min_ratio=det_params['min_ratio'],
                    max_ratio=det_params['max_ratio'],
                    min_solidity=det_params['min_solidity'],
                    min_laplacian=det_params['min_laplacian'],
                    return_mask=True,
                    auto_tune=auto_tune
                )
                
                elapsed_time = time.time() - start_time
                
                st.write("### 📊 Thống kê bộ lọc (Tracking Stats):")
                st.code(f"""
- Tổng vùng màu tìm thấy (HSV): {stats['hsv_cnt']}
- Vượt qua vòng Hình học (Geo): {stats['geo_pass']}
- Vượt qua vòng Focus/Laplacian: {stats['filt_pass']}
- Được SVM xác nhận là Biển báo: {stats['svm_pass']}
- Giữ lại cuối cùng sau NMS: {stats['nms_kept']}
                """)

                if not auto_tune and show_debug:
                    st.subheader("🔍 Mặt nạ quét màu (Mask)")
                    st.image(mask, caption='Vùng màu trắng là nơi OpenCV đang tìm kiếm biển báo', use_container_width=True)
                
                if not boxes:
                    st.warning(f"Không tìm thấy biển báo nào (Thời gian quét: {elapsed_time:.2f}s). Hãy thử giảm Saturation/Value.")
                else:
                    st.success(f"Tìm thấy {len(boxes)} biển báo trong {elapsed_time:.2f} giây!")
                    
                    results = []
                    sv_params = stats.get('sv_params', [])
                    
                    for i, box in enumerate(boxes):
                        x, y, w, h = box
                        roi = img_bgr[y:y+h, x:x+w]
                        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        
                        img_batch = preprocess_image_for_cnn(roi_pil)
                        pred_id, conf = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
                        label = class_names.get(pred_id, "Không xác định")
                        
                        s_val, v_val = sv_params[i] if i < len(sv_params) else ("?", "?")
                        
                        results.append({
                            "box": (x, y, w, h),
                            "label": label,
                            "id": pred_id,
                            "conf": conf,
                            "s": s_val,
                            "v": v_val
                        })
                        
                        draw = ImageDraw.Draw(display_pil)
                        draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=3)
                        display_label = f"{label} (S:{s_val}, V:{v_val})"
                        display_pil = draw_vietnamese_text(display_pil, display_label, (x, y-25))

                    st.image(display_pil, caption='Kết quả phát hiện v4.2', use_container_width=True)
                    
                    st.write("### Chi tiết các biển báo:")
                    cols = st.columns(min(len(results), 4))
                    for idx, res in enumerate(results):
                        with cols[idx % 4]:
                            meta_path = os.path.join(current_dir, "dataset", "Meta", f"{res['id']}.png")
                            if os.path.exists(meta_path):
                                 st.image(meta_path, caption=f"Mẫu #{res['id']}", width=100)
                            st.write(f"**{res['label']}**")
                            st.write(f"Độ tin cậy: {res['conf']:.2f}")
                            st.markdown(f"**`S={res['s']} | V={res['v']}`**")
