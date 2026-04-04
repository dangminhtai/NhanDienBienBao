import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import load_detection_system, predict_hybrid
from src.detector import TrafficSignDetector
from components.ui_helpers import draw_vietnamese_text

def render_video_detect_view(cnn_extractor, rec_scaler, svm_model, class_names, current_dir, det_params):
    st.header("📽️ Phát hiện Biển báo từ Video (Beta)")
    
    st.info("""
    **Hướng dẫn:** 
    1. Chọn video từ thư mục có sẵn hoặc tải lên video mới (.mp4, .avi).
    2. Điều chỉnh tham số **Frame Skip** (bỏ qua khung hình) để tăng tốc độ xử lý.
    3. Nhấn **Bắt đầu xử lý** để xem kết quả real-time.
    
    *Lưu ý: Chế độ Video đang ở bản Beta. Do xử lý AI từng khung hình trên web nên tốc độ sẽ chậm hơn video gốc. Hãy tăng Frame Skip và giảm Resize Scale để mượt hơn.*
    """)

    # --- Sidebar bổ sung cho Video ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Cấu hình Video")
    frame_skip = st.sidebar.slider("Bỏ qua khung hình (Frame Skip)", 1, 30, 5, help="Ví dụ: chọn 5 nghĩa là chỉ xử lý 1 khung hình sau mỗi 5 khung hình video.")
    resize_factor = st.sidebar.slider("Tỷ lệ thu nhỏ (Resize Scale)", 0.2, 1.0, 0.5, 0.1, help="Thu nhỏ khung hình giúp xử lý nhanh hơn.")
    process_limit = st.sidebar.number_input("Giới hạn thời gian xử lý (giây)", 10, 600, 60)

    # Thư mục video có sẵn
    video_dir = os.path.join(current_dir, "videos")
    available_videos = []
    if os.path.exists(video_dir):
        available_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Chọn nguồn video
    source_option = st.radio("Chọn nguồn video:", ["Video có sẵn trong thư mục", "Tải lên video mới (.mp4)"], horizontal=True)

    video_path = None
    if source_option == "Video có sẵn trong thư mục":
        if available_videos:
            selected_video = st.selectbox("Chọn video:", available_videos)
            video_path = os.path.join(video_dir, selected_video)
        else:
            st.warning("Không tìm thấy video nào trong thư mục `streamlit/videos`.")
    else:
        uploaded_video = st.file_uploader("Tải lên video...", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

    if video_path:
        # Load detector
        det_model, det_scaler = load_detection_system()
        detector = TrafficSignDetector(
            model_path=os.path.join(current_dir, "models", "detect_model.pkl"),
            scaler_path=os.path.join(current_dir, "models", "detect_scaler.pkl")
        )

        col_v1, col_v2 = st.columns([2, 1])
        
        with col_v1:
            st.subheader("🎥 Luồng video xử lý")
            frame_placeholder = st.empty()
            
        with col_v2:
            st.subheader("📈 Trạng thái")
            status_text = st.empty()
            progress_bar = st.progress(0)
            log_placeholder = st.empty()

        if st.button("🚀 BẮT ĐẦU XỬ LÝ VIDEO", type="primary"):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Không thể mở file video.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            status_text.write(f"Đang xử lý: **{os.path.basename(video_path)}**")
            log_placeholder.info(f"Tổng khung hình: {total_frames} | FPS: {fps:.1f}")

            count = 0
            processed_count = 0
            start_process_time = time.time()
            found_signs = []

            # Nút dừng
            stop_btn = st.button("🛑 DỪNG LẠI")

            try:
                while cap.isOpened():
                    if stop_btn:
                        st.warning("Đã dừng xử lý theo yêu cầu người dùng.")
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Theo dõi thời gian xử lý
                    elapsed_total = time.time() - start_process_time
                    if elapsed_total > process_limit:
                        st.warning(f"Đã chạm giới hạn thời gian xử lý ({process_limit}s).")
                        break

                    # Chỉ xử lý các khung hình theo frame_skip
                    if count % frame_skip == 0:
                        processed_count += 1
                        
                        # Resize khung hình để tăng tốc
                        if resize_factor < 1.0:
                            h, w = frame.shape[:2]
                            frame_small = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))
                        else:
                            frame_small = frame.copy()

                        # Đổi màu về RGB cho PIL
                        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                        display_pil = Image.fromarray(frame_rgb)
                        
                        # Gọi Detector
                        boxes, stats = detector.detect(
                            frame_small, 
                            min_s=det_params['min_s'], 
                            min_v=det_params['min_v'], 
                            min_size=int(det_params['min_size'] * resize_factor),
                            nms_threshold=det_params['nms_thresh'],
                            min_ratio=det_params['min_ratio'],
                            max_ratio=det_params['max_ratio'],
                            min_solidity=det_params['min_solidity'],
                            min_laplacian=det_params['min_laplacian'],
                            auto_tune=False # Không dùng auto-tune cho video vì quá chậm
                        )

                        if boxes:
                            draw = ImageDraw.Draw(display_pil)
                            for box in boxes:
                                bx, by, bw, bh = box
                                roi = frame_small[by:by+bh, bx:bx+bw]
                                if roi.size == 0: continue
                                
                                # Nhận diện biển báo (Hybrid: CNN + SVM)
                                try:
                                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                    img_batch = preprocess_image_for_cnn(roi_pil)
                                    pred_id, conf = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
                                    label = class_names.get(pred_id, "Unknown")
                                    
                                    # Vẽ
                                    draw.rectangle([bx, by, bx+bw, by+bh], outline=(0, 255, 0), width=3)
                                    display_pil = draw_vietnamese_text(display_pil, f"{label} ({conf:.1f}%)", (bx, by-25))
                                    
                                    if label not in [s['label'] for s in found_signs]:
                                        found_signs.append({
                                            "label": label, 
                                            "id": pred_id, 
                                            "conf": conf, 
                                            "frame": count,
                                            "roi": roi_pil # Lưu lại ảnh đã cắt để hậu kiểm
                                        })
                                except:
                                    continue

                        # Cập nhật GUI
                        frame_placeholder.image(display_pil, use_container_width=True)
                        
                        # Cập nhật tiến độ
                        progress = min(count / total_frames, 1.0) if total_frames > 0 else 0
                        progress_bar.progress(progress)
                        
                    count += 1
                
                cap.release()
                st.success(f"Hoàn thành xử lý video! Tổng khung hình đã quét: {processed_count}")

                if found_signs:
                    st.subheader("📋 Các biển báo đã phát hiện:")
                    # Sử dụng 2 cột lớn để hiển thị thẻ so sánh (Comparison Cards) nhằm giúp anh quan sát rõ hơn
                    cols = st.columns(2)
                    for i, sign in enumerate(found_signs):
                        with cols[i % 2]:
                            with st.container(border=True):
                                st.markdown(f"#### 📍 {sign['label']}")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.image(sign['roi'], caption="🖼️ Ảnh cắt từ Video", use_container_width=True)
                                with col_b:
                                    meta_path = os.path.join(current_dir, "dataset", "Meta", f"{sign['id']}.png")
                                    if os.path.exists(meta_path):
                                         st.image(meta_path, caption="📘 Mẫu từ Dataset", use_container_width=True)
                                    else:
                                         st.warning("Không tìm thấy mẫu")
                                
                                st.caption(f"**Khung hình:** {sign['frame']} | **Độ tin cậy:** {sign['conf']:.1f}%")
                else:
                    st.info("Không phát hiện được biển báo nào trong video này với tham số hiện tại.")

            except Exception as e:
                st.error(f"Lỗi trong quá trình xử lý: {str(e)}")
                if cap: cap.release()
