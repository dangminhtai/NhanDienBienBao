import streamlit as st
import os
import time
import cv2
from PIL import Image, ImageDraw
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import load_detection_system, predict_hybrid
from src.detector import TrafficSignDetector
from components.ui_helpers import draw_vietnamese_text

def render_batch_process_view(current_dir, cnn_extractor, rec_scaler, svm_model, class_names, det_params, auto_tune):
    st.header("📂 Chế độ quét hàng loạt")
    folder_path = st.text_input("Nhập đường dẫn thư mục ảnh:", value=r"f:\X-FILE\Code_UNI\Python\Math for AI\CuoiKy\NhanDienBienBao\Tests")
    
    if st.button("🚀 BẮT ĐẦU QUÉT THƯ MỤC"):
        if os.path.isdir(folder_path):
            from src.batch_processor import BatchProcessor
            
            det_model, det_scaler = load_detection_system()
            detector = TrafficSignDetector(
                model_path=os.path.join(current_dir, "models", "detect_model.pkl"),
                scaler_path=os.path.join(current_dir, "models", "detect_scaler.pkl")
            )
            
            def classifier_wrapper(roi_bgr):
                roi_pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
                img_batch = preprocess_image_for_cnn(roi_pil)
                pred_id, conf = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
                return pred_id, conf

            processor = BatchProcessor(detector, classifier_wrapper)
            
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            found_count = 0
            
            start_batch = time.time()
            for i, filename in enumerate(files):
                status_text.text(f"Đang quét ({i+1}/{len(files)}): {filename}")
                img_path = os.path.join(folder_path, filename)
                image_bgr = cv2.imread(img_path)
                if image_bgr is None: continue
                
                boxes, stats = detector.detect(image_bgr, 
                                     min_s=det_params.get('min_s', 80), 
                                     min_v=det_params.get('min_v', 80), 
                                     min_size=det_params.get('min_size', 30),
                                     nms_threshold=det_params.get('nms_thresh', 0.3),
                                     min_ratio=det_params.get('min_ratio', 0.6),
                                     max_ratio=det_params.get('max_ratio', 1.4),
                                     min_solidity=det_params.get('min_solidity', 0.3),
                                     min_laplacian=det_params.get('min_laplacian', 40),
                                     auto_tune=auto_tune)
                
                if boxes:
                    found_count += 1
                    detections = []
                    for box in boxes:
                        x, y, w, h = box
                        pred_id, conf = classifier_wrapper(image_bgr[y:y+h, x:x+w])
                        detections.append({
                            'box': box,
                            'id': pred_id,
                            'label': class_names.get(pred_id, "Unknown"),
                            'conf': conf
                        })
                    all_results.append({'filename': filename, 'path': img_path, 'detections': detections})
                
                progress_bar.progress((i + 1) / len(files))
            
            end_batch = time.time()
            st.success(f"✅ Quét xong {len(files)} ảnh trong {end_batch - start_batch:.2f} giây!")
            st.info(f"Tìm thấy biển báo trong {found_count} ảnh.")
            
            if all_results:
                st.subheader("🖼️ Kết quả chi tiết:")
                for res in all_results:
                    with st.expander(f"File: {res['filename']} - Tìm thấy {len(res['detections'])} biển báo"):
                        img_bgr_res = cv2.imread(res['path'])
                        img_pil_res = Image.fromarray(cv2.cvtColor(img_bgr_res, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil_res)
                        
                        for det in res['detections']:
                            x, y, w, h = det['box']
                            draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=5)
                            label_text = f"{det['label']} ({det['conf']:.1f}%)"
                            img_pil_res = draw_vietnamese_text(img_pil_res, label_text, (x, y-35), font_size=30)
                            
                            meta_path = os.path.join(current_dir, "dataset", "Meta", f"{det['id']}.png")
                            if os.path.exists(meta_path):
                                st.image(meta_path, width=80, caption=f"Biển chuẩn #{det['id']}")
                        
                        st.image(img_pil_res, use_container_width=True)
        else:
            st.error("Đường dẫn không tồn tại!")
