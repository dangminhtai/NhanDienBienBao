import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Import các module nội bộ từ thư mục src/
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import load_hybrid_system, predict_hybrid, load_detection_system
from src.class_metadata import get_class_names
from src.detector import TrafficSignDetector

def load_css(file_name):
    """Nạp file CSS từ đường dẫn cục bộ."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Recognition v4.0",
    page_icon="🚦",
    layout="wide" # Chuyển sang wide để hiển thị ảnh detection tốt hơn
)

# --- NẠP CSS NGOÀI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
load_css(os.path.join(current_dir, 'assets', 'style.css'))

def main():
    # Sidebar
    st.sidebar.title("🚦 Hệ thống Nhận diện")
    
    app_mode = st.sidebar.radio("Chọn chế độ hoạt động:", 
                                ["Dự đoán nhanh (Single Sign)", "Phát hiện & Nhận diện (Full Image)"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div style="padding:10px; border:1px solid #eee; border-radius:10px">
        <b>Phiên bản:</b> 4.0 (Hybrid)<br>
        <b>Chế độ:</b> {app_mode}<br>
        <b>Kiến trúc:</b> CNN + SVM Linear<br>
        <b>Detection:</b> SVM Binary + HOG
        </div>
    """, unsafe_allow_html=True)

    st.title("🚦 Nhận diện Biển báo Giao thông")
    
    # 1. Load Recognition Models
    cnn_extractor, rec_scaler, svm_model = load_hybrid_system()
    class_names = get_class_names()

    if cnn_extractor is None or rec_scaler is None or svm_model is None:
        st.warning("⚠️ Hệ thống nhận diện đang thiếu thành phần. Vui lòng kiểm tra models/")
        return

    # 2. Giao diện Upload
    uploaded_file = st.file_uploader("📥 Chọn ảnh...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        if app_mode == "Dự đoán nhanh (Single Sign)":
            # Căn giữa ảnh bằng cột
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
                            <div class="label-id">NHÃN: #{prediction_id} | ĐỘ TIN CẬY: {confidence:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

        else: # Phát hiện & Nhận diện (Full Image)
            st.info("Chế độ Nâng cao: Hệ thống sẽ tự động tìm các biển báo trong ảnh và nhận diện từng cái.")
            
            # Load Detection Models
            det_model, det_scaler = load_detection_system()
            if det_model is not None and det_scaler is not None:
                detector = TrafficSignDetector(
                    model_path=os.path.join(current_dir, "models", "detect_model.pkl"),
                    scaler_path=os.path.join(current_dir, "models", "detect_scaler.pkl")
                )
                
                if st.button("🚀 PHÁT HIỆN & NHẬN DIỆN"):
                    with st.spinner('Đang quét biển báo...'):
                        # Chuyển PIL sang OpenCV BGR
                        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        boxes = detector.detect(img_bgr)
                        
                        if not boxes:
                            st.warning("Không tìm thấy biển báo nào trong ảnh này.")
                        else:
                            st.success(f"Tìm thấy {len(boxes)} biển báo!")
                            
                            # Vẽ và nhận diện
                            display_img = img_bgr.copy()
                            results = []
                            
                            for i, box in enumerate(boxes):
                                x, y, w, h = box
                                # Crop và nhận diện bằng Hybrid CNN-SVM
                                roi = img_bgr[y:y+h, x:x+w]
                                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                
                                img_batch = preprocess_image_for_cnn(roi_pil)
                                pred_id, conf = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
                                label = class_names.get(pred_id, "Unknown")
                                
                                results.append({
                                    "box": (x, y, w, h),
                                    "label": label,
                                    "id": pred_id,
                                    "conf": conf
                                })
                                
                                # Vẽ lên ảnh
                                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                cv2.putText(display_img, f"{label}", (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            # Hiển thị ảnh kết quả
                            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption='Kết quả phát hiện', use_container_width=True)
                            
                            # Hiển thị danh sách kết quả bên dưới
                            st.write("### Chi tiết các biển báo:")
                            cols = st.columns(min(len(results), 3))
                            for idx, res in enumerate(results):
                                with cols[idx % 3]:
                                    x, y, w, h = res["box"]
                                    st.image(image.crop((x, y, x+w, y+h)), width=100)
                                    st.write(f"**{res['label']}**")
                                    st.write(f"ID: {res['id']} | Conf: {res['conf']:.2f}")

        # --- PHẦN MINH BẠCH TOÁN HỌC ---
        st.divider()
        with st.expander("📊 CƠ SỞ TOÁN HỌC VÀ QUY TRÌNH HỆ THỐNG"):
            st.subheader("1. Kiến trúc Hybrid v4.0")
            st.write("""
            Hệ thống kết hợp **CNN** để trích xuất đặc trưng sâu (256 chiều) và **SVM** để phân loại. 
            Đối với tác vụ phát hiện, chúng tôi sử dụng lọc màu **HSV** kết hợp với **SVM Binary** dựa trên đặc trưng **HOG**.
            """)

            if app_mode == "Dự đoán nâng cao (Full Image)":
                st.markdown("""
                ```mermaid
                graph TD
                    A[Ảnh toàn cảnh] --> B[Lọc màu HSV]
                    B --> C[Morphology Cleanup]
                    C --> D[Tìm Contours]
                    D --> E[Trích xuất HOG từ các vùng ứng viên]
                    E --> F[SVM Binary Detector]
                    F --> G[Cắt các vùng là Biển báo]
                    G --> H[CNN Feature Extractor]
                    H --> I[SVM Multi-class Classifier]
                    I --> J[Kết quả Nhận diện]
                ```
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                ```mermaid
                graph LR
                    A[Ảnh biển báo] --> B[CNN Feature Extractor]
                    B --> C[SVM Classifier]
                    C --> D[Kết quả Nhận diện]
                ```
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
