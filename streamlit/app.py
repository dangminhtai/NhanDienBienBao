import streamlit as st
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

# Import các module nội bộ từ thư mục src/
from src.data_processor import preprocess_image_for_cnn
from src.model_handler import load_hybrid_system, predict_hybrid, load_detection_system
from src.class_metadata import get_class_names
from src.detector import TrafficSignDetector

def draw_vietnamese_text(image_pil, text, position, font_size=20, color=(0, 255, 0)):
    """Vẽ chữ tiếng Việt lên ảnh PIL."""
    draw = ImageDraw.Draw(image_pil)
    # Thử nạp font Arial trên Windows, nếu không dùng default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Vẽ nền cho chữ để dễ đọc
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 100))
    draw.text(position, text, font=font, fill=color)
    return image_pil

def load_css(file_name):
    """Nạp file CSS từ đường dẫn cục bộ."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Recognition v4.0",
    page_icon="🚦",
    layout="wide" 
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
    
    # Tham số Detection (Chỉ hiện khi ở chế độ Full Image)
    det_params = {}
    if app_mode == "Phát hiện & Nhận diện (Full Image)":
        st.sidebar.subheader("🛠️ Cấu hình Detetion")
        det_params['min_s'] = st.sidebar.slider("Độ bão hòa tối thiểu (Saturation)", 0, 255, 80)
        det_params['min_v'] = st.sidebar.slider("Độ sáng tối thiểu (Value)", 0, 255, 80)
        det_params['min_size'] = st.sidebar.slider("Kích thước tối thiểu (Px)", 10, 200, 30)
        auto_tune = st.sidebar.checkbox("Chế độ Hyper-Scan (Siêu quét 225 vòng)", value=True, help="Hệ thống quét mịn 225 tổ hợp màu sắc để không bỏ sót bất kỳ biển mãu nào.")
        show_debug = st.sidebar.checkbox("Hiện mặt nạ quét màu (Debug Mask)", value=False)
    
    st.sidebar.markdown(f"""
        <div style="padding:10px; border:1px solid #eee; border-radius:10px">
        <b>Phiên bản:</b> 4.0 (Hybrid)<br>
        <b>Kiến trúc:</b> CNN + SVM Linear<br>
        <b>Tiếng Việt:</b> Đã hỗ trợ
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
        image = Image.open(uploaded_file).convert("RGB")
        
        if app_mode == "Dự đoán nhanh (Single Sign)":
            # Căn giữa ảnh bằng cột
            row_col1, row_col2, row_col3 = st.columns([1, 1, 1])
            with row_col2:
                st.image(image, caption='Ảnh đã tải lên', width="stretch")
            
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
            # Load Detection Models
            det_model, det_scaler = load_detection_system()
            if det_model is not None and det_scaler is not None:
                detector = TrafficSignDetector(
                    model_path=os.path.join(current_dir, "models", "detect_model.pkl"),
                    scaler_path=os.path.join(current_dir, "models", "detect_scaler.pkl")
                )
                
                if st.button("🚀 BẮT ĐẦU QUÉT TOÀN CẢNH"):
                    import time
                    start_time = time.time()
                    with st.spinner('Đang thực hiện quét đa sắc...'):
                        # Chuyển PIL sang OpenCV BGR
                        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        boxes, mask = detector.detect(
                            img_bgr, 
                            min_s=det_params['min_s'], 
                            min_v=det_params['min_v'],
                            min_size=det_params['min_size'],
                            return_mask=True,
                            auto_tune=auto_tune
                        )
                        
                        elapsed_time = time.time() - start_time
                        
                        if show_debug:
                            st.subheader("🔍 Mặt nạ quét màu (Mask)")
                            st.image(mask, caption='Vùng màu trắng là nơi OpenCV đang tìm kiếm biển báo', width="stretch")
                        
                        if not boxes:
                            st.warning(f"Không tìm thấy biển báo nào (Thời gian quét: {elapsed_time:.2f}s). Hãy thử giảm Saturation/Value.")
                        else:
                            st.success(f"Tìm thấy {len(boxes)} biển báo trong {elapsed_time:.2f} giây!")
                            
                            # Vẽ và nhận diện
                            display_pil = image.copy()
                            results = []
                            
                            for i, box in enumerate(boxes):
                                x, y, w, h = box
                                # Crop và nhận diện bằng Hybrid CNN-SVM
                                roi = img_bgr[y:y+h, x:x+w]
                                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                
                                img_batch = preprocess_image_for_cnn(roi_pil)
                                pred_id, conf = predict_hybrid(img_batch, cnn_extractor, rec_scaler, svm_model)
                                label = class_names.get(pred_id, "Không xác định")
                                
                                results.append({
                                    "box": (x, y, w, h),
                                    "label": label,
                                    "id": pred_id,
                                    "conf": conf
                                })
                                
                                # Vẽ lên ảnh PIL bằng hàm hỗ trợ tiếng Việt
                                draw = ImageDraw.Draw(display_pil)
                                draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=3)
                                display_pil = draw_vietnamese_text(display_pil, label, (x, y-25))

                            # Hiển thị ảnh kết quả
                            st.image(display_pil, caption='Kết quả phát hiện v4.1', width="stretch")
                            
                            # Hiển thị danh sách kết quả bên dưới
                            st.write("### Chi tiết các biển báo:")
                            cols = st.columns(min(len(results), 4))
                            for idx, res in enumerate(results):
                                with cols[idx % 4]:
                                    x, y, w, h = res["box"]
                                    st.image(image.crop((x, y, x+w, y+h)), width=120)
                                    st.write(f"**{res['label']}**")
                                    st.write(f"Độ tin cậy: {res['conf']:.2f}")

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
