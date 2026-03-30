import streamlit as st
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import time

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
                                ["Dự đoán nhanh (Single Sign)", "Phát hiện & Nhận diện (Full Image)", "Quét Thư mục (Batch Mode)"])
    
    st.sidebar.markdown("---")
    
    # Tham số Detection (Chỉ hiện khi ở chế độ Full Image hoặc Batch Mode)
    det_params = {}
    if app_mode in ["Phát hiện & Nhận diện (Full Image)", "Quét Thư mục (Batch Mode)"]:
        st.sidebar.subheader("🛠️ Cấu hình Detection")
        det_params['min_s'] = st.sidebar.slider("Độ bão hòa tối thiểu (Saturation)", 0, 255, 80)
        det_params['min_v'] = st.sidebar.slider("Độ sáng tối thiểu (Value)", 0, 255, 40)
        det_params['min_size'] = st.sidebar.slider("Kích thước tối thiểu (Px)", 10, 200, 30)
        det_params['nms_thresh'] = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.3)
        
        st.sidebar.markdown("---")
        st.sidebar.write("### 📐 Tùy chỉnh Hình học (Bước 4)")
        det_params['min_ratio'], det_params['max_ratio'] = st.sidebar.slider(
            "Dải tỷ lệ khung hình (Rộng/Cao)", 
            0.1, 2.0, (0.6, 1.4), 0.1,
            help="Biển báo chuẩn thường là 1.0 (Hình vuông). Loại bỏ các vật thể quá dài hoặc quá dẹt."
        )
        det_params['min_solidity'] = st.sidebar.slider(
            "Độ đặc tối thiểu (Solidity)", 
            0.1, 1.0, 0.3, 0.05,
            help="Tỷ lệ diện tích màu trắng so với diện tích ô vuông. Biển báo kim loại thường rất 'đặc'."
        )
        
        auto_tune = st.sidebar.checkbox("Chế độ Hyper-Scan (Siêu quét 16 vòng)", value=False)
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

    # 2. Logic theo Chế độ hoạt động
    if app_mode == "Quét Thư mục (Batch Mode)":
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
                                # Vẽ khung chữ nhật
                                draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=5)
                                # Vẽ nhãn Tiếng Việt
                                label_text = f"{det['label']} ({det['conf']:.1f}%)"
                                img_pil_res = draw_vietnamese_text(img_pil_res, label_text, (x, y-35), font_size=30)
                                
                                # Hiện meta mẫu vào expander thay cho ảnh crop mờ
                                meta_path = os.path.join(current_dir, "dataset", "Meta", f"{det['id']}.png")
                                if os.path.exists(meta_path):
                                    st.image(meta_path, width=80, caption=f"Biển chuẩn #{det['id']}")
                            
                            st.image(img_pil_res, use_container_width=True)
            else:
                st.error("Đường dẫn không tồn tại!")

    else:
        # Chế độ ảnh đơn (Upload)
        uploaded_file = st.file_uploader("📥 Chọn ảnh...", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            
            if app_mode == "Dự đoán nhanh (Single Sign)":
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
                        
                        # --- THÊM ẢNH META MINH HỌA ---
                        meta_path = os.path.join(current_dir, "dataset", "Meta", f"{prediction_id}.png")
                        if os.path.exists(meta_path):
                            col_m1, col_m2, col_m3 = st.columns([1, 0.5, 1])
                            with col_m2:
                                st.image(meta_path, caption=f"Ảnh mẫu (Chuẩn)", use_container_width=True)
                        st.balloons()

            elif app_mode == "Phát hiện & Nhận diện (Full Image)":
                det_model, det_scaler = load_detection_system()
                if det_model is not None and det_scaler is not None:
                    detector = TrafficSignDetector(
                        model_path=os.path.join(current_dir, "models", "detect_model.pkl"),
                        scaler_path=os.path.join(current_dir, "models", "detect_scaler.pkl")
                    )
                    # --- TRỰC QUAN HÓA BƯỚC 1 (CLAHE) ---
                    st.write("### 📸 Bước 1: Mổ xẻ chi tiết Toán học từng lớp ảnh (CLAHE)")
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
                        
                    st.info("""
                    Bước này để xử lý các biển báo bị ngược sáng trong bóng râm, dùng CLAHE để đảm bao các biển báo tối màu sẽ giữ được đặc trung của nó
                    """)
                    st.divider()
                    
                    # --- TRỰC QUAN HÓA BƯỚC 2 (HSV FILTERING) ---
                    st.write("### 📸 Bước 2: Chuyển đổi Không gian Màu (HSV) và Cắt Rập (Masking)")
                    
                    hsv = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2HSV)
                    min_s, min_v = det_params['min_s'], det_params['min_v']
                    
                    # Tính toán mô phỏng 3 mask
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
                    
                    # Trực quan hóa Khối Quét 3 tia (Đỏ/Xanh/Vàng phát sáng trên nền đen RGB)
                    mask_colored_rgb = np.zeros_like(img_bgr)
                    mask_colored_rgb[mask_blue > 0] = [0, 150, 255]   # Xanh dương sáng
                    mask_colored_rgb[mask_yellow > 0] = [255, 255, 0] # Vàng
                    mask_colored_rgb[mask_red > 0] = [255, 0, 0]      # Đỏ chót (Vẽ chồng lên cùng để Đỏ ưu tiên hiển thị nếu đè)
                    
                    s_channel = hsv[:,:,1] # Tự tách kênh Độ Bão Hòa S
                    
                    c2_1, c2_2, c2_3, c2_4 = st.columns(4)
                    with c2_1:
                        st.image(img_clahe_pil, caption="1. Liền mạch kết quả (Từ Bước 1)", use_container_width=True)
                    with c2_2:
                        st.image(s_channel, caption="2. Chiết xuất Độ Bão Hòa (Kênh S - HSV)", use_container_width=True, clamp=True)
                    with c2_3:
                        st.image(mask_colored_rgb, caption="3. Di quét rải 3 dải màu định vị", use_container_width=True)
                    with c2_4:
                        st.image(mask_combined, caption="4. Đập khuôn Mặt nạ Tổng hợp Trắng/Đen", use_container_width=True, clamp=True)
                        
                    st.info(f"""
Loại bỏ vùng ảnh không liên quan như bầu trời, mặc đường cây cối, cần điều chỉnh S và V của từng ảnh sao cho phù hợp                     """)
                    st.divider()

                    # --- TRỰC QUAN HÓA BƯỚC 3 (MORPHOLOGY & CONTOURS) ---
                    st.write("### 📸 Bước 3: Nối hạt (Morphology) và Vẽ Phác thảo Ứng viên")
                    
                    # Mô phỏng toán học Bước 3
                    kernel_close = np.ones((9,9), np.uint8)
                    mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)
                    
                    kernel_open = np.ones((3,3), np.uint8)
                    mask_final = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
                    
                    # Vẽ khoanh vùng phác thảo (Drawing Bounding Boxes)
                    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Ảnh 3: Vẽ box trên nền Đen/Trắng (để User thấy bản chất)
                    mask_boxed = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
                    # Ảnh 4: Vẽ box trên nền Ảnh màu
                    preview_contours = img_clahe_bgr.copy()
                    
                    for cnt in contours:
                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        # Vẽ khung xanh để User thấy cách máy tính "bao vây" hòn đảo trắng
                        cv2.rectangle(mask_boxed, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.rectangle(preview_contours, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
                    
                    preview_contours_pil = Image.fromarray(cv2.cvtColor(preview_contours, cv2.COLOR_BGR2RGB))

                    c3_1, c3_2, c3_3, c3_4, c3_5 = st.columns(5)
                    with c3_1:
                        st.image(mask_combined, caption="1. Mặt nạ gốc (Từ Bước 2)", use_container_width=True, clamp=True)
                    with c3_2:
                        st.image(mask_closed, caption="2. Vá lỗ hổng (Close)", use_container_width=True, clamp=True)
                    with c3_3:
                        st.image(mask_final, caption="3. Tẩy hạt bụi (Open)", use_container_width=True, clamp=True)
                    with c3_4:
                        st.image(mask_boxed, caption="4. Bản chất khoanh Box nhị phân", use_container_width=True, clamp=True)
                    with c3_5:
                        st.image(preview_contours_pil, caption="5. Trích xuất ứng viên Màu", use_container_width=True)
                        
                    st.info("""
Biển báo thường có chữ hoặc hình vẽ màu đen/trắng ở lõi, mô hình sẽ hiểu biển báo là mảnh vỡ li ti rời rạc. Do vậy cần vá lỗ hổng. Việc xóa nhiễu (tẩy hạt bụi) để tránh mô hình dự đoán sai. Cuối cùng dùng findContours & boundingRect ở CV2 để phân vùng và xác định khung hình chữ nhật.                    """)
                    st.divider()

                    # --- TRỰC QUAN HÓA BƯỚC 4 (GEOMETRY FILTERING) ---
                    st.write("### 📸 Bước 4: Sát hạch Hình học (Geometry Filter)")
                    
                    # Mô phỏng phẫu thuật Bước 4
                    geo_passed_img = img_clahe_bgr.copy()
                    geo_rejected_img = img_clahe_bgr.copy()
                    
                    geo_passed_count = 0
                    geo_rejected_count = 0
                    
                    for cnt in contours:
                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        aspect_ratio = w_box / float(h_box)
                        area_cnt = cv2.contourArea(cnt)
                        solidity = area_cnt / float(w_box * h_box)
                        
                        # Điều kiện sát hạch
                        is_size_ok = w_box >= det_params['min_size'] and h_box >= det_params['min_size']
                        is_ratio_ok = det_params['min_ratio'] < aspect_ratio < det_params['max_ratio']
                        is_solid_ok = solidity > det_params['min_solidity']
                        
                        if is_size_ok and is_ratio_ok and is_solid_ok:
                            cv2.rectangle(geo_passed_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                            geo_passed_count += 1
                        else:
                            cv2.rectangle(geo_rejected_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                            geo_rejected_count += 1

                    c4_1, c4_2 = st.columns(2)
                    with c4_1:
                        st.image(geo_rejected_img, channels="BGR", caption=f"1. Loại bỏ {geo_rejected_count} nghi can (Sai hình dáng/kích thước)", use_container_width=True)
                    with c4_2:
                        st.image(geo_passed_img, channels="BGR", caption=f"2. {geo_passed_count} ứng viên vượt qua vòng Hình học", use_container_width=True)
                        
                    st.info(f"""
**Bước 4:** Đây là vòng "Phỏng vấn hình dáng". Biển báo phải có hình vuông, tròn hoặc tam giác đều (Tỷ lệ {det_params['min_ratio']} - {det_params['max_ratio']}) và phải là một khối đặc chứ không phải là những đường kẻ uốn lượn hay cành cây (Độ đặc > {det_params['min_solidity']*100:.0f}%). Những ai quá dài, quá dẹt hoặc rỗng tuếch sẽ bị "đánh trượt" ngay lập tức để tiết kiệm sức cho AI ở các vòng sau.
                    """)
                    st.divider()
                    
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
                                return_mask=True,
                                auto_tune=auto_tune
                            )
                            
                            elapsed_time = time.time() - start_time
                            
                            # Hiển thị Tracking Stats để giải đáp thắc mắc của anh
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
                                        # Hiển thị meta mẫu chuẩn thay cho ảnh cắt mờ
                                        meta_path = os.path.join(current_dir, "dataset", "Meta", f"{res['id']}.png")
                                        if os.path.exists(meta_path):
                                             st.image(meta_path, caption=f"Mẫu #{res['id']}", width=100)
                                        st.write(f"**{res['label']}**")
                                        st.write(f"Độ tin cậy: {res['conf']:.2f}")
                                        st.markdown(f"**`S={res['s']} | V={res['v']}`**")

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
