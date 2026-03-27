import os
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
import joblib

# Kích thước ảnh chuẩn theo Notebook
IMG_SIZE = (32, 32)

def extract_hybrid_features(images):
    """
    Trích xuất đặc trưng lai ghép (HOG + HSV Color Histogram)
    Tương thích với mô hình 1812 chiều trong GTSRBv2.ipynb
    
    Input: images - mảng numpy/list chứa các ảnh RGB (N, 32, 32, 3)
    Output: mảng numpy chứa vector đặc trưng (N, 1812)
    """
    features_list = []
    # Khởi tạo bộ lọc CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    for img in images:
        # Đảm bảo ảnh ở định dạng uint8 và đúng kích thước
        if isinstance(img, Image.Image):
            img = np.array(img.resize(IMG_SIZE))
        
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)

        # ==========================================
        # BƯỚC 1: HOG (Hình học) - 1764 đặc trưng
        # ==========================================
        # Chuyển ảnh RGB sang xám
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Cân bằng ánh sáng cục bộ bằng CLAHE
        gray_clahe = clahe.apply(gray)

        # Trích xuất Fine-grained HOG (lưới 4x4)
        hog_features = hog(gray_clahe,
                           orientations=9,
                           pixels_per_cell=(4, 4),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys',
                           visualize=False,
                           feature_vector=True)

        # ==========================================
        # BƯỚC 2: COLOR HISTOGRAM (Màu sắc) - 48 đặc trưng
        # ==========================================
        # Chuyển ảnh RGB sang HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Tính Histogram 16 bins cho mỗi kênh H, S, V
        hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])

        # Chuẩn hóa L2 Norm
        cv2.normalize(hist_h, hist_h, norm_type=cv2.NORM_L2)
        cv2.normalize(hist_s, hist_s, norm_type=cv2.NORM_L2)
        cv2.normalize(hist_v, hist_v, norm_type=cv2.NORM_L2)

        # Nối 3 histogram thành 1 vector màu (48 chiều)
        color_features = np.concatenate((hist_h.flatten(), hist_s.flatten(), hist_v.flatten()))

        # ==========================================
        # BƯỚC 3: DUNG HỢP ĐẶC TRƯNG - 1812 chiều
        # ==========================================
        hybrid_features = np.concatenate((hog_features, color_features))
        features_list.append(hybrid_features)

    return np.array(features_list)

def preprocess_image(image_path):
    """
    Tiền xử lý một hình ảnh từ đường dẫn.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        
        # Trích xuất đặc trưng lai ghép
        features = extract_hybrid_features([img_array])
        
        return features[0]
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return None

def load_data(data_dir):
    """
    Tải toàn bộ dữ liệu từ thư mục Train/ sử dụng logic Hybrid Features.
    """
    data_list = []
    labels_list = []
    
    for class_id in range(43):
        class_path = os.path.join(data_dir, str(class_id))
        if not os.path.exists(class_path):
            continue
            
        print(f"Đang tải nhãn: {class_id}...")
        for img_name in os.listdir(class_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    data_list.append(np.array(img))
                    labels_list.append(class_id)
                except:
                    continue
                    
    # Trích xuất đặc trưng hàng loạt để tối ưu tốc độ
    print("Bắt đầu trích xuất đặc trưng Hybrid...")
    X = extract_hybrid_features(np.array(data_list))
    y = np.array(labels_list)
    
    return X, y

def get_class_names():
    """
    Từ điển ánh xạ ClassId sang tên biển báo.
    """
    return {
        0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
        9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
        12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
        16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
        19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
        22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
        25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
        29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
        35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
        38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
        41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
    }
