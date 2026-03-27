import numpy as np
import cv2
from skimage.feature import hog

# Kích thước ảnh chuẩn
IMG_SIZE = (32, 32)

def extract_hybrid_features(images, visualize=False):
    """
    Trích xuất đặc trưng lai ghép (HOG 4x4 + HSV Color Histogram 16 bins).
    Trả về vector 1812 chiều. 
    Nếu visualize=True, trả về thêm ảnh HOG để hiển thị.
    """
    features_list = []
    hog_images = []
    
    # Khởi tạo bộ lọc CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    for img in images:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)

        # --- BƯỚC 1: HOG (1764 đặc trưng) ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_clahe = clahe.apply(gray)
        
        if visualize:
            hog_feat, hog_img = hog(gray_clahe,
                                    orientations=9,
                                    pixels_per_cell=(4, 4),
                                    cells_per_block=(2, 2),
                                    block_norm='L2-Hys',
                                    visualize=True,
                                    feature_vector=True)
            hog_images.append(hog_img)
            hog_features = hog_feat
        else:
            hog_features = hog(gray_clahe,
                               orientations=9,
                               pixels_per_cell=(4, 4),
                               cells_per_block=(2, 2),
                               block_norm='L2-Hys',
                               visualize=False,
                               feature_vector=True)

        # --- BƯỚC 2: HSV Histogram (48 đặc trưng) ---
        color_features = get_hsv_histograms(img)

        # --- BƯỚC 3: Fusion ---
        hybrid_features = np.concatenate((hog_features, color_features))
        features_list.append(hybrid_features)

    if visualize:
        return np.array(features_list), hog_images[0] if hog_images else None
    
    return np.array(features_list)

def get_hsv_histograms(img):
    """Trích xuất histogram HSV riêng biệt (16 bins mỗi kênh)."""
    if img.shape[:2] != IMG_SIZE:
        img = cv2.resize(img, IMG_SIZE)
        
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])

    cv2.normalize(hist_h, hist_h, norm_type=cv2.NORM_L2)
    cv2.normalize(hist_s, hist_s, norm_type=cv2.NORM_L2)
    cv2.normalize(hist_v, hist_v, norm_type=cv2.NORM_L2)

    return np.concatenate((hist_h.flatten(), hist_s.flatten(), hist_v.flatten()))
