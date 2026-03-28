import cv2
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import random

# Cấu hình HOG từ notebook GTSDB_(Hog_+_HSV).ipynb
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'feature_vector': True
}
TARGET_SIZE = (32, 32)

DATASET_DIR = r'f:\X-FILE\Code_UNI\Python\Math for AI\CuoiKy\NhanDienBienBao\streamlit\dataset\Train'
OUTPUT_SCALER = r'f:\X-FILE\Code_UNI\Python\Math for AI\CuoiKy\NhanDienBienBao\streamlit\models\detect_scaler.pkl'

def extract_hog(img):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Trích xuất HOG
    fd = hog(gray, **HOG_PARAMS)
    return fd

def collect_positives(limit_per_class=20):
    pos_features = []
    print(f"Đang thu thập mẫu dương từ {DATASET_DIR}...")
    
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for cls in classes:
        cls_path = os.path.join(DATASET_DIR, cls)
        images = os.listdir(cls_path)[:limit_per_class]
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, TARGET_SIZE)
                pos_features.append(extract_hog(img))
    
    print(f"Thu thập được {len(pos_features)} mẫu dương.")
    return pos_features

def generate_synthetic_negatives(count=1000):
    neg_features = []
    print(f"Đang tạo {count} mẫu âm tổng hợp...")
    
    for _ in range(count):
        mode = random.choice(['noise', 'gradient', 'solid', 'random_shapes'])
        img = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)
        
        if mode == 'noise':
            img = np.random.randint(0, 256, (TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)
        elif mode == 'solid':
            color = np.random.randint(0, 256, 3)
            img[:] = color
        elif mode == 'gradient':
            c1 = np.random.randint(0, 256, 3)
            c2 = np.random.randint(0, 256, 3)
            for i in range(TARGET_SIZE[0]):
                img[i, :] = c1 * (i / TARGET_SIZE[0]) + c2 * (1 - i / TARGET_SIZE[0])
        elif mode == 'random_shapes':
            img[:] = np.random.randint(0, 256, 3)
            for _ in range(5):
                pt1 = (random.randint(0, 31), random.randint(0, 31))
                pt2 = (random.randint(0, 31), random.randint(0, 31))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.line(img, pt1, pt2, color, random.randint(1, 3))
        
        neg_features.append(extract_hog(img))
        
    return neg_features

def main():
    # 1. Thu thập dữ liệu
    pos_list = collect_positives(limit_per_class=25) # Khoảng 1000+ mẫu
    neg_list = generate_synthetic_negatives(count=len(pos_list))
    
    X = np.vstack([pos_list, neg_list])
    print(f"Tổng số mẫu huấn luyện scaler: {X.shape[0]}, Số đặc trưng: {X.shape[1]}")
    
    if X.shape[1] != 324:
        print(f"LỖI: Số đặc trưng trích xuất được là {X.shape[1]}, yêu cầu 324.")
        return

    # 2. Fit Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # 3. Lưu Scaler
    os.makedirs(os.path.dirname(OUTPUT_SCALER), exist_ok=True)
    joblib.dump(scaler, OUTPUT_SCALER)
    print(f"Đã lưu scaler thành công tại: {OUTPUT_SCALER}")

if __name__ == "__main__":
    main()
