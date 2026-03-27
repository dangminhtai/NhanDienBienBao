import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image
import sys

# Thêm thư mục hiện tại vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import extract_hybrid_features

def rebuild_scaler(train_dir, output_path, limit_per_class=20):
    """
    Tạo lại Scaler từ tập dữ liệu huấn luyện cục bộ.
    """
    print(f"Bắt đầu rebuild scaler từ: {train_dir}")
    X_samples = []
    
    # Duyệt qua các class
    for class_id in range(43):
        class_path = os.path.join(train_dir, str(class_id))
        if not os.path.exists(class_path):
            continue
            
        print(f"Lấy mẫu từ nhãn {class_id}...")
        count = 0
        try:
            img_names = os.listdir(class_path)
            # Lấy tối đa limit_per_class ảnh để tính toán scaler nhanh
            for img_name in img_names:
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32))
                        X_samples.append(np.array(img))
                        count += 1
                    except:
                        continue
                if count >= limit_per_class:
                    break
        except:
            continue
                
    if not X_samples:
        print(f"Không tìm thấy dữ liệu ảnh để rebuild scaler tại {train_dir}!")
        return
        
    print(f"Tổng số mẫu thu thập: {len(X_samples)}")
    print("Đang trích xuất đặc trưng Hybrid (HOG 4x4 + HSV)...")
    X_features = extract_hybrid_features(np.array(X_samples))
    
    print(f"Shape đặc trưng: {X_features.shape}")
    
    print("Đang fit StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_features)
    
    # Lưu scaler
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, output_path)
    print(f"Đã lưu scaler mới tại: {output_path}")
    print(f"Số lượng đặc trưng chuẩn hóa: {scaler.n_features_in_}")

if __name__ == "__main__":
    # Đường dẫn mới: streamlit/dataset/Train/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'streamlit', 'dataset', 'Train')
    
    if not os.path.exists(train_path):
        # Fallback về đường dẫn cũ nếu cần
        train_path = os.path.join(base_dir, 'streamlit', 'Train')
        
    output_scaler = os.path.join(base_dir, 'streamlit', 'models', 'scaler_1812.joblib')
    
    if os.path.exists(train_path):
        rebuild_scaler(train_path, output_scaler, limit_per_class=30)
    else:
        print(f"CẢNH BÁO: Không tìm thấy thư mục Train tại {train_path}. Không thể rebuild scaler.")
