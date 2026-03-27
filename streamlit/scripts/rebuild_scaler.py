import os
import sys

# Đảm bảo có thể import src
current_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_dir = os.path.dirname(current_dir)
if streamlit_dir not in sys.path:
    sys.path.append(streamlit_dir)

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image
from src.data_processor import extract_hybrid_features

def rebuild_scaler(train_dir, output_path, limit_per_class=30):
    print(f"--- Rebuilding Scaler từ: {train_dir} ---")
    X_samples = []
    
    for class_id in range(43):
        class_path = os.path.join(train_dir, str(class_id))
        if not os.path.exists(class_path): continue
        
        print(f"Lấy mẫu lớp {class_id}...")
        count = 0
        try:
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(os.path.join(class_path, img_name)).convert('RGB')
                    X_samples.append(np.array(img.resize((32, 32))))
                    count += 1
                if count >= limit_per_class: break
        except: continue
                
    if not X_samples:
        print("Không tìm thấy dữ liệu!")
        return

    print(f"Tổng mẫu: {len(X_samples)}. Trích xuất đặc trưng...")
    X_features = extract_hybrid_features(np.array(X_samples))
    
    print("Fitting Scaler...")
    scaler = StandardScaler()
    scaler.fit(X_features)
    
    joblib.dump(scaler, output_path)
    print(f"Hoàn tất! Đã lưu tại: {output_path}")

if __name__ == "__main__":
    # Đường dẫn bám sát cấu trúc mới trong streamlit/
    train_path = os.path.join(streamlit_dir, 'dataset', 'Train')
    output_scaler = os.path.join(streamlit_dir, 'models', 'scaler_1812.joblib')
    
    if os.path.exists(train_path):
        rebuild_scaler(train_path, output_scaler)
    else:
        print(f"Không tìm thấy dữ liệu Train tại {train_path}")
