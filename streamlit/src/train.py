import os
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Import các hàm hỗ trợ từ data_utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import load_data, extract_hog_features

def main():
    # 1. Đường dẫn dữ liệu (Dữ liệu đã được người dùng để trong streamlit/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'streamlit', 'Train')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'traffic_sign_svm.joblib')
    SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.joblib')

    print(f"Bắt đầu quy trình huấn luyện...")
    print(f"Đường dẫn dữ liệu: {TRAIN_DIR}")

    # 2. Tải và tiền xử lý dữ liệu
    # Ở đây load_data đã thực hiện preprocess_image (bao gồm extract_hog_features)
    X, y = load_data(TRAIN_DIR)
    
    print(f"Đã tải {len(X)} mẫu dữ liệu.")
    print(f"Shape của X: {X.shape}")

    # 3. Chia tập huấn luyện và kiểm chuẩn (Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Chuẩn hóa dữ liệu (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Lưu scaler để dùng cho inference sau này
    joblib.dump(scaler, SCALER_PATH)
    print(f"Đã lưu scaler vào {SCALER_PATH}")

    # 5. Tối ưu hóa siêu tham số bằng GridSearchCV
    # Lấy 10% dữ liệu để Tuning cho nhanh (giống Notebook)
    num_samples_tune = int(X_train_scaled.shape[0] * 0.1)
    X_tune, _, y_tune, _ = train_test_split(
        X_train_scaled, y_train, train_size=num_samples_tune, stratify=y_train, random_state=42
    )

    print("Đang chạy GridSearchCV để tìm tham số tốt nhất...")
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.001]}
    ]
    
    svc = SVC(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_tune, y_tune)

    print(f"Tham số tốt nhất: {grid_search.best_params_}")

    # 6. Huấn luyện mô hình cuối cùng với tham số tốt nhất
    print("Đang huấn luyện mô hình cuối cùng trên toàn bộ tập huấn luyện...")
    best_svc = grid_search.best_estimator_
    start_time = time.time()
    best_svc.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Thời gian huấn luyện: {training_time:.2f} giây")

    # 7. Đánh giá
    y_pred = best_svc.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"Độ chính xác trên tập Validation: {acc * 100:.2f}%")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_val, y_pred))

    # 8. Lưu mô hình
    joblib.dump(best_svc, MODEL_PATH)
    print(f"Đã lưu mô hình vào {MODEL_PATH}")

if __name__ == "__main__":
    main()
