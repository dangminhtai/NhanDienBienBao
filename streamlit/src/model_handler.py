import os
import joblib

def load_model_and_scaler(base_dir):
    """
    Tải mô hình SVC và bộ chuẩn hóa từ thư mục models/.
    """
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return model, scaler

def get_prediction(model, scaler, feature_vector):
    """
    Thực hiện dự đoán từ vector đặc trưng.
    """
    if scaler is not None and scaler.n_features_in_ == feature_vector.shape[1]:
        feature_vector = scaler.transform(feature_vector)
    
    return model.predict(feature_vector)[0]
