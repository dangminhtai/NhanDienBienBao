import streamlit as st
import os
import joblib
import numpy as np
@st.cache_resource
def load_model_and_scaler(base_dir):
    """
    Tải mô hình SVC và bộ chuẩn hóa từ thư mục models/.
    Sử dụng cache của Streamlit để tránh load lại nhiều lần.
    """
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return model, scaler

def get_prediction(model, scaler, feature_vector):
    """
    Thực hiện dự đoán từ vector đặc trưng.
    Trả về (prediction_id, confidence_score).
    """
    if scaler is not None and scaler.n_features_in_ == feature_vector.shape[1]:
        feature_vector = scaler.transform(feature_vector)
    
    prediction = model.predict(feature_vector)[0]
    
    # Lấy độ tin cậy nếu model hỗ trợ (hoặc dùng distance tới hyperplane)
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(feature_vector)
            confidence = np.max(probs)
        except:
            pass
    elif hasattr(model, 'decision_function'):
        # Với SVM, giá trị decision_function càng xa 0 càng tin cậy
        dists = model.decision_function(feature_vector)
        # Chuẩn hóa đơn giản (sigmoid-like) để hiển thị %
        confidence = 1 / (1 + np.exp(-np.max(np.abs(dists)) / 10)) 

    return prediction, confidence
