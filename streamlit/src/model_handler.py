import joblib
import os
import streamlit as st

# Kiểm tra xem tensorflow có sẵn không để hiển thị lỗi thân thiện
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Đường dẫn mô hình v4.0 (Hybrid)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "final_hybrid_svm.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "svm_scaler.pkl")
CNN_PATH = os.path.join(MODELS_DIR, "cnn_feature_extractor.h5")

# Đường dẫn mô hình Detection (v4.0+)
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "detect_model.pkl")
DETECTION_SCALER_PATH = os.path.join(MODELS_DIR, "detect_scaler.pkl")

@st.cache_resource
def load_hybrid_system():
    """
    Tải hệ thống lai ghép gồm 3 thành phần:
    1. CNN Feature Extractor (.h5)
    2. Scaler (.pkl)
    3. SVM Classifier (.pkl)
    """
    if not HAS_TENSORFLOW:
        st.error("❌ Thư viện 'tensorflow' chưa được cài đặt. Vui lòng cài đặt để chạy mô hình v4.0.")
        return None, None, None

    try:
        # 1. Tải CNN Feature Extractor
        if not os.path.exists(CNN_PATH):
            st.error(f"❌ Không tìm thấy bộ trích xuất CNN tại: {CNN_PATH}")
            return None, None, None
        cnn_extractor = load_model(CNN_PATH)

        # 2. Tải Scaler
        if not os.path.exists(SCALER_PATH):
            st.error(f"❌ Không tìm thấy bộ chuẩn hóa tại: {SCALER_PATH}")
            return None, None, None
        scaler = joblib.load(SCALER_PATH)

        # 3. Tải SVM Model
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Không tìm thấy mô hình SVM tại: {MODEL_PATH}")
            return None, None, None
        svm_model = joblib.load(MODEL_PATH)

        return cnn_extractor, scaler, svm_model
    except Exception as e:
        st.error(f"❌ Lỗi khi tải hệ thống mô hình: {str(e)}")
        return None, None, None

@st.cache_resource
def load_detection_system():
    """
    Tải hệ thống phát hiện biển báo (Detection):
    1. SVM Binary Classifier (.pkl)
    2. HOG Scaler (.pkl)
    """
    try:
        if not os.path.exists(DETECTION_MODEL_PATH):
            st.error(f"❌ Không tìm thấy mô hình Detection tại: {DETECTION_MODEL_PATH}")
            return None, None
            
        if not os.path.exists(DETECTION_SCALER_PATH):
            st.error(f"❌ Không tìm thấy Scaler cho Detection tại: {DETECTION_SCALER_PATH}")
            return None, None
            
        model = joblib.load(DETECTION_MODEL_PATH)
        scaler = joblib.load(DETECTION_SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Lỗi khi tải hệ thống Detection: {str(e)}")
        return None, None

def predict_hybrid(image_batch, cnn_extractor, scaler, svm_model):
    """
    Quy trình dự đoán lai ghép:
    1. Trích xuất đặc trưng sâu từ CNN (256 dims)
    2. Chuẩn hóa đặc trưng bằng Scaler
    3. Phân loại bằng SVM
    """
    # 1. Trích xuất đặc trưng (CNN)
    # Output của cnn_extractor.predict là (1, 256)
    deep_features = cnn_extractor.predict(image_batch, verbose=0)
    
    # 2. Chuẩn hóa (Scaler)
    scaled_features = scaler.transform(deep_features)
    
    # 3. Dự đoán (SVM)
    prediction = svm_model.predict(scaled_features)[0]
    
    # 4. Tính toán độ tin cậy (Decision Function)
    # SVM Linear trả về khoảng cách tới siêu mặt phẳng
    decision_scores = svm_model.decision_function(scaled_features)[0]
    confidence = decision_scores[prediction]
    
    return prediction, confidence
