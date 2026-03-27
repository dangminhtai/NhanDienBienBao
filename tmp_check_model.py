import joblib
import os

model_path = r'f:\X-FILE\Code_UNI\Python\Math for AI\CuoiKy\NhanDienBienBao\streamlit\models\model.pkl'
scaler_path = r'f:\X-FILE\Code_UNI\Python\Math for AI\CuoiKy\NhanDienBienBao\streamlit\models\scaler.joblib'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    if hasattr(model, 'kernel'):
        print(f"Kernel: {model.kernel}")
    if hasattr(model, 'C'):
        print(f"C: {model.C}")
    print(f"Number of classes: {len(model.classes_)}")
    print(f"Features expected: {model.n_features_in_}")

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"Scaler type: {type(scaler)}")
    print(f"Number of features in scaler: {scaler.n_features_in_}")
