import joblib
import os

model_path = r'f:\X-FILE\Code_UNI\Python\Math for AI\CuoiKy\NhanDienBienBao\models\svm_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Kernel: {model.kernel}")
    print(f"C: {model.C}")
    if hasattr(model, 'gamma'):
        print(f"Gamma: {model.gamma}")
else:
    print("Model file not found")
