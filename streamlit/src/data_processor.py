import cv2
import numpy as np
from PIL import Image

def preprocess_image_for_cnn(image):
    """
    Tiền xử lý ảnh theo đúng quy trình của CNN trong notebook:
    1. Resize về (32, 32)
    2. Chuyển sang mảng numpy và chuẩn hóa về dải [0, 1]
    3. Thêm chiều batch (1, 32, 32, 3)
    """
    # Đảm bảo ảnh ở định dạng RGB
    if isinstance(image, Image.Image):
        img_rgb = image.convert('RGB')
    else:
        # Nếu là mảng numpy (từ cv2), chuyển sang RGB nếu cần
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = Image.fromarray(img_rgb)
    
    # 1. Resize
    img_resized = img_rgb.resize((32, 32))
    
    # 2. Chuyển sang numpy và chuẩn hóa
    img_array = np.array(img_resized).astype('float32') / 255.0
    
    # 3. Thêm chiều batch (1, 32, 32, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch
