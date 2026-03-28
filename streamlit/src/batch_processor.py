import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    def __init__(self, detector, classifier_func):
        """
        Khởi tạo BatchProcessor.
        detector: Đối tượng TrafficSignDetector
        classifier_func: Hàm nhận diện hybrid (predict_hybrid)
        """
        self.detector = detector
        self.classifier_func = classifier_func

    def process_folder(self, folder_path, min_s=80, min_v=80, min_size=30, auto_tune=False):
        """
        Quét thư mục và trả về danh sách kết quả.
        Trả về: list of dictionaries { 'filename', 'image_path', 'detections' }
        """
        results = []
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        files.sort()
        
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Chạy detection
            boxes = self.detector.detect(image, min_s=min_s, min_v=min_v, min_size=min_size, auto_tune=auto_tune)
            
            if len(boxes) > 0:
                detections = []
                for box in boxes:
                    x, y, w, h = box
                    roi = image[y:y+h, x:x+w]
                    try:
                        pred_class, conf = self.classifier_func(roi)
                        detections.append({
                            'box': box,
                            'class': pred_class,
                            'confidence': conf
                        })
                    except Exception as e:
                        print(f"Error classifying {filename}: {e}")
                
                results.append({
                    'filename': filename,
                    'image_path': img_path,
                    'detections': detections
                })
        
        return results
