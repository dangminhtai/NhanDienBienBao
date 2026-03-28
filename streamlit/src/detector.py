import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

class TrafficSignDetector:
    def __init__(self, model_path, scaler_path):
        """
        Khởi tạo detector với mô hình SVM và scaler đã huấn luyện.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Không tìm thấy scaler tại {scaler_path}")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Cấu hình HOG (phải khớp với lúc huấn luyện/tạo scaler)
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'visualize': False,
            'feature_vector': True
        }

    def _get_hsv_mask(self, hsv_img):
        """Lọc màu đỏ và xanh dương để tìm vùng ứng viên."""
        # Màu đỏ (2 dải)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Màu xanh dương
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        return cv2.bitwise_or(red_mask, blue_mask)

    def _preprocess_mask(self, mask):
        """Làm sạch mask bằng morphology."""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def nms(self, boxes, probs, threshold=0.3):
        """Non-Maximum Suppression để loại bỏ các box chồng lấp."""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(probs)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def detect(self, image_bgr):
        """
        Sơ đồ thực hiện: HSV Mask -> Contours -> SVM Verification -> NMS
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = self._get_hsv_mask(hsv)
        mask = self._preprocess_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Lọc sơ bộ theo kích thước và tỷ lệ
            if w > 20 and h > 20 and 0.7 < aspect_ratio < 1.3:
                # Cắt vùng ảnh ứng viên (ROI)
                roi = image_bgr[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # Resize về 64x64 để trích xuất HOG (khớp với detect_scaler 324 features)
                # 324 features = (64/8 - 1)^2 * 2^2 * 9 = 7^2 * 4 * 9 = 49 * 36 = 1764? 
                # Khoan, 324 features = 9 orientations * 2x2 cells * (size/8 - 1)^2
                # Nếu size=32: (32/8 - 1)^2 * 4 * 9 = 3^2 * 36 = 9 * 36 = 324. CHÍNH XÁC!
                # Phải resize ROI về 32x32.
                roi_resized = cv2.resize(roi, (32, 32))
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                
                # Trích xuất HOG
                features = hog(roi_gray, **self.hog_params)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Dự đoán bằng SVM nhị phân
                prediction = self.model.predict(features_scaled)
                # Lấy xác suất nếu model hỗ trợ (hoặc dùng decision_function)
                # detect_model.pkl thường là SVC nhị phân
                if prediction[0] == 1: # Giả định 1 là Traffic Sign
                    # Tính score đơn giản (SVM decision function)
                    score = self.model.decision_function(features_scaled)[0]
                    if score > 0: # Chỉ lấy các mẫu nằm ngoài lề phân tách dương
                        candidates.append(((x, y, w, h), score))

        if not candidates:
            return []

        boxes = [c[0] for c in candidates]
        scores = [c[1] for c in candidates]
        
        # Áp dụng NMS
        final_boxes = self.nms(boxes, scores)
        
        return final_boxes
