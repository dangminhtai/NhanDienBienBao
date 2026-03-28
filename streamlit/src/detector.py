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

    def _get_hsv_mask(self, hsv_img, min_s=100, min_v=100):
        """Lọc màu đỏ, xanh dương và vàng để tìm vùng ứng viên."""
        # Màu đỏ (Mở rộng dải Hue để không sót biển Cấm/Dừng)
        lower_red1 = np.array([0, min_s, min_v])
        upper_red1 = np.array([15, 255, 255])
        mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([155, min_s, min_v])
        upper_red2 = np.array([179, 255, 255])
        mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Màu xanh dương
        lower_blue = np.array([100, min_s, min_v])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        
        # Màu vàng (Tối ưu để bắt biển DIAMOND, loại bỏ lá cây xanh/nâu)
        lower_yellow = np.array([17, min_s, min_v])
        upper_yellow = np.array([33, 255, 255])
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(red_mask, blue_mask)
        combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
        return combined_mask

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

    def _apply_clahe(self, image_bgr):
        """Cân bằng độ tương phản (Tự động tối ưu ánh sáng)."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def detect(self, image_bgr, min_s=100, min_v=100, min_size=20, return_mask=False, auto_tune=False):
        """
        Sơ đồ thực hiện: [Hyper Turbo-Scan] -> Downscale -> Grid Mask -> Upscale ROI -> SVM -> NMS
        Nếu auto_tune=True, nó sẽ quét lưới mịn trên ảnh thu nhỏ để đạt tốc độ tối đa.
        """
        if auto_tune:
            # 1. Tối ưu ánh sáng (CLAHE) - Làm 1 lần duy nhất trên ảnh gốc
            image_normalized = self._apply_clahe(image_bgr)
            
            # 2. Hạ độ phân giải để quét (Downscaling)
            h, w = image_bgr.shape[:2]
            scale = 800.0 / max(w, h) if max(w, h) > 800 else 1.0
            image_small = cv2.resize(image_normalized, (0, 0), fx=scale, fy=scale)
            hsv_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2HSV)
            
            # 3. Quét lưới mịn nhưng an toàn (v4.6 Ghost Hunter)
            s_levels = [50, 90, 130]
            v_levels = [50, 90, 130] 
            target_min_size_small = int(min_size * scale)
            
            all_candidates = []
            seen_boxes = set()
            
            for s in s_levels:
                for v in v_levels:
                    mask = self._get_hsv_mask(hsv_small, s, v)
                    # morphology mạnh hơn để nối các phần biển báo bị vỡ (như vạch trắng giữa biển đỏ)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        xs, ys, ws, hs = cv2.boundingRect(cnt)
                        if ws >= target_min_size_small and hs >= target_min_size_small:
                            # 1. [GEOMETRY] Solidity > 0.3 (Cho phép biển có họa tiết trắng lớn)
                            area = cv2.contourArea(cnt)
                            if area < (ws * hs * 0.30): continue 
                            
                            x, y = int(xs / scale), int(ys / scale)
                            w_full, h_full = int(ws / scale), int(hs / scale)
                            
                            # Khử trùng lặp (Dùng tọa độ chia 10 để gom nhóm chặt hơn)
                            box_key = (x//10, y//10, w_full//10, h_full//10)
                            if box_key in seen_boxes: continue
                            
                            aspect_ratio = w_full / float(h_full)
                            if 0.6 < aspect_ratio < 1.4:
                                y_end, x_end = min(y+h_full, h), min(x+w_full, w)
                                roi = image_normalized[y:y_end, x:x_end]
                                if roi.size == 0: continue
                                
                                # 2. [VIBRANCE] Kiểm tra độ rực rỡ (Nới lỏng xuống 30)
                                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                avg_s = np.mean(roi_hsv[:,:,1])
                                if avg_s < 30: continue # Biển báo thật thường rất rực rỡ
                                
                                # 3. [FOCUS] Kiểm tra độ nét (Nới lỏng xuống 40)
                                roi_gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                lap_var = cv2.Laplacian(roi_gray_full, cv2.CV_64F).var()
                                if lap_var < 40: continue # Biển báo kim loại luôn có texture sắc nét
                                
                                roi_resized = cv2.resize(roi, (32, 32))
                                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                                features = hog(roi_gray, **self.hog_params)
                                features_scaled = self.scaler.transform(features.reshape(1, -1))
                                
                                if self.model.predict(features_scaled)[0] == 1:
                                    score = self.model.decision_function(features_scaled)[0]
                                    # 4. [SVM] Trả về ngưỡng 0.0 tiêu chuẩn
                                    if score > 0.0:
                                        all_candidates.append(((x, y, w_full, h_full), score))
                                        seen_boxes.add(box_key)

            if not all_candidates:
                return ([], None) if return_mask else []
                
            boxes = [c[0] for c in all_candidates]
            scores = [c[1] for c in all_candidates]
            final_boxes = self.nms(boxes, scores)
            
            return (final_boxes, None) if return_mask else final_boxes

        # Chế độ Thường (Manual)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = self._get_hsv_mask(hsv, min_s, min_v)
        mask = self._preprocess_mask(mask)

        candidates = self._get_raw_candidates_from_mask(image_bgr, mask, min_size)
        
        if not candidates:
            return ([], mask) if return_mask else []

        boxes = [c[0] for c in candidates]
        scores = [c[1] for c in candidates]
        final_boxes = self.nms(boxes, scores)
        
        if return_mask:
            return final_boxes, mask
        return final_boxes


    def _get_raw_candidates(self, image_bgr, s, v, min_size):
        """Hàm nội bộ để lấy raw candidates từ các ngưỡng cụ thể."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = self._get_hsv_mask(hsv, s, v)
        mask = self._preprocess_mask(mask)
        return self._get_raw_candidates_from_mask(image_bgr, mask, min_size)

    def _get_raw_candidates_from_mask(self, image_bgr, mask, min_size):
        """Trích xuất raw candidates từ một binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if w > min_size and h > min_size and 0.6 < aspect_ratio < 1.4:
                roi = image_bgr[y:y+h, x:x+w]
                if roi.size == 0: continue
                roi_resized = cv2.resize(roi, (32, 32))
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                features = hog(roi_gray, **self.hog_params)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                if self.model.predict(features_scaled)[0] == 1:
                    score = self.model.decision_function(features_scaled)[0]
                    if score > -0.5:
                        candidates.append(((x, y, w, h), score))
        return candidates
