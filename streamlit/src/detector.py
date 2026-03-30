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

    def nms(self, boxes, probs, threshold=0.3, return_indices=False):
        """Non-Maximum Suppression để loại bỏ các box chồng lấp."""
        if len(boxes) == 0:
            return ([], []) if return_indices else []

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

        if return_indices:
            return [boxes[i] for i in keep], keep
        return [boxes[i] for i in keep]

    def _apply_clahe(self, image_bgr):
        """Cân bằng độ tương phản (Tự động tối ưu ánh sáng)."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def detect(self, image_bgr, min_s=100, min_v=100, min_size=20, nms_threshold=0.3, return_mask=False, auto_tune=False):
        """
        Sơ đồ thực hiện: [Hyper Turbo-Scan] -> Grid Mask -> SVM -> NMS
        Returns: (final_boxes, mask, stats) if return_mask else (final_boxes, stats)
        """
        stats = {
            'hsv_cnt': 0,
            'geo_pass': 0,
            'filt_pass': 0,
            'svm_pass': 0,
            'nms_kept': 0
        }
        
        if auto_tune:
            image_normalized = self._apply_clahe(image_bgr)
            h, w = image_bgr.shape[:2]
            hsv_full = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2HSV)
            
            s_levels = [40, 80, 120, 160] 
            v_levels = [40, 80, 120, 160] 
            
            all_candidates = []
            
            for s in s_levels:
                for v in v_levels:
                    mask = self._get_hsv_mask(hsv_full, s, v)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    stats['hsv_cnt'] += len(contours)
                    
                    for cnt in contours:
                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        if w_box >= min_size and h_box >= min_size:
                            area_cnt = cv2.contourArea(cnt)
                            if area_cnt < (w_box * h_box * 0.30): continue 
                            
                            aspect_ratio = w_box / float(h_box)
                            if 0.6 < aspect_ratio < 1.4:
                                stats['geo_pass'] += 1
                                y_end, x_end = min(y+h_box, h), min(x+w_box, w)
                                roi = image_normalized[y:y_end, x:x_end]
                                if roi.size == 0: continue
                                
                                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                avg_s = np.mean(roi_hsv[:,:,1])
                                if avg_s < 30: continue 
                                
                                _, s_map, _ = cv2.split(roi_hsv)
                                high_s_ratio = np.sum(s_map > 50) / float(roi.size / 3)
                                if high_s_ratio > 0.95: continue 
                                
                                roi_gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                lap_var = cv2.Laplacian(roi_gray_full, cv2.CV_64F).var()
                                if lap_var < 40: continue 
                                
                                stats['filt_pass'] += 1
                                roi_resized = cv2.resize(roi, (32, 32))
                                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                                features = hog(roi_gray, **self.hog_params)
                                features_scaled = self.scaler.transform(features.reshape(1, -1))
                                
                                if self.model.predict(features_scaled)[0] == 1:
                                    score = self.model.decision_function(features_scaled)[0]
                                    if score > 0.0:
                                        stats['svm_pass'] += 1
                                        all_candidates.append(((x, y, w_box, h_box), score, s, v))

            if not all_candidates:
                return ([], None, stats) if return_mask else ([], stats)
                
            boxes = [c[0] for c in all_candidates]
            scores = [c[1] for c in all_candidates]
            s_list = [c[2] for c in all_candidates]
            v_list = [c[3] for c in all_candidates]
            
            final_boxes, keep_indices = self.nms(boxes, scores, threshold=nms_threshold, return_indices=True)
            stats['nms_kept'] = len(final_boxes)
            stats['sv_params'] = [(s_list[i], v_list[i]) for i in keep_indices]
            
            return (final_boxes, None, stats) if return_mask else (final_boxes, stats)

        # Chế độ Thường (Manual)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = self._get_hsv_mask(hsv, min_s, min_v)
        mask = self._preprocess_mask(mask)

        # Trích xuất và tracking cho Manual mode
        raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stats['hsv_cnt'] = len(raw_contours)
        
        candidates = self._get_raw_candidates_from_mask(image_bgr, mask, min_size)
        # Vì _get_raw_candidates_from_mask gọi các bước nội bộ, ta chỉ lấy svm_pass là chính
        stats['svm_pass'] = len(candidates)
        
        boxes = [c[0] for c in candidates]
        scores = [c[1] for c in candidates]
        final_boxes, keep_indices = self.nms(boxes, scores, threshold=nms_threshold, return_indices=True)
        stats['nms_kept'] = len(final_boxes)
        stats['sv_params'] = [(min_s, min_v) for _ in keep_indices]
        
        if return_mask:
            return final_boxes, mask, stats
        return final_boxes, stats


    def _get_raw_candidates(self, image_bgr, s, v, min_size):
        """Hàm nội bộ để lấy raw candidates từ các ngưỡng cụ thể."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = self._get_hsv_mask(hsv, s, v)
        mask = self._preprocess_mask(mask)
        return self._get_raw_candidates_from_mask(image_bgr, mask, min_size)

    def _get_raw_candidates_from_mask(self, image_bgr, mask, min_size):
        """Trích xuất raw candidates từ một binary mask với các bộ lọc v5.0."""
        # Áp dụng CLAHE một lần nữa cho ROI để đồng nhất với lúc train (tùy chọn)
        image_normalized = self._apply_clahe(image_bgr)
        h, w = image_bgr.shape[:2]
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box)
            
            if w_box >= min_size and h_box >= min_size and 0.6 < aspect_ratio < 1.4:
                # 1. [GEOMETRY] Solidity check
                area_cnt = cv2.contourArea(cnt)
                if area_cnt < (w_box * h_box * 0.30): continue
                
                y_end, x_end = min(y+h_box, h), min(x+w_box, w)
                roi = image_normalized[y:y_end, x:x_end]
                if roi.size == 0: continue
                
                # 2. [VIBRANCE + DENSITY] Lọc đốm nhiễu 
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                avg_s = np.mean(roi_hsv[:,:,1])
                if avg_s < 30: continue 
                
                _, s_map, _ = cv2.split(roi_hsv)
                high_s_ratio = np.sum(s_map > 50) / float(roi.size / 3)
                if high_s_ratio > 0.95: continue 
                
                # 3. [FOCUS] Kiểm tra độ nét trên ROI gốc
                roi_gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(roi_gray_full, cv2.CV_64F).var()
                if lap_var < 40: continue 
                
                # 4. [HOG + SVM]
                roi_resized = cv2.resize(roi, (32, 32))
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                features = hog(roi_gray, **self.hog_params)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                if self.model.predict(features_scaled)[0] == 1:
                    score = self.model.decision_function(features_scaled)[0]
                    if score > 0.0:
                        candidates.append(((x, y, w_box, h_box), score))
        return candidates
