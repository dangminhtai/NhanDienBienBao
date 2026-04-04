import json
import os

class UIContent:
    def __init__(self, config_path=None):
        if config_path is None:
            # Tìm đường dẫn mặc định: ../config/content.json (tương đối với file này)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "content.json")
        
        self.config_path = config_path
        self.content = {}
        self.load()

    def load(self):
        """Nạp nội dung từ file JSON."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.content = json.load(f)
            except Exception as e:
                print(f"Lỗi khi đọc file content.json: {e}")
                self.content = {}
        else:
            print(f"Không tìm thấy file cấu hình tại: {self.config_path}")
            self.content = {}

    def get(self, key_path, default=""):
        """
        Lấy giá trị từ key path (ví dụ: 'single_predict.title').
        Nếu không thấy, trả về giá trị mặc định.
        """
        keys = key_path.split('.')
        value = self.content
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Singleton instance để dùng chung toàn app
_ui_instance = None

def get_ui():
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = UIContent()
    return _ui_instance
