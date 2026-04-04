import json
import os
import streamlit as st

@st.cache_data(show_spinner=False)
def load_json_content(config_path):
    """Nạp nội dung từ file JSON (có cache và tự động reload khi file thay đổi)."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Lỗi khi đọc file content.json: {e}")
            return {}
    return {}

class UIContent:
    def __init__(self, config_path=None):
        if config_path is None:
            # Tìm đường dẫn mặc định: ../config/content.json (tương đối với file này)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "content.json")
        
        self.config_path = config_path

    def get(self, key_path, default=""):
        """
        Lấy giá trị từ key path (ví dụ: 'single_predict.title').
        Sử dụng content đã được cache bởi Streamlit.
        """
        content = load_json_content(self.config_path)
        keys = key_path.split('.')
        value = content
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

def get_ui():
    """Trả về instance của UIContent."""
    return UIContent()
