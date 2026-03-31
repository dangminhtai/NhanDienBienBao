import streamlit as st
import os
from PIL import ImageDraw, ImageFont

def draw_vietnamese_text(image_pil, text, position, font_size=20, color=(0, 255, 0)):
    """Vẽ chữ tiếng Việt lên ảnh PIL."""
    draw = ImageDraw.Draw(image_pil)
    # Thử nạp font Arial trên Windows, nếu không dùng default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Vẽ nền cho chữ để dễ đọc
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 100))
    draw.text(position, text, font=font, fill=color)
    return image_pil

def load_css(file_name):
    """Nạp file CSS từ đường dẫn cục bộ."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
