import streamlit as st
from src.content_manager import get_ui

ui = get_ui()

def render_math_section(app_mode):
    # --- PHẦN MINH BẠCH TOÁN HỌC ---
    st.divider()
    title = ui.get("math_section.title", "SƠ ĐỒ HỆ THỐNG")
    with st.expander(title if title else "Sơ đồ Hệ thống"):
        sub_title = ui.get("math_section.hybrid_arch_title", "")
        if sub_title:
            st.subheader(sub_title)
            
        desc = ui.get("math_section.hybrid_arch_desc", "")
        if desc:
            st.write(desc)

        if app_mode in ["Phát hiện & Nhận diện (Full Image)", "Quét Thư mục (Batch Mode)"]:
            st.graphviz_chart("""
            digraph G {
                node [shape=box, style=filled, fillcolor=lightblue, color=royalblue, fontname="Helvetica"];
                edge [color=gray, fontname="Helvetica"];
                
                A [label="Ảnh toàn cảnh", fillcolor=lightyellow];
                B [label="Lọc màu HSV"];
                C [label="Morphology Cleanup"];
                D [label="Tìm Contours"];
                E [label="Trích xuất HOG từ các vùng ứng viên"];
                F [label="SVM Binary Detector\n(Nhận diện Viền)"];
                G [label="Cắt các vùng Biển báo\n(Sau NMS)"];
                H [label="CNN Feature Extractor\n(Trích xuát Sâu)"];
                I [label="SVM Multi-class Classifier\n(Định danh)"];
                J [label="Kết quả Cuối cùng", fillcolor=lightgreen];
                
                A -> B -> C -> D -> E -> F -> G -> H -> I -> J;
            }
            """)
        else:
            st.graphviz_chart("""
            digraph G {
                rankdir=LR;
                node [shape=box, style=filled, fillcolor=lightblue, color=royalblue, fontname="Helvetica"];
                edge [color=gray];
                
                A [label="Ảnh Biển Báo\n(Đã Crop)", fillcolor=lightyellow];
                B [label="CNN Feature Extractor\n(256 chiều)"];
                C [label="SVM Classifier\n(Mô hình Tuyến tính)"];
                D [label="Kết quả Nhận diện", fillcolor=lightgreen];
                
                A -> B -> C -> D;
            }
            """)
