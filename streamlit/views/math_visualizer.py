import streamlit as st

def render_math_section(app_mode):
    # --- PHẦN MINH BẠCH TOÁN HỌC ---
    st.divider()
    with st.expander("📊 CƠ SỞ TOÁN HỌC VÀ QUY TRÌNH HỆ THỐNG"):
        st.subheader("1. Kiến trúc Hybrid v4.0")
        st.write("""
        Hệ thống kết hợp **CNN** để trích xuất đặc trưng sâu (256 chiều) và **SVM** để phân loại. 
        Đối với tác vụ phát hiện, chúng tôi sử dụng lọc màu **HSV** kết hợp với **SVM Binary** dựa trên đặc trưng **HOG**.
        """)

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
