import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def render_svm_classification(svm_model, scaled_features, class_names, current_prediction_id, current_dir):
    """Mổ xẻ Bước 4: Ra quyết định cuối cùng (SVM Classification)."""
    st.markdown("---")
    st.header("🧠 Bước 4: Ra quyết định (Phân loại SVM)")
    
    # 1. Hướng dẫn lý thuyết
    step4_path = os.path.join(current_dir, "docs", "predict", "steps", "step4.md")
    if os.path.exists(step4_path):
        with open(step4_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # Khởi tạo giá trị mặc định để tránh lỗi UnboundLocalError
    top_5_names = [class_names.get(current_prediction_id, f"Nhãn {current_prediction_id}")]
    top_5_probs = [1.0] # Mặc định 100% nếu không tính được xác suất cụ thể
    
    try:
        # Kiểm tra xem Model có hỗ trợ xác suất không
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(scaled_features)[0]
        else:
            # Fallback: Dùng decision_function (khoảng cách tới siêu mặt phẳng)
            # Sau đó dùng Softmax để đưa về dạng xác suất (mô phỏng)
            scores = svm_model.decision_function(scaled_features)[0]
            # Softmax function
            exp_scores = np.exp(scores - np.max(scores)) # Tránh tràn số
            probs = exp_scores / exp_scores.sum()
            st.info("ℹ️ Hệ thống dùng thuật toán Khoảng cách (Decision Function) để ước tính xác suất.")

        # Sắp xếp để lấy top 5
        top_5_indices = np.argsort(probs)[-5:][::-1]
        top_5_probs = probs[top_5_indices]
        top_5_names = [class_names.get(idx, f"Nhãn {idx}") for idx in top_5_indices]
        
        col_chart, col_explain = st.columns([0.6, 0.4])
        
        with col_chart:
            # Vẽ biểu đồ cột nằm ngang
            fig_p, ax_p = plt.subplots(figsize=(8, 4))
            bars = ax_p.barh(top_5_names, top_5_probs * 100, color='#28a745')
            ax_p.set_xlabel("Độ tin cậy (%)")
            ax_p.set_title("Xác suất phân loại của SVM")
            ax_p.invert_yaxis() # Đảo ngược để Top 1 ở trên cùng
            
            # Thêm nhãn % vào sau thanh
            for bar in bars:
                width = bar.get_width()
                ax_p.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
                
            st.pyplot(fig_p)

        with col_explain:
            st.info(f"""
            🌟 **Lựa chọn cuối cùng:** 
            **{top_5_names[0]}** (Độ tin cậy {top_5_probs[0]*100:.2f}%)
            
            Đây là nhãn có "khoảng cách" an toàn nhất so với các Siêu mặt phẳng biên giới.
            """)
            
            if len(top_5_probs) > 1 and top_5_probs[1] > 0.05:
                st.warning(f"💡 **AI đang phân vân?** Có vẻ biển báo này cũng khá giống với **{top_5_names[1]}**.")

    except Exception as e:
        st.error(f"⚠️ Lỗi khi phân tích xác suất: {str(e)}")

    # 3. Minh họa Không gian Quyết định (Decision Space Simulation)
    st.markdown("### 🗺️ Bản đồ Không gian Quyết định (Mô phỏng 2D)")
    
    fig_map, ax_map = plt.subplots(figsize=(8, 6))
    
    # Tạo các "vùng lãnh thổ" giả lập quanh ảnh hiện tại
    np.random.seed(42)
    # Vùng cho Top 1 (Green)
    points_t1 = np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2))
    # Vùng cho Top 2 (Orange)
    points_t2 = np.random.normal(loc=[-2, 1], scale=0.6, size=(25, 2))
    # Vùng cho các loại khác (Grey)
    points_other = np.random.normal(loc=[0, -2], scale=1.0, size=(40, 2))
    
    ax_map.scatter(points_t1[:,0], points_t1[:,1], c='green', alpha=0.3, label=f"Vùng {top_5_names[0]}")
    ax_map.scatter(points_t2[:,0], points_t2[:,1], c='orange', alpha=0.3, label=f"Vùng {top_5_names[1]}")
    ax_map.scatter(points_other[:,0], points_other[:,1], c='gray', alpha=0.1, label="Các vùng khác")
    
    # Vị trí của ảnh hiện tại (Ngôi sao đỏ)
    # Giả lập vị trí: Nếu tự tin cao thì nằm sâu trong vùng xanh, nếu phân vân thì nằm gần biên giới
    current_pos = [2.2 - (1 - top_5_probs[0])*4, 2.2 - (1 - top_5_probs[0])*2]
    ax_map.scatter(current_pos[0], current_pos[1], c='red', s=200, marker='*', label="Ảnh của Anh", edgecolors='black')
    
    # Vẽ "Bức tường" (Hyperplane) giả lập giữa Vùng 1 và Vùng 2
    x_h = np.linspace(-4, 4, 100)
    y_h = 0.5 * x_h + 1.5
    ax_map.plot(x_h, y_h, color='black', linestyle='--', alpha=0.5, label="Siêu mặt phẳng (Bức tường)")

    ax_map.set_title("Vị trí của Ảnh trong Vũ trụ 256 chiều")
    ax_map.legend(loc='lower right')
    ax_map.grid(True, alpha=0.1)
    st.pyplot(fig_map)
    st.caption("Ngôi sao đỏ chính là 'Danh tính' của bức ảnh sau khi được CNN đóng gói và chuẩn hóa. SVM đã thấy nó nằm trọn trong vùng xanh của biển báo này.")

    st.success(f"🏆 **KẾT LUẬN:** BIỂN BÁO ĐƯỢC XÁC ĐỊNH LÀ: **{top_5_names[0].upper()}**")
