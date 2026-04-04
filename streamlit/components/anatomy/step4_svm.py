import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def render_svm_classification(svm_model, scaled_features, class_names, current_prediction_id, current_dir):
    """Mổ xẻ Bước 4: Ra quyết định (Phân loại SVM) - Pedagogical Storytelling Edition."""
    st.markdown("---")
    st.header("🧠 Bước 4: Ra quyết định (Phân loại SVM)")
    
    # 1. Hướng dẫn lý thuyết & Bản kế hoạch Tracking Step 4
    plan4_path = os.path.join(current_dir, "docs", "predict", "plans", "plan4.md")
    if os.path.exists(plan4_path):
        with st.expander("📑 Xem Bản kế hoạch Tracking Bước 4 (Chi tiết)", expanded=False):
            with open(plan4_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
    
    step4_path = os.path.join(current_dir, "docs", "predict", "steps", "step4.md")
    if os.path.exists(step4_path):
        with open(step4_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # --- TIỂU BƯỚC 4.1: TIẾP NHẬN ĐẶC TRƯNG ---
    st.subheader("4.1. Tiếp nhận 256 Mã Gene")
    st.markdown("CNN đã nén ảnh thành 256 đặc trưng (Gene). SVM sẽ dùng bộ mã này làm 'tọa độ' để tìm vị trí biển báo trên bản đồ tri thức.")

    # --- TIỂU BƯỚC 4.2: PHÉP MÀU KERNEL (THE KERNEL TRICK STORY) ---
    st.subheader("4.2. Phép màu Kernel: Biến cái không thể thành có thể")
    
    with st.expander("🤔 Tại sao chúng ta cần Kernel? (Bấm để xem giải mã)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Vấn đề: Thế giới 1 chiều bế tắc**")
            st.markdown("Hãy tưởng tượng các biển báo nằm xen kẽ trên một sợi dây. Không có một 'nhát cắt' thẳng nào có thể tách riêng màu Đỏ ra khỏi màu Xanh.")
            _plot_1d_problem()
            
        with col2:
            st.markdown("**Giải pháp: Nhấc bổng không gian (Lifting)**")
            st.markdown("SVM 'bẻ cong' sợi dây thành hình chữ U trong không gian 2 chiều. Lúc này, chỉ cần một nhát cắt ngang phẳng là chia đôi được chúng!")
            _plot_2d_lifting()
        
        st.success("💡 **Bản chất:** Kernel giúp SVM 'nhìn' thấy các chiều không gian cao hơn (như 256 chiều của bạn) nơi mà các biển báo vốn dĩ đã được tách biệt rõ ràng.")

    # --- TIỂU BƯỚC 4.3: PHÒNG THÍ NGHIỆM 4 LOẠI KERNEL ---
    st.subheader("4.3. Bốn 'Triết lý' Phán quyết")
    st.markdown("Mỗi loại Kernel có một 'tính cách' khác nhau khi kẻ đường biên:")
    _render_kernel_gallery(svm_model, scaled_features)

    # --- TIỂU BƯỚC 4.4: HỘI ĐỒNG CHỨNG CỨ & PHÁN QUYẾT ---
    st.subheader("4.4. Hội đồng Chứng cứ & Phán quyết Cuối cùng")
    top_5_names, top_5_probs, top_5_indices = _get_predictions(svm_model, scaled_features, class_names, current_prediction_id)
    _render_duel_and_result(svm_model, scaled_features, top_5_indices, top_5_names, top_5_probs)

def _plot_1d_problem():
    """Vẽ minh họa 1D không thể phân tách."""
    x = np.array([-2, -1, 0, 1, 2])
    colors = ['green', 'green', 'red', 'green', 'green']
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.scatter(x, np.zeros_like(x), c=colors, s=100, edgecolors='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_yticks([])
    ax.set_title("1D: Bế tắc (Inseparable)")
    st.pyplot(fig)

def _plot_2d_lifting():
    """Vẽ minh họa 2D sau khi dùng Kernel Trick (y = x^2)."""
    x = np.array([-2, -1, 0, 1, 2])
    y = x**2
    colors = ['green', 'green', 'red', 'green', 'green']
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.scatter(x, y, c=colors, s=100, edgecolors='black')
    # Nhát cắt ngăn cách
    ax.axhline(0.5, color='blue', linestyle='--', label='Nhát cắt SVM')
    ax.set_title("2D: Phép màu (Separable!)")
    st.pyplot(fig)

def _render_kernel_gallery(svm_model, scaled_features):
    """Trực quan hóa 4 loại Kernel với ẩn dụ dễ hiểu."""
    formulas = {
        'linear': r"x^T x'",
        'rbf': r"\exp(-\gamma \|x - x'\|^2)",
        'poly': r"(\gamma x^T x' + r)^d",
        'sigmoid': r"\tanh(\gamma x^T x' + r)"
    }
    active_k = getattr(svm_model, 'kernel', 'linear')
    
    cols = st.columns(4)
    titles = ["Linear", "RBF", "Poly", "Sigmoid"]
    metaphors = ["Cây thước thẳng", "Vùng lan tỏa", "Đường uốn lượn", "Nơ-ron ảo"]
    
    # Giả lập tọa độ từ 256D (Lấy 2 Gene mạnh nhất để 'nhảy' theo ảnh)
    x_star = scaled_features[0, 6] 
    y_star = scaled_features[0, 200]
    
    xx, yy = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for i, (k, title, meta) in enumerate(zip(['linear', 'rbf', 'poly', 'sigmoid'], titles, metaphors)):
        with cols[i]:
            st.markdown(f"**{title}**")
            st.caption(f"*{meta}*")
            
            fig, ax = plt.subplots(figsize=(3, 3))
            if k == 'linear': Z = grid[:, 0] + grid[:, 1]
            elif k == 'rbf': Z = np.exp(-(grid[:, 0]**2 + grid[:, 1]**2)/2) - 0.5
            elif k == 'poly': Z = (grid[:, 0]**3 + grid[:, 1]**2) - 1
            else: Z = np.tanh(grid[:, 0] + grid[:, 1])
            
            ax.contourf(xx, yy, Z.reshape(xx.shape), cmap='RdYlGn', alpha=0.3)
            ax.scatter(x_star, y_star, c='red', marker='*', s=100, edgecolors='black')
            ax.set_xticks([]); ax.set_yticks([])
            st.pyplot(fig)
            st.latex(formulas[k])
            
            if k == active_k:
                st.success("💎 QUÁN QUÂN")

def _get_predictions(svm_model, scaled_features, class_names, current_id):
    """Tính toán logic nội bộ SVM."""
    try:
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(scaled_features)[0]
        else:
            scores = svm_model.decision_function(scaled_features)[0]
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / exp_s.sum()
        top_indices = np.argsort(probs)[-5:][::-1]
        return [class_names.get(i, f"L{i}") for i in top_indices], probs[top_indices], top_indices
    except:
        return [class_names.get(current_id, "Unknown")], [1.0], [current_id]

def _render_duel_and_result(svm_model, scaled_features, top_indices, top_names, top_probs):
    """Phần 4.4: Duel và Kết quả."""
    if len(top_indices) >= 2 and getattr(svm_model, 'kernel', '') == 'linear' and hasattr(svm_model, 'coef_'):
        st.markdown(f"**🔎 Phân tích chứng cứ:** Tại sao chọn '{top_names[0]}' thay vì '{top_names[1]}'?")
        i, j = sorted([top_indices[0], top_indices[1]])
        n_classes = len(svm_model.classes_)
        duel_idx = int(i * (n_classes - 1) - i * (i + 1) / 2 + j - 1)
        weights = svm_model.coef_[duel_idx]
        contrib = (weights * scaled_features.flatten()) * (1 if i == top_indices[0] else -1)
        
        top_10 = np.argsort(np.abs(contrib))[-10:]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar([f"Gene #{idx}" for idx in top_10], contrib[top_10], color=['green' if v > 0 else 'red' for v in contrib[top_10]])
        ax.axhline(0, color='black', linewidth=0.8)
        st.pyplot(fig)
        st.caption("Cột xanh ủng hộ kết quả hiện tại. Cột đỏ ủng hộ phương án dự phòng.")

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(top_names, top_probs*100, color='#28a745')
        ax.invert_yaxis()
        st.pyplot(fig)
    with c2:
        st.success(f"🏆 Dự đoán: **{top_names[0]}**\n\nTin cậy: **{top_probs[0]*100:.2f}%**")
