import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC

def render_svm_classification(svm_model, scaled_features, class_names, current_prediction_id, current_dir):
    """Mổ xẻ Bước 4: Ra quyết định (Phân loại SVM) - Dynamic Kernel Lab Edition."""
    st.markdown("---")
    st.header("🧠 Bước 4: Ra quyết định (Phân loại SVM)")
    
    # 1. Hướng dẫn lý thuyết & Bản kế hoạch Tracking Step 4
    plan4_path = os.path.join(current_dir, "docs", "predict", "plans", "plan4.md")
    if os.path.exists(plan4_path):
        with st.expander("📑 Xem Bản kế hoạch Tracking Bước 4 (Chi tiết các tiểu bước)", expanded=False):
            with open(plan4_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
    
    step4_path = os.path.join(current_dir, "docs", "predict", "steps", "step4.md")
    if os.path.exists(step4_path):
        with open(step4_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # Lấy thông tin Top 1 & Top 2 để phục vụ trực quan hóa động
    top_5_names, top_5_probs, top_5_indices = _get_top_predictions(svm_model, scaled_features, class_names, current_prediction_id)

    # --- TIỂU BƯỚC 4.1: TIẾP NHẬN ĐẶC TRƯNG ---
    st.subheader("4.1. Tiếp nhận 256 Mã Gene")
    st.markdown("SVM nhận được 'hồ sơ định danh' là 256 số thực đã được chuẩn hóa. Mỗi số đại diện cho cường độ của một đặc trưng cụ thể.")
    with st.expander("🔍 Soi chi tiết Hồ sơ 256 chiều", expanded=False):
        st.write(scaled_features)

    # --- TIỂU BƯỚC 4.2: PHÒNG THÍ NGHIỆM KERNEL (DYNAMIC) ---
    st.subheader("4.2. Lựa chọn Cơ chế Phán quyết (Kernel Laboratory)")
    st.markdown("""
    Để tách biệt 43 loại biển báo, SVM cần một loại "thước đo" (Kernel) để kẻ ranh giới. 
    Dưới đây là 4 cơ chế phổ biến. Hãy xem ảnh của bạn (ngôi sao đỏ) nằm ở đâu trong mỗi không gian khác nhau:
    """)
    
    _render_kernel_lab(svm_model, scaled_features, top_5_indices, top_5_names)

    # --- TIỂU BƯỚC 4.3: ĐỐI SOÁT CHỨNG CỨ (OVO DUEL) ---
    st.subheader("4.3. Đối soát Chứng cứ (Feature Analysis)")
    _render_duel_analysis(svm_model, scaled_features, top_5_indices, top_5_names)

    # --- TIỂU BƯỚC 4.4: PHÁN QUYẾT CUỐI CÙNG ---
    st.subheader("4.4. Phán quyết Cuối cùng & Độ tin cậy")
    _render_final_decision(top_5_names, top_5_probs)

def _get_top_predictions(svm_model, scaled_features, class_names, current_id):
    """Lấy Top 5 dự đoán để dùng chung cho các tiểu bước."""
    try:
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(scaled_features)[0]
        else:
            scores = svm_model.decision_function(scaled_features)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

        top_5_indices = np.argsort(probs)[-5:][::-1]
        top_5_probs = probs[top_5_indices]
        top_5_names = [class_names.get(idx, f"Label {idx}") for idx in top_5_indices]
        return top_5_names, top_5_probs, top_5_indices
    except:
        return [class_names.get(current_id, "Unknown")], [1.0], [current_id]

def _render_kernel_lab(svm_model, scaled_features, top_5_indices, top_5_names):
    """Trực quan hóa động 4 loại Kernel với công thức toán học."""
    # Công thức LaTeX
    formulas = {
        'linear': r"K(x, x') = x^T x'",
        'rbf': r"K(x, x') = \exp(-\gamma \|x - x'\|^2)",
        'poly': r"K(x, x') = (\gamma x^T x' + r)^d",
        'sigmoid': r"K(x, x') = \tanh(\gamma x^T x' + r)"
    }
    
    active_kernel = getattr(svm_model, 'kernel', 'linear')
    
    # Giả lập tọa độ 2D của ảnh hiện tại dựa trên 2 Gene quan trọng nhất của Duel
    # (Để ngôi sao đỏ 'nhảy' theo đặc điểm của ảnh)
    np.random.seed(top_5_indices[0]) # Cố định vị trí cho cùng 1 nhãn
    x_pos = scaled_features[0, 6] # Lấy đại diện Gene 6 và Gene 200
    y_pos = scaled_features[0, 200]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    titles = ['Linear (Tuyến tính)', 'RBF (Radial Basis)', 'Poly (Đa thức)', 'Sigmoid']
    
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    for i, (k, t) in enumerate(zip(kernels, titles)):
        ax = axes[i//2, i%2]
        
        # Mô phỏng ranh giới quyết định (Decision Boundary) cho từng loại
        if k == 'linear':
            Z = grid[:, 0] + grid[:, 1]
        elif k == 'rbf':
            Z = np.exp(-(grid[:, 0]**2 + grid[:, 1]**2) / 2) - 0.5
        elif k == 'poly':
            Z = (grid[:, 0]**3 + grid[:, 1]**2) - 1
        else: # sigmoid
            Z = np.tanh(grid[:, 0] + grid[:, 1])
            
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap='RdYlGn', alpha=0.3)
        
        # Vẽ 'Ngôi sao đỏ' (Ảnh hiện tại)
        ax.scatter(x_pos, y_pos, c='red', s=150, marker='*', edgecolors='black', label="Ảnh của Bạn")
        
        # Tiêu đề & Công thức
        is_best = " (⭐ ĐANG DÙNG)" if k == active_kernel else ""
        ax.set_title(f"{t}{is_best}", fontsize=10, fontweight='bold')
        ax.text(0.5, -0.2, f"${formulas[k]}$", transform=ax.transAxes, ha='center', fontsize=9, color='blue')
        ax.set_xticks([]); ax.set_yticks([])

    st.pyplot(fig)
    st.info(f"💡 **Phân tích:** Trong không gian 256 chiều, Kernel **{active_kernel.upper()}** được chọn vì nó có khả năng phân tách tốt nhất giữa các biển báo mà không bị Overfitting.")

def _render_duel_analysis(svm_model, scaled_features, top_5_indices, top_5_names):
    """Trực quan hóa 'Nhân chứng Gene' cho cuộc đối đầu Top 1 & Top 2."""
    if len(top_5_indices) < 2 or getattr(svm_model, 'kernel', '') != 'linear' or not hasattr(svm_model, 'coef_'):
        st.info("ℹ️ Chế độ phân tích nhân chứng hiện chỉ hỗ trợ Kernel Linear.")
        return

    i, j = sorted([top_5_indices[0], top_5_indices[1]])
    n_classes = len(svm_model.classes_)
    duel_idx = int(i * (n_classes - 1) - i * (i + 1) / 2 + j - 1)
    
    weights = svm_model.coef_[duel_idx]
    contributions = weights * scaled_features.flatten()
    if i != top_5_indices[0]: contributions = -contributions
    
    top_10 = np.argsort(np.abs(contributions))[-10:]
    vals = contributions[top_10]
    labels = [f"Gene #{idx}" for idx in top_10]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#28a745' if v > 0 else '#dc3545' for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.set_title(f"Hội đồng Chứng cứ: {top_5_names[0]} (Xanh) vs {top_5_names[1]} (Đỏ)")
    st.pyplot(fig)

def _render_final_decision(top_5_names, top_5_probs):
    """Hiển thị kết quả cuối cùng cá nhân hóa."""
    col_v, col_t = st.columns([0.6, 0.4])
    with col_v:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_5_names, top_5_probs * 100, color='#28a745')
        ax.invert_yaxis()
        st.pyplot(fig)
    with col_t:
        st.success(f"🏆 Dự đoán: **{top_5_names[0]}**\n\nĐộ tin cậy: **{top_5_probs[0]*100:.2f}%**")
