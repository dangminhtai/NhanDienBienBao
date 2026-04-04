import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC

def render_svm_classification(svm_model, scaled_features, class_names, current_prediction_id, current_dir):
    """Mổ xẻ Bước 4: Ra quyết định (Phân loại SVM) - Balance Scale Edition."""
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

    # Lấy thông tin Top 1 & Top 2 để phục vụ trực quan hóa động
    top_5_names, top_5_probs, top_5_indices = _get_top_predictions(svm_model, scaled_features, class_names, current_prediction_id)

    # --- TIỂU BƯỚC 4.1: TIẾP NHẬN ĐẶC TRƯNG ---
    st.subheader("4.1. Bước đệm: Tiếp nhận hồ sơ 256 chiều")
    st.markdown("""
    Sau khi được CNN mổ xẻ và chuẩn hóa, ảnh của anh giờ đây là một bộ mã gồm 256 số thực. 
    Mỗi con số này là một 'đặc điểm nhận dạng' cực kỳ tinh vi.
    """)
    with st.expander("🔍 Xem hồ sơ định danh (256 Mã Gene)", expanded=False):
        st.write(scaled_features)

    # --- TIỂU BƯỚC 4.2: PHÉP MÀU KERNEL (THE STORY) ---
    st.subheader("4.2. Cơ chế Phán quyết (Kernel Logic)")
    _render_kernel_story(svm_model, scaled_features, top_5_indices)

    # --- TIỂU BƯỚC 4.3: HỘI ĐỒNG CHỨNG CỨ & CÁN CÂN LOGIC ---
    st.subheader("4.3. Hội đồng Chứng cứ & Cán cân Logic")
    _render_balance_scale_logic(svm_model, scaled_features, top_5_indices, top_5_names)

    # --- TIỂU BƯỚC 4.4: PHÁN QUYẾT CUỐI CÙNG ---
    st.subheader("4.4. Phán quyết Cuối cùng")
    _render_final_verdict(top_5_names, top_5_probs)

def _get_top_predictions(svm_model, scaled_features, class_names, current_id):
    """Tính toán logic nội bộ SVM."""
    try:
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(scaled_features)[0]
        else:
            scores = svm_model.decision_function(scaled_features)[0]
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / exp_s.sum()
        top_indices = np.argsort(probs)[-5:][::-1]
        probs_top = probs[top_indices]
        names_top = [class_names.get(idx, f"Label {idx}") for idx in top_indices]
        return names_top, probs_top, top_indices
    except:
        return [class_names.get(current_id, "Unknown")], [1.0], [current_id]

def _render_kernel_story(svm_model, scaled_features, top_5_indices):
    """Câu chuyện bẻ cong không gian."""
    with st.expander("💡 Tại sao chúng ta cần Kernel? (Bấm để xem sự ảo diệu)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trạng thái 1D: Bế tắc**")
            st.caption("Các điểm xanh đỏ nằm lẫn lộn, không thể chia đôi bằng một đường thẳng phẳng.")
            _plot_1d()
        with c2:
            st.markdown("**Sau khi dùng Kernel: Nhấc bổng**")
            st.caption("Không gian được uốn cong, ranh giới (đường xanh) hiện ra cực kỳ rõ ràng.")
            _plot_2d()
        st.success("Mô hình của anh sử dụng bộ mã 256 chiều, nơi mọi ranh giới đều được tách biệt rõ ràng nhờ phép màu Kernel.")

def _render_balance_scale_logic(svm_model, scaled_features, top_indices, top_names):
    """Trực quan hóa Cán cân Logic giữa Top 1 và Top 2."""
    if len(top_indices) < 2 or getattr(svm_model, 'kernel', '') != 'linear' or not hasattr(svm_model, 'coef_'):
        st.info("ℹ️ Chế độ 'Cán cân Logic' hiện chỉ khả dụng cho mô hình Linear SVM.")
        return

    st.markdown(f"**Cuộc tranh luận giữa:** '{top_names[0]}' (Xanh) và '{top_names[1]}' (Đỏ)")
    
    # 1. Tính toán Contribution
    i, j = sorted([top_indices[0], top_indices[1]])
    n_classes = len(svm_model.classes_)
    duel_idx = int(i * (n_classes - 1) - i * (i + 1) / 2 + j - 1)
    
    weights = svm_model.coef_[duel_idx]
    x = scaled_features.flatten()
    contrib = weights * x
    if i != top_indices[0]: contrib = -contrib # Luôn để Top 1 là dương
    
    # 2. Lấy Top 5 tích cực và Top 5 tiêu cực
    idx_sorted = np.argsort(contrib)
    worst_5 = idx_sorted[:5] # Ủng hộ nhãn kia (Red)
    best_5 = idx_sorted[-5:] # Ủng hộ nhãn hiện tại (Green)
    
    all_selected = np.concatenate([worst_5, best_5])
    all_vals = contrib[all_selected]
    all_labels = [f"Yếu tố #{idx}" for idx in all_selected]
    
    # 3. Vẽ Cán cân Ngang (Horizontal Balance)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#dc3545' if v < 0 else '#28a745' for v in all_vals]
    ax.barh(all_labels, all_vals, color=colors)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_title("Cán cân Logic: Những yếu tố đang 'kéo' kết quả", fontsize=12)
    ax.set_xlabel("Mức độ thuyết phục (Sức kéo của Gene)")
    st.pyplot(fig)
    
    # 4. Hội thoại Nhân chứng (Dialogue Analysis)
    st.markdown("**🎙️ Lời khai của các nhân chứng (TOP CHỨNG CỨ):**")
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("<h4 style='color:green;'>👍 Phe ủng hộ (Pro)</h4>", unsafe_allow_html=True)
        st.write(f"• **Yếu tố #{best_5[-1]}**: 'Tôi thấy các đường nét này cực kỳ giống biển {top_names[0]}!' (Đóng góp mạnh nhất)")
        st.write(f"• **Yếu tố #{best_5[-2]}**: 'Màu sắc và hình khối hoàn toàn trùng khớp.'")
    with col_con:
        st.markdown("<h4 style='color:red;'>👎 Phe phản đối (Con)</h4>", unsafe_allow_html=True)
        st.write(f"• **Yếu tố #{worst_5[0]}**: 'Nhưng vết mờ này cũng hơi giống biển {top_names[1]} đó nha!'")
        st.write(f"• **Yếu tố #{worst_5[1]}**: 'Cần cẩn thận vì góc cạnh này dễ gây nhầm lẫn.'")

def _render_final_verdict(top_names, top_probs):
    """Thiết kế lại Thẻ pháng quyết Premium."""
    st.markdown("---")
    res_col, bar_col = st.columns([0.4, 0.6])
    
    with res_col:
        st.markdown(f"""
        <div style="border: 2px solid #28a745; padding: 20px; border-radius: 10px; background-color: #f8fff9;">
            <h2 style="color: #28a745; margin-top: 0;">🏆 PHÁN QUYẾT</h2>
            <p style="font-size: 1.2em;">Sau khi cân soát mọi chứng cứ, AI kết luận đây là:</p>
            <h1 style="color: #155724; font-size: 2.5em; text-transform: uppercase;">{top_names[0]}</h1>
            <hr>
            <p style="font-weight: bold; font-size: 1.1em;">Độ tin cậy: {top_probs[0]*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with bar_col:
        st.markdown("#### Bảng xếp hạng Độ tin cậy (Top 5)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_names[:5], top_probs[:5] * 100, color=['#28a745'] + ['#6c757d']*4)
        ax.invert_yaxis()
        ax.set_xlabel("Xác suất (%)")
        st.pyplot(fig)

def _plot_1d():
    x = np.array([-2, -1, 0, 1, 2]); c = ['green', 'green', 'red', 'green', 'green']
    f, a = plt.subplots(figsize=(4, 1.5)); a.scatter(x, [0]*5, c=c, s=100); a.axhline(0, color='gray')
    a.set_yticks([]); st.pyplot(f)

def _plot_2d():
    x = np.array([-2, -1, 0, 1, 2]); y = x**2; c = ['green', 'green', 'red', 'green', 'green']
    f, a = plt.subplots(figsize=(4, 1.5)); a.scatter(x, y, c=c, s=100); a.axhline(0.5, color='blue', ls='--')
    st.pyplot(f)
