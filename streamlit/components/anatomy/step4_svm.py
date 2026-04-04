import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def render_svm_classification(svm_model, scaled_features, class_names, current_prediction_id, current_dir):
    """Mổ xẻ Bước 4: Ra quyết định cuối cùng (SVM Classification)."""
    st.markdown("---")
    st.header("🧠 Bước 4: Ra quyết định (Phân loại SVM)")
    
    # 1. Hướng dẫn lý thuyết & Bản kế hoạch Tracking Step 4
    plan4_path = os.path.join(current_dir, "docs", "predict", "plans", "plan4.md")
    if os.path.exists(plan4_path):
        with st.expander("📑 Xem Bản kế hoạch Tracking Bước 4 (Chi tiết logic SVM)", expanded=False):
            with open(plan4_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
    
    step4_path = os.path.join(current_dir, "docs", "predict", "steps", "step4.md")
    if os.path.exists(step4_path):
        with open(step4_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # --- PHẦN 1: TUYỂN CHỌN KERNEL (Grid Search Logic) ---
    st.markdown("### 🏟️ Tuyển chọn Kernel (Grid Search Tournament)")
    st.markdown("""
    Trong file `GTSRB_(SVM_+_CNN).py` (Dòng 297), anh đã yêu cầu AI tìm kiếm Kernel tốt nhất qua 3 bước:
    - **Thử nghiệm:** Chạy cả 4 loại Kernel (Linear, RBF, Poly, Sigmoid).
    - **Đối soát:** Dùng xác thực chéo (Cross-Validation) để chấm điểm `accuracy`.
    - **Lựa chọn:** Chọn ra 'Quán quân' để lưu vào mô hình hiện tại.
    """)
    
    # Hiển thị bảng ứng cử viên
    candidates = pd.DataFrame({
        "Ứng cử viên": ["Linear (Đường thẳng)", "RBF (Đường cong Radial)", "Poly (Đường cong đa thức)", "Sigmoid (Hình chữ S)"],
        "Ưu điểm": ["Cực nhanh, tốt cho 2D/3D đơn giản", "Vô địch trong không gian phức tạp", "Tốt cho dữ liệu có tính chu kỳ", "Giống mô phỏng nơ-ron"],
        "Tình trạng": ["Đang dùng (Best)" if getattr(svm_model, 'kernel', '') == 'linear' else "Ứng viên",
                      "Đang dùng (Best)" if getattr(svm_model, 'kernel', '') == 'rbf' else "Ứng viên",
                      "Đang dùng (Best)" if getattr(svm_model, 'kernel', '') == 'poly' else "Ứng viên",
                      "Đang dùng (Best)" if getattr(svm_model, 'kernel', '') == 'sigmoid' else "Ứng viên"]
    })
    st.table(candidates)

    # --- PHẦN 2: PHÂN TÍCH XÁC SUẤT ---
    st.markdown("### 📊 Hội đồng Xét duyệt: Top 5 Ứng cử viên")
    
    # Khởi tạo giá trị mặc định để tránh lỗi UnboundLocalError
    top_5_names = [class_names.get(current_prediction_id, f"Nhãn {current_prediction_id}")]
    top_5_probs = [1.0]
    top_5_indices = [current_prediction_id]
    
    try:
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(scaled_features)[0]
        else:
            scores = svm_model.decision_function(scaled_features)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            st.info("ℹ️ Hệ thống dùng thuật toán Khoảng cách (Decision Function) để ước tính xác suất.")

        top_5_indices = np.argsort(probs)[-5:][::-1]
        top_5_probs = probs[top_5_indices]
        top_5_names = [class_names.get(idx, f"Nhãn {idx}") for idx in top_5_indices]
        
        col_chart, col_explain = st.columns([0.6, 0.4])
        
        with col_chart:
            fig_p, ax_p = plt.subplots(figsize=(8, 4))
            bars = ax_p.barh(top_5_names, top_5_probs * 100, color='#28a745')
            ax_p.set_xlabel("Độ tin cậy (%)")
            ax_p.invert_yaxis()
            for bar in bars:
                width = bar.get_width()
                ax_p.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
            st.pyplot(fig_p)
            
        with col_explain:
            st.info(f"🌟 **Lựa chọn cuối cùng:** **{top_5_names[0]}**")
            if len(top_5_probs) > 1 and top_5_probs[1] > 0.05:
                st.warning(f"💡 **AI đang phân vân?** Có vẻ biển báo này cũng khá giống với **{top_5_names[1]}**.")

    except Exception as e:
        st.error(f"⚠️ Lỗi khi phân tích xác suất: {str(e)}")

    # --- PHẦN 3: CUỘC TỬ CHIẾN GIỮA TOP 1 VÀ TOP 2 (OVO TRACE) ---
    if len(top_5_indices) >= 2:
        st.markdown(f"### ⚔️ Cuộc tử chiến: {top_5_names[0]} vs {top_5_names[1]}")
        
        # Nếu là Linear, ta trích xuất được 'Nhân chứng' (Coefficients)
        if getattr(svm_model, 'kernel', '') == 'linear' and hasattr(svm_model, 'coef_'):
            with st.spinner("Đang trích xuất lời khai của các nhân chứng (Genes)..."):
                # Tính toán Duel Index (One-vs-One)
                i, j = sorted([top_5_indices[0], top_5_indices[1]])
                n_classes = len(svm_model.classes_)
                # Index trong OVO của sklearn: i * (n_classes - 1) - i*(i+1)/2 + j - 1
                duel_idx = int(i * (n_classes - 1) - i * (i + 1) / 2 + j - 1)
                
                # Trích xuất trọng số của 256 Gene cho cuộc đối đầu này
                weights = svm_model.coef_[duel_idx]
                # Tính 'Sự đóng góp' thực tế: w * x
                contributions = weights * scaled_features.flatten()
                
                # Nếu i là top 1, thì giá trị dương ủng hộ i, âm ủng hộ j.
                # Nếu i là top 2, thì ngược lại. Ta chuẩn hóa để Top 1 luôn là dương.
                if i != top_5_indices[0]:
                    contributions = -contributions
                
                # Lấy Top 10 Gene có ảnh hưởng mạnh nhất (cả âm và dương)
                top_10_indices_cont = np.argsort(np.abs(contributions))[-10:]
                top_10_vals = contributions[top_10_indices_cont]
                top_10_labels = [f"Gene #{idx}" for idx in top_10_indices_cont]
                
                fig_c, ax_c = plt.subplots(figsize=(10, 5))
                colors_c = ['#28a745' if v > 0 else '#dc3545' for v in top_10_vals]
                ax_c.bar(top_10_labels, top_10_vals, color=colors_c)
                ax_c.axhline(0, color='black', linewidth=0.8)
                ax_c.set_title(f"Các Gene 'Nhân chứng' quyết định kết quả giữa {top_5_names[0]} và {top_5_names[1]}")
                ax_c.set_ylabel("Mức độ thuyết phục (W * X)")
                st.pyplot(fig_c)
                
                st.caption(f"💡 **Giải thích:** Gene màu xanh ủng hộ **{top_5_names[0]}**, Gene màu đỏ ủng hộ **{top_5_names[1]}**. Tổng hợp lại, xanh thắng đỏ!")
        else:
            st.info("ℹ️ Vì mô hình không dùng Linear Kernel, AI dùng thuật toán khoảng cách phi tuyến (RBF/Poly) để phân biệt dựa trên mật độ dữ liệu.")

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
