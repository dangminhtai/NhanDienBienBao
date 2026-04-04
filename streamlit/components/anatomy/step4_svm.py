import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def render_svm_classification(svm_model, scaled_features, class_names, current_prediction_id, current_dir):
    """Mổ xẻ Bước 4: Ra quyết định cuối cùng (SVM Classification) - Pedagogical Edition."""
    st.markdown("---")
    st.header("🧠 Bước 4: Ra quyết định (Phân loại SVM)")
    
    # 1. Hướng dẫn lý thuyết & Bản kế hoạch
    plan4_path = os.path.join(current_dir, "docs", "predict", "plans", "plan4.md")
    if os.path.exists(plan4_path):
        with st.expander("📑 Xem Bản kế hoạch Giải phẫu Bước 4", expanded=False):
            with open(plan4_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
    
    step4_path = os.path.join(current_dir, "docs", "predict", "steps", "step4.md")
    if os.path.exists(step4_path):
        with open(step4_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # --- TIỂU BƯỚC 4.1: TIẾP NHẬN ĐẶC TRƯNG ---
    st.subheader("4.1. Tiếp nhận Đặc trưng Định danh")
    st.markdown(f"""
    Tại bước này, SVM tiếp nhận **256 mã số** (mã Gene) đã được CNN trích xuất và chuẩn hóa. 
    Đây là "bản tóm tắt" tinh túy nhất về hình dáng, màu sắc và chi tiết của biển báo.
    """)
    with st.expander("🔍 Xem vector đặc trưng (256 chiều)", expanded=False):
        st.write(scaled_features)

    # --- TIỂU BƯỚC 4.2: THIẾT LẬP RANH GIỚI QUYẾT ĐỊNH ---
    st.subheader("4.2. Lựa chọn Cơ chế Phán quyết (Kernel)")
    st.markdown("""
    Để phân loại chính xác, hệ thống phải chọn ra loại "thước đo" phù hợp nhất. 
    Dựa trên quá trình thử nghiệm tự động, hệ thống đã xác định cơ chế tối ưu cho dữ liệu này:
    """)
    
    # Hiển thị bảng ứng cử viên chuyên nghiệp
    candidates = pd.DataFrame({
        "Cơ chế (Kernel)": ["Linear (Tuyến tính)", "RBF (Phi tuyến Radial)", "Poly (Đa thức)", "Sigmoid"],
        "Mô tả": ["Tìm ranh giới bằng các đường thẳng", "Tạo ranh giới bao quanh các cụm dữ liệu", "Sử dụng các đường cong phức tạp", "Mô phỏng hàm kích hoạt nơ-ron"],
        "Trạng thái": ["💎 Đang sử dụng" if getattr(svm_model, 'kernel', '') == 'linear' else "Ứng viên",
                      "💎 Đang sử dụng" if getattr(svm_model, 'kernel', '') == 'rbf' else "Ứng viên",
                      "💎 Đang sử dụng" if getattr(svm_model, 'kernel', '') == 'poly' else "Ứng viên",
                      "💎 Đang sử dụng" if getattr(svm_model, 'kernel', '') == 'sigmoid' else "Ứng viên"]
    })
    st.table(candidates)

    # --- TIỂU BƯỚC 4.3: PHÂN TÍCH CHỨNG CỨ (FEATURE WEIGHTING) ---
    st.subheader("4.3. Đối soát Chứng cứ (Feature Analysis)")
    
    # Khởi tạo giá trị mặc định cho logic OVO
    top_5_names = [class_names.get(current_prediction_id, f"Nhãn {current_prediction_id}")]
    top_5_probs = [1.0]
    top_5_indices = [current_prediction_id]
    
    try:
        # Lấy xác suất hoặc điểm số quyết định
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(scaled_features)[0]
        else:
            scores = svm_model.decision_function(scaled_features)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

        top_5_indices = np.argsort(probs)[-5:][::-1]
        top_5_probs = probs[top_5_indices]
        top_5_names = [class_names.get(idx, f"Nhãn {idx}") for idx in top_5_indices]

        if len(top_5_indices) >= 2 and getattr(svm_model, 'kernel', '') == 'linear' and hasattr(svm_model, 'coef_'):
            st.markdown(f"**Cuộc đối đầu quyết định:** '{top_5_names[0]}' so với '{top_5_names[1]}'")
            
            # Tính toán Duel Index
            i, j = sorted([top_5_indices[0], top_5_indices[1]])
            n_classes = len(svm_model.classes_)
            duel_idx = int(i * (n_classes - 1) - i * (i + 1) / 2 + j - 1)
            
            weights = svm_model.coef_[duel_idx]
            contributions = weights * scaled_features.flatten()
            if i != top_5_indices[0]:
                contributions = -contributions
            
            top_10_indices_cont = np.argsort(np.abs(contributions))[-10:]
            top_10_vals = contributions[top_10_indices_cont]
            top_10_labels = [f"Yếu tố #{idx}" for idx in top_10_indices_cont]
            
            fig_c, ax_c = plt.subplots(figsize=(10, 4))
            colors_c = ['#28a745' if v > 0 else '#dc3545' for v in top_10_vals]
            ax_c.bar(top_10_labels, top_10_vals, color=colors_c)
            ax_c.axhline(0, color='black', linewidth=0.8)
            ax_c.set_title("Top 10 yếu tố định đoạt kết quả")
            ax_c.set_ylabel("Mức độ thuyết phục")
            st.pyplot(fig_c)
            st.caption(f"💡 Cột màu xanh ủng hộ biển báo hiện tại, cột màu đỏ ủng hộ phương án dự phòng. Kết quả chung cuộc: Màu xanh chiếm ưu thế.")
    except Exception as e:
        st.error(f"⚠️ Lỗi phân tích chứng cứ: {str(e)}")

    # --- TIỂU BƯỚC 4.4: PHÁN QUYẾT CUỐI CÙNG ---
    st.subheader("4.4. Phán quyết Cuối cùng")
    
    col_chart, col_res = st.columns([0.6, 0.4])
    with col_chart:
        fig_p, ax_p = plt.subplots(figsize=(8, 4))
        bars = ax_p.barh(top_5_names, top_5_probs * 100, color='#28a745')
        ax_p.set_xlabel("Độ tin cậy (%)")
        ax_p.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            ax_p.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
        st.pyplot(fig_p)

    with col_res:
        st.success(f"""
        🏆 **KẾT LUẬN:**
        Biển báo được xác định là:
        **{top_5_names[0].upper()}**
        
        Độ tin cậy: **{top_5_probs[0]*100:.2f}%**
        """)

    # Minh họa không gian (Conceptual)
    st.markdown("### 🗺️ Bản đồ Không gian Quyết định")
    fig_map, ax_map = plt.subplots(figsize=(8, 5))
    np.random.seed(42)
    # Vùng cho Top 1 (Green)
    points_t1 = np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2))
    # Vùng cho Top 2 (Orange)
    points_t2 = np.random.normal(loc=[-2, 1], scale=0.6, size=(25, 2))
    ax_map.scatter(points_t1[:,0], points_t1[:,1], c='green', alpha=0.2, label=f"Vùng '{top_5_names[0]}'")
    ax_map.scatter(points_t2[:,0], points_t2[:,1], c='orange', alpha=0.2, label=f"Vùng '{top_5_names[1]}'")
    
    # Vị trí hiện tại
    current_pos = [2.2 - (1 - top_5_probs[0])*4, 2.2 - (1 - top_5_probs[0])*2]
    ax_map.scatter(current_pos[0], current_pos[1], c='red', s=250, marker='*', label="Tọa độ Ảnh của Bạn", edgecolors='black')
    
    ax_map.legend(loc='lower right')
    ax_map.set_title("Vị trí của tấm ảnh trong Không gian Đặc trưng")
    st.pyplot(fig_map)
