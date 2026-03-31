import streamlit as st
import os

def render_cnn_overview(raw_ndarray, current_dir):
    """Bản đồ tổng quan và Chuẩn hóa (Normalization)."""
    # 1. Hiển thị nội dung plan2.md
    plan2_path = os.path.join(current_dir, "docs", "predict", "plans", "plan2.md")
    if os.path.exists(plan2_path):
        with open(plan2_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # 2.0: Bản đồ dòng chảy (Architecture Map - Mermaid Edition)
    st.markdown("---")
    st.markdown("### 🗺️ 2.0: Bản đồ Tổng quan (Full Architecture Map)")
    
    # ... (Mermaid code remains the same)
    mermaid_code = """
    graph TD
        %% --- BƯỚC 1 ---
        Input["🖼️ 1. Ảnh Biển Báo (32x32x3)"]
        Norm["🛠️ 2.1.1: Chuẩn hóa / 255.0"]
        Input --> Norm

        %% --- KHỐI CONV 1 (TẦM SOÁT CẠNH) ---
        subgraph "🧱 2.1: KHỐI CONV 1 (Sơ cấp)"
            Norm --> C1["🔍 2.1.2: Conv2D_1 <br/> 30x30x32"]
            C1 --> R1["⚡ 2.1.3: ReLU_1"]
            R1 --> C2["🔍 2.1.4: Conv2D_2 <br/> 28x28x64"]
            C2 --> R2["⚡ ReLU_2"]
            R2 --> P1["🧹 2.1.5: Pool_1 <br/> 14x14x64"]
            P1 --> D1["🚿 Dropout"]
        end

        %% --- KHỐI CONV 2 (SOI CHI TIẾT) ---
        subgraph "🧱 2.2: KHỐI CONV 2 (Trung cấp)"
            D1 --> C3["🧐 Conv2D_3 <br/> 12x12x32"]
            C3 --> R3["⚡ ReLU_3"]
            R3 --> C4["🧐 Conv2D_4 <br/> 10x10x64"]
            C4 --> R4["⚡ ReLU_4"]
            R4 --> P2["🧹 Pool_2 <br/> 5x5x64"]
            P2 --> D2["🚿 Dropout"]
        end

        %% --- BƯỚC NÉN (FLATTEN & DENSE) ---
        subgraph "🧬 2.3: BƯỚC NÉN (Định danh)"
            D2 --> Flat["📏 Flatten <br/> 1,600 số"]
            Flat --> Dense["🧬 Dense (feature_dense) <br/> 256 Mã Gene"]
        end

        %% --- KẾT QUẢ ---
        Dense --> Output["💎 Vector Đặc Trưng Deep Feature (256,)"]

        %% --- STYLE ---
        style Input fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
        style Output fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
        style Dense fill:#a5d6a7,stroke:#388e3c,stroke-width:3px
        style C1 fill:#e3f2fd,stroke:#2196f3
        style C2 fill:#e3f2fd,stroke:#2196f3
        style C3 fill:#e3f2fd,stroke:#2196f3
        style C4 fill:#e3f2fd,stroke:#2196f3
        style P1 fill:#c8e6c9,stroke:#388e3c
        style P2 fill:#c8e6c9,stroke:#388e3c
    """
    
    # Render Mermaid (Using components.html as before)
    import streamlit.components.v1 as components
    html = f"""
    <div id="mermaid-diag" class="mermaid" style="display: flex; justify-content: center;">
        {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    components.html(html, height=800, scrolling=True)

    # 2.1: KHỐI CONV 1 (Sơ cấp)
    st.markdown("---")
    st.header("🧱 2.1: KHỐI CONV 1 (Sơ cấp)")
    
    # 2.1.1: Tracking chuẩn hóa (Normalization)
    st.markdown("### 🛠️ 2.1.1: Phép chia Chuẩn hóa (Dữ liệu vào CNN)")
    normalized_rows = (raw_ndarray[0:3, :, 0]) / 255.0
    st.code(f"""
# Công thức: raw_ndarray / 255.0
# Kết quả: 3 dòng đầu (Full 32 cột) - Kênh màu Đỏ

{normalized_rows}
    """, language="python")
    st.caption("Dữ liệu lúc này đã là số thực [0, 1]. Đây là 'thức ăn' chuẩn cho các Nơ-ron.")

    st.info("""
    **Hành trình "Gạn đục khơi trong":**
    1. **Khối 1 & 2:** Biến đổi các Pixel vô hồn thành những khối đặc trưng hình học (cạnh, nét, hình dáng).
    2. **Flatten:** Xếp chồng 1.600 đặc trưng nhỏ lẻ lại thành một bản danh sách dài.
    3. **Dense:** "Đúc" danh sách 1.600 mục đó thành đúng **256 từ khóa then chốt** để SVM nhận diện.
    """)
