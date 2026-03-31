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
        Norm["🛠️ Chuẩn hóa / 255.0"]
        Input --> Norm

        %% --- KHỐI CONV 1 (TẦM SOÁT CẠNH) ---
        subgraph "🧱 2.1: KHỐI CONV 1 (Sơ cấp)"
            Norm --> C1["🔍 Conv2D (32 Filter, 3x3) <br/> 30x30x32"]
            C1 --> R1["⚡ ReLU Activation"]
            R1 --> C2["🔍 Conv2D (64 Filter, 3x3) <br/> 28x28x64"]
            C2 --> R2["⚡ ReLU Activation"]
            R2 --> P1["🧹 MaxPooling (2x2) <br/> 14x14x64"]
            P1 --> D1["🚿 Dropout (0.25)"]
        end

        %% --- KHỐI CONV 2 (SOI CHI TIẾT) ---
        subgraph "🧱 2.2: KHỐI CONV 2 (Trung cấp)"
            D1 --> C3["🧐 Conv2D (32 Filter, 3x3) <br/> 12x12x32"]
            C3 --> R3["⚡ ReLU Activation"]
            R3 --> C4["🧐 Conv2D (64 Filter, 3x3) <br/> 10x10x64"]
            C4 --> R4["⚡ ReLU Activation"]
            R4 --> P2["🧹 MaxPooling (2x2) <br/> 5x5x64"]
            P2 --> D2["🚿 Dropout (0.25)"]
        end

        %% --- BƯỚC NÉN (FLATTEN & DENSE) ---
        subgraph "🧬 2.3: BƯỚC NÉN (Định danh)"
            D2 --> Flat["📏 Flatten (Duỗi thẳng) <br/> 1,600 con số"]
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
