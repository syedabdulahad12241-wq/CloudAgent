"""
Streamlit Web Application for Deep Learning Vision Decision Agent
Deployable on Streamlit Community Cloud (share.streamlit.io) or locally.
"""

import io
import base64
import streamlit as st
from PIL import Image, ImageDraw

from agent.decision_agent import DecisionAgent

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Vision Decision Agent | YOLO + ReAct",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Cached Agent Initializer (Loads YOLO model once)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 Deep Learning Model...")
def get_agent():
    return DecisionAgent(yolo_model="yolov8n.pt", yolo_conf=0.25)

agent = get_agent()

# -------------------------------------------------------------
# Custom Styling
# -------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #9333ea, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .decision-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.markdown('<div class="main-title">🧠 Vision Decision Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Agentic AI that takes photos, runs YOLOv8 Deep Learning, deduces <b>what it is</b>, and makes actionable decisions.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar: System Controls & Presets
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    goal_input = st.text_input(
        "Agent Objective / Question",
        value="What is this photo and what decision should be taken?"
    )
    
    conf_threshold = st.slider(
        "YOLO Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )
    agent.yolo_tool.conf_threshold = conf_threshold

    st.markdown("---")
    st.subheader("🚀 Quick Test Scenarios")
    sample_choice = st.selectbox(
        "Or pick a preset scene:",
        ["None (Upload your own)", "Workspace Scene", "Traffic Scene", "Nature Scene"]
    )

    st.markdown("---")
    st.caption("Architecture: YOLOv8 + ReAct Decision Engine + Streamlit")

# -------------------------------------------------------------
# Image Ingestion (Upload, Camera, or Preset)
# -------------------------------------------------------------
input_col, action_col = st.columns([3, 1])

pil_image = None

with input_col:
    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📸 Live Camera"])
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload photo (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")

    with tab_camera:
        camera_photo = st.camera_input("Take a photo with your webcam")
        if camera_photo is not None:
            pil_image = Image.open(camera_photo).convert("RGB")

# Handle Presets if selected
if sample_choice != "None (Upload your own)" and pil_image is None:
    w, h = 640, 480
    pil_image = Image.new("RGB", (w, h), color=(240, 243, 246))
    draw = ImageDraw.Draw(pil_image)

    if sample_choice == "Workspace Scene":
        draw.rectangle([0, 0, w, 280], fill=(225, 230, 235))
        draw.rectangle([0, 280, w, h], fill=(160, 140, 120))
        draw.rectangle([200, 180, 440, 320], fill=(50, 50, 50))
        draw.rectangle([210, 190, 430, 310], fill=(70, 130, 180))
        draw.rectangle([180, 320, 460, 340], fill=(90, 90, 90))
        draw.rectangle([480, 260, 540, 330], fill=(210, 60, 60))
    elif sample_choice == "Traffic Scene":
        draw.rectangle([0, 0, w, 220], fill=(135, 206, 235))
        draw.rectangle([0, 220, w, h], fill=(80, 80, 80))
        draw.rectangle([180, 260, 460, 360], fill=(220, 40, 40))
        draw.rectangle([240, 210, 400, 260], fill=(200, 30, 30))
        draw.ellipse([210, 340, 270, 400], fill=(30, 30, 30))
        draw.ellipse([370, 340, 430, 400], fill=(30, 30, 30))
    else: # Nature
        draw.rectangle([0, 0, w, 260], fill=(176, 224, 230))
        draw.rectangle([0, 260, w, h], fill=(60, 140, 60))
        draw.ellipse([240, 200, 400, 360], fill=(139, 69, 19))
        draw.ellipse([340, 160, 420, 240], fill=(139, 69, 19))

# -------------------------------------------------------------
# Run Decision Agent
# -------------------------------------------------------------
if pil_image is not None:
    st.markdown("---")
    
    with st.spinner("🤖 Agent is inspecting photo, invoking YOLO, and synthesizing decision..."):
        result = agent.process_photo(pil_image, user_goal=goal_input)

    # 1. Visual Comparison Row
    col_raw, col_yolo = st.columns(2)
    
    with col_raw:
        st.subheader("📷 Raw Input Photo")
        st.image(pil_image, use_container_width=True)

    with col_yolo:
        st.subheader("🎯 YOLO Deep Learning Bounding Boxes")
        annotated_b64 = result.get("annotated_image", "")
        if annotated_b64.startswith("data:image"):
            header, encoded = annotated_b64.split(",", 1)
            annotated_bytes = base64.b64decode(encoded)
            st.image(annotated_bytes, use_container_width=True)
        else:
            st.image(pil_image, use_container_width=True)

    # 2. Agent Decision Result Card
    st.markdown("---")
    st.subheader("📋 Autonomous Agent Decision")

    m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
    with m_col1:
        st.markdown(f"**Domain Category:** `{result.get('category', 'General')}`")
        st.markdown(f"### 🔍 What it is:\n**{result.get('primary_identification', 'Subject detected')}**")
    with m_col2:
        conf_pct = int(result.get("confidence", 0.8) * 100)
        st.metric("Agent Confidence", f"{conf_pct}%")
    with m_col3:
        total_objs = len(result.get("detected_entities", []))
        st.metric("Total Objects Found", total_objs)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.info(f"⚖️ **Autonomous Decision:**\n\n{result.get('decision', 'N/A')}")
    with d_col2:
        st.success(f"🚀 **Prescribed Action:**\n\n{result.get('action_recommendation', 'N/A')}")

    # 3. Agent Reasoning Trace (ReAct)
    with st.expander("🧠 View Agent Step-by-Step Reasoning Trace (ReAct)", expanded=True):
        for step in result.get("reasoning_steps", []):
            tool_tag = f" `[Tool: {step['tool']}]`" if step.get("tool") else ""
            st.markdown(f"**Step {step['step']}: {step['phase']}**{tool_tag}")
            st.caption(step['thought'])

    # 4. Detected Entities Breakdown Table
    if result.get("detected_entities"):
        with st.expander("🏷️ Detected Objects & Confidence Breakdown"):
            st.table([
                {
                    "Object": d["label"].capitalize(),
                    "Confidence": f"{int(d['confidence'] * 100)}%",
                    "Location": d.get("location", "center"),
                    "Prominent": "Yes" if d.get("is_prominent") else "No"
                }
                for d in result.get("detected_entities", [])
            ])

else:
    st.info("👆 Upload an image or select a preset scene on the left sidebar to start the Agent Decision workflow!")
