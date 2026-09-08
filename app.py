"""
FastAPI Deployment Server for Vision Decision Agent
Exposes REST APIs and serves an interactive web dashboard for real-time image analysis.
"""

import io
import os
import base64
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw

from agent.decision_agent import DecisionAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vision_agent_app")

app = FastAPI(
    title="Deep Learning Vision Decision Agent",
    description="Autonomous Agentic AI that takes photos, applies YOLO deep learning, and makes contextual decisions.",
    version="1.0.0"
)

# Enable CORS for external client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Decision Agent
agent = DecisionAgent(yolo_model="yolov8n.pt", yolo_conf=0.25)

# Pre-warm YOLO model so first request is instant
try:
    logger.info("Pre-warming YOLO model...")
    agent.yolo_tool._get_model()
except Exception as e:
    logger.warning(f"Model pre-warm warning: {e}")

# Ensure static folder exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class Base64AnalyzeRequest(BaseModel):
    image_base64: str
    goal: Optional[str] = "What is this photo and what decision should be taken?"


@app.get("/api/status")
def get_status():
    """System and Model Status Health Check."""
    is_yolo_ready = agent.yolo_tool._check_ultralytics()
    return {
        "status": "online",
        "agent": "VisionDecisionAgent-v1",
        "yolo_ready": is_yolo_ready,
        "active_model": agent.yolo_tool.model_name,
        "capabilities": [
            "YOLOv8 Deep Learning Object Detection",
            "Spatial Prominence & Layout Analysis",
            "Holistic Scene Context Extraction",
            "Multi-step Autonomous ReAct Decision Making",
            "Real-time Annotated Visual Rendering"
        ]
    }


@app.post("/api/analyze")
async def analyze_image_upload(
    file: UploadFile = File(...),
    goal: Optional[str] = Form(default="What is this photo and what decision should be taken?")
):
    """
    Analyze uploaded image file with YOLO and the Decision Agent.
    """
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Run agent reasoning cycle
        result = agent.process_photo(pil_image, user_goal=goal)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze-base64")
async def analyze_image_base64(payload: Base64AnalyzeRequest):
    """
    Analyze base64-encoded image (e.g. from webcam capture).
    """
    try:
        result = agent.process_photo(payload.image_base64, user_goal=payload.goal)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/sample-scene")
def generate_sample_scene(scene_type: str = "workspace"):
    """
    Generate a synthetic test image (e.g., workspace, car, animal) to test the agent immediately.
    """
    w, h = 640, 480
    img = Image.new("RGB", (w, h), color=(240, 243, 246))
    draw = ImageDraw.Draw(img)

    if scene_type == "workspace":
        # Background wall & desk
        draw.rectangle([0, 0, w, 280], fill=(225, 230, 235))
        draw.rectangle([0, 280, w, h], fill=(160, 140, 120))  # Wooden desk
        # Laptop shape
        draw.rectangle([200, 180, 440, 320], fill=(50, 50, 50))
        draw.rectangle([210, 190, 430, 310], fill=(70, 130, 180))
        draw.rectangle([180, 320, 460, 340], fill=(90, 90, 90))
        # Coffee cup
        draw.rectangle([480, 260, 540, 330], fill=(210, 60, 60))
        draw.ellipse([475, 255, 545, 270], fill=(230, 80, 80))
        title = "Synthetic Workstation Scene"
    elif scene_type == "traffic":
        # Road & sky
        draw.rectangle([0, 0, w, 220], fill=(135, 206, 235))
        draw.rectangle([0, 220, w, h], fill=(80, 80, 80))
        # Car body
        draw.rectangle([180, 260, 460, 360], fill=(220, 40, 40))
        draw.rectangle([240, 210, 400, 260], fill=(200, 30, 30))
        # Wheels
        draw.ellipse([210, 340, 270, 400], fill=(30, 30, 30))
        draw.ellipse([370, 340, 430, 400], fill=(30, 30, 30))
        title = "Synthetic Traffic Scene"
    else:
        # Nature & animal silhouette
        draw.rectangle([0, 0, w, 260], fill=(176, 224, 230))
        draw.rectangle([0, 260, w, h], fill=(60, 140, 60))
        draw.ellipse([240, 200, 400, 360], fill=(139, 69, 19))
        draw.ellipse([340, 160, 420, 240], fill=(139, 69, 19))
        title = "Synthetic Nature Scene"

    draw.text((20, 20), title, fill=(40, 40, 40))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    # Analyze synthetic scene
    result = agent.process_photo(img, user_goal="What is this scene and what action should be taken?")
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the interactive web interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Deep Learning Vision Decision Agent</h1><p>Index file not found in static/</p>")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"  Deep Learning Vision Decision Agent Server Launching")
    print(f"  Access Web UI at: http://127.0.0.1:{port}")
    print(f"  API Docs at:     http://127.0.0.1:{port}/docs")
    print(f"=======================================================\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
