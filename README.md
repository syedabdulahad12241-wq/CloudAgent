# Deep Learning Vision Decision Agent 🧠📷

An autonomous **Agentic AI** system that takes photos, applies deep learning computer vision tools (**YOLOv8** + Scene Context Extraction), deduces **"what it is"**, and makes contextual, actionable decisions.

---

## 🌟 Key Features

1. **Autonomous Decision Making**:
   - Implements a multi-phase **ReAct (Reasoning + Acting)** cycle:
     `Observe Photo` ➔ `Select & Invoke YOLO Detector` ➔ `Analyze Scene Context` ➔ `Synthesize Semantics` ➔ `Formulate Decision & Action`.
   - Explains **"what it is"** (e.g. Workplace workstation, Animal in nature, Pedestrian in transit, Electronic hardware, etc.).
   - Issues prescribed decisions (e.g., clearance checks, inventory logging, hazard alerts, occupancy state).

2. **Deep Learning Vision Tools**:
   - **Ultralytics YOLOv8**: Real-time object detection with bounding boxes, spatial localization, class labels, and confidence metrics.
   - **Scene Feature Analyzer**: Inferred environmental setting, lighting conditions, and composition analysis.
   - **Visual Annotation Engine**: High-contrast, color-coded bounding box overlays rendered in real-time.

3. **Deployable Web Application & API**:
   - **Interactive Web Dashboard**: Drag & drop photo upload, webcam snapshot mode, and 1-click test scenarios.
   - **Full ReAct Thought Log**: Displays the agent's step-by-step thinking process in real time.
   - **FastAPI Backend**: Async REST endpoints for production integration.
   - **Container Ready**: Includes `Dockerfile` and `docker-compose.yml`.

---

## 📁 Architecture

```
vision_decision_agent/
├── agent/
│   ├── __init__.py
│   └── decision_agent.py     # Core ReAct Decision Engine
├── tools/
│   ├── __init__.py
│   ├── yolo_tool.py          # YOLOv8 Deep Learning Tool & Annotator
│   └── scene_classifier.py   # Environmental & Feature Classifier
├── static/
│   └── index.html            # Modern Interactive Web Dashboard
├── tests/
│   └── test_agent.py         # Automated Test Suite
├── app.py                    # FastAPI Deployment Server
├── run_agent.py              # CLI Runner
├── requirements.txt          # Dependencies
├── Dockerfile                # Container Specification
└── docker-compose.yml        # Orchestration
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Deployment Server
```bash
python app.py
```
Open your browser at:
👉 **`http://127.0.0.1:8000`**

Interactive API documentation available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 💻 CLI Usage

You can also run the agent directly on any photo from the terminal:

```bash
# Analyze any image file
python run_agent.py --image "path/to/photo.jpg"

# Provide a specific goal or question
python run_agent.py --image "photo.jpg" --goal "Is there any obstacle on the road?"

# Run on a quick synthetic sample without supplying an image
python run_agent.py --sample
```

---

## 🐳 Docker Deployment

To deploy the agent as an isolated microservice with Docker:

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build with Docker directly
docker build -t vision-decision-agent .
docker run -p 8000:8000 vision-decision-agent
```

---

## 📡 REST API Reference

### `POST /api/analyze`
Upload a photo to receive the agent's decision and annotated visual.

**Request:** `multipart/form-data`
- `file`: Image file (JPG, PNG, WebP)
- `goal` (optional): Decision objective / question

**Response:**
```json
{
  "success": true,
  "primary_identification": "Workplace / Computing Scene: Person with laptop, mouse.",
  "decision": "Active digital workstation environment detected. Normal office/study activity.",
  "action_recommendation": "Log workstation state as active; permit continued operation.",
  "category": "Workplace & Technology",
  "confidence": 0.94,
  "reasoning_steps": [
    {
      "step": 1,
      "phase": "Goal Formulation",
      "thought": "Goal received: 'Identify the subject of this photo and evaluate the scene.'"
    },
    {
      "step": 2,
      "phase": "Tool Selection & Invocation",
      "thought": "Invoking YOLO deep learning detector to identify, locate, and bound objects in the scene.",
      "tool": "yolo_detector"
    },
    ...
  ],
  "detected_entities": [
    {
      "label": "person",
      "confidence": 0.94,
      "bbox": [120, 80, 410, 420],
      "location": "center",
      "is_prominent": true
    }
  ],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

---

## 🧪 Running Tests

Verify the tools and decision logic with:
```bash
python -m unittest tests/test_agent.py
```
