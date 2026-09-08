"""
Agentic AI Decision Engine
Orchestrates deep learning tools (YOLO, scene classifiers), analyzes visual inputs,
and makes autonomous decisions regarding 'what it is' and appropriate actions.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from PIL import Image

from tools.yolo_tool import YOLOTool
from tools.scene_classifier import SceneClassifierTool

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Autonomous Vision Decision Agent.
    Executes a multi-stage ReAct cycle (Observe -> Tool Use -> Synthesize -> Decide).
    """

    def __init__(self, yolo_model: str = "yolov8n.pt", yolo_conf: float = 0.25):
        self.yolo_tool = YOLOTool(model_name=yolo_model, conf_threshold=yolo_conf)
        self.scene_tool = SceneClassifierTool()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    def process_photo(self, image_input: Any, user_goal: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a photo, determine what it is, and make an autonomous decision.

        Args:
            image_input: Image file path, PIL Image, base64 string, or bytes.
            user_goal: Optional prompt or goal (e.g. "Is it safe?", "Identify item for inventory").

        Returns:
            Structured decision response containing:
              - primary_identification: Clear statement of what the image depicts.
              - decision: Autonomous decision or state evaluation.
              - action_recommendation: Prescribed next action.
              - category: Broad domain category.
              - confidence: Confidence percentage (0.0 to 1.0).
              - reasoning_steps: Step-by-step trace of agent's thought process.
              - detected_entities: Object breakdown with bounding boxes.
              - scene_context: Environmental features.
              - annotated_image: Base64 image with YOLO bounding boxes.
        """
        reasoning_steps: List[Dict[str, Any]] = []

        # Step 1: Perception and Goal Ingestion
        goal_text = user_goal or "Identify the subject of this photo and evaluate the scene."
        reasoning_steps.append({
            "step": 1,
            "phase": "Goal Formulation",
            "thought": f"Goal received: '{goal_text}'. Initiating computer vision perception pipeline.",
            "tool": None
        })

        # Step 2: Tool Execution - YOLO Deep Learning Object Detection
        reasoning_steps.append({
            "step": 2,
            "phase": "Tool Selection & Invocation",
            "thought": "Invoking YOLO deep learning detector to identify, locate, and bound objects in the scene.",
            "tool": "yolo_detector"
        })
        
        yolo_result = self.yolo_tool.detect(image_input)
        detections = yolo_result.get("detections", [])
        counts = yolo_result.get("counts", {})
        dominant_obj = yolo_result.get("dominant_object", "unknown")
        total_objects = yolo_result.get("total_objects", 0)

        reasoning_steps.append({
            "step": 3,
            "phase": "Tool Observation",
            "thought": f"YOLO detected {total_objects} entity(ies): {counts}. Dominant entity: '{dominant_obj}'.",
            "tool": "yolo_detector",
            "observation": {
                "total_detected": total_objects,
                "counts": counts,
                "dominant_object": dominant_obj,
                "density": yolo_result.get("scene_density")
            }
        })

        # Step 3: Tool Execution - Scene Classification
        pil_image = self.yolo_tool._load_image(image_input)
        scene_result = self.scene_tool.analyze_scene(pil_image)

        reasoning_steps.append({
            "step": 4,
            "phase": "Context Analysis",
            "thought": f"Scene tool assessed environment as '{scene_result['inferred_environment']}' under '{scene_result['lighting_condition']}' conditions.",
            "tool": "scene_classifier",
            "observation": scene_result
        })

        # Step 4: Decision Making & Semantic Synthesis
        reasoning_steps.append({
            "step": 5,
            "phase": "Decision Reasoning",
            "thought": "Synthesizing detected objects, spatial placement, and scene context to deduce 'what it is' and make an actionable decision.",
            "tool": None
        })

        # Synthesize identification and decision
        identification, category, decision, action, confidence = self._synthesize_decision(
            detections=detections,
            counts=counts,
            scene_info=scene_result,
            user_goal=goal_text
        )

        reasoning_steps.append({
            "step": 6,
            "phase": "Final Decision Formulated",
            "thought": f"Identification: '{identification}'. Decision: '{decision}'. Action: '{action}'.",
            "tool": None
        })

        return {
            "success": True,
            "primary_identification": identification,
            "decision": decision,
            "action_recommendation": action,
            "category": category,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
            "detected_entities": detections,
            "counts": counts,
            "scene_context": scene_result,
            "annotated_image": yolo_result.get("annotated_image_base64", ""),
            "model_metadata": {
                "detector": yolo_result.get("model_used"),
                "total_detected": total_objects
            }
        }

    def _synthesize_decision(
        self,
        detections: List[Dict[str, Any]],
        counts: Dict[str, int],
        scene_info: Dict[str, Any],
        user_goal: str
    ) -> tuple:
        """
        Multi-heuristic and semantic decision engine.
        Deduces what the subject is and what decision should be taken.
        """
        num_items = len(detections)
        env = scene_info.get("inferred_environment", "")
        lighting = scene_info.get("lighting_condition", "")

        # Case 0: Empty scene
        if num_items == 0:
            identification = f"Open {env} with no prominent foreground objects detected."
            category = "General Scene / Landscape"
            decision = "Clear field of view. No specific target objects identified."
            action = "Maintain monitoring or adjust camera angle/lighting."
            confidence = 0.65
            return identification, category, decision, action, confidence

        # Extract top prominent detection
        top_det = detections[0]
        top_label = top_det["label"].capitalize()
        top_conf = top_det["confidence"]

        # Check domain clusters
        is_human = "person" in counts
        is_vehicle = any(k in counts for k in ["car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat"])
        is_tech = any(k in counts for k in ["laptop", "cell phone", "mouse", "keyboard", "tv", "monitor"])
        is_animal = any(k in counts for k in ["dog", "cat", "bird", "horse", "sheep", "cow", "bear", "elephant"])
        is_furniture = any(k in counts for k in ["chair", "couch", "bed", "dining table"])
        is_food = any(k in counts for k in ["apple", "banana", "sandwich", "orange", "pizza", "donut", "cake", "bottle", "cup", "bowl"])

        # Semantic Synthesis Logic
        if is_human and is_tech:
            identification = f"Workplace / Computing Scene: Person with {', '.join([k for k in counts if k != 'person'])}."
            category = "Workplace & Technology"
            decision = "Active digital workstation environment detected. Normal office/study activity."
            action = "Log workstation state as active; permit continued operation."
            confidence = round(max(top_conf, 0.88), 2)

        elif is_human and is_vehicle:
            identification = f"Transit / Commute Scene: Person with {', '.join([k for k in counts if k in ['car', 'bicycle', 'motorcycle', 'truck']])}."
            category = "Transportation & Mobility"
            decision = "Pedestrian and vehicle interaction in progress. Caution recommended."
            action = "Enforce safety clearance; signal pedestrian awareness."
            confidence = round(max(top_conf, 0.89), 2)

        elif is_vehicle:
            vehicle_list = [f"{v} {k}" for k, v in counts.items() if k in ["car", "truck", "bus", "motorcycle", "bicycle", "boat"]]
            identification = f"Automotive / Vehicle Scene: {', '.join(vehicle_list)}."
            category = "Vehicular & Traffic"
            decision = "Vehicular presence detected on travel pathway."
            action = "Check speed restrictions and path clearance."
            confidence = round(top_conf, 2)

        elif is_animal:
            animal_names = [k for k in counts if k in ["dog", "cat", "bird", "horse", "sheep", "cow", "bear", "elephant"]]
            identification = f"Animal / Wildlife: Detected {', '.join(animal_names)} in {env}."
            category = "Biological & Pets"
            decision = f"Live animal ({animal_names[0]}) identified with {int(top_conf*100)}% confidence."
            action = "Record animal sighting; verify containment/welfare if in domestic area."
            confidence = round(top_conf, 2)

        elif is_food:
            food_items = [k for k in counts if k in ["apple", "banana", "sandwich", "orange", "pizza", "donut", "cake", "bottle", "cup", "bowl"]]
            identification = f"Culinary / Consumable: {', '.join(food_items)}."
            category = "Food & Refreshment"
            decision = "Consumable items identified in dining or preparation setting."
            action = "Categorize under food inventory; verify freshness or hygiene standards."
            confidence = round(top_conf, 2)

        elif is_furniture:
            furniture_items = [f"{v} {k}" for k, v in counts.items() if k in ["chair", "couch", "bed", "dining table"]]
            identification = f"Interior Furnishing: {', '.join(furniture_items)} in {lighting} setting."
            category = "Interior Design & Facilities"
            decision = "Furnished indoor zone identified. Area is static."
            action = "Record room layout and furniture occupancy."
            confidence = round(top_conf, 2)

        elif is_tech:
            tech_items = [f"{v} {k}" for k, v in counts.items() if k in ["laptop", "cell phone", "mouse", "keyboard", "tv", "monitor"]]
            identification = f"Electronic Equipment: {', '.join(tech_items)}."
            category = "Hardware & Electronics"
            decision = "Valuable IT/electronic hardware identified."
            action = "Catalog serial/inventory records; ensure secure handling."
            confidence = round(top_conf, 2)

        else:
            # General single or multi-object identification
            if num_items == 1:
                identification = f"{top_label} located in the {top_det.get('location', 'center')} of the view."
            else:
                summary_items = [f"{v} {k}" for k, v in list(counts.items())[:3]]
                identification = f"Multiple items: {', '.join(summary_items)} situated in {env}."
            
            category = "General Object Identification"
            decision = f"Identified primary entity as '{top_label}' ({int(top_conf * 100)}% confidence)."
            action = f"Proceed with processing '{top_label}' according to operational guidelines."
            confidence = round(top_conf, 2)

        return identification, category, decision, action, confidence
