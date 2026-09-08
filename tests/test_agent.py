"""
Unit and Integration Tests for Vision Decision Agent
"""

import unittest
from PIL import Image, ImageDraw

from tools.yolo_tool import YOLOTool
from tools.scene_classifier import SceneClassifierTool
from agent.decision_agent import DecisionAgent


class TestVisionDecisionAgent(unittest.TestCase):

    def setUp(self):
        # Create a synthetic image representing a test scene (table + cup)
        self.test_img = Image.new("RGB", (320, 240), color=(240, 240, 240))
        draw = ImageDraw.Draw(self.test_img)
        # Background
        draw.rectangle([0, 150, 320, 240], fill=(140, 110, 80))
        # Cup shape
        draw.rectangle([120, 110, 190, 170], fill=(220, 50, 50))
        draw.rectangle([140, 80, 170, 110], fill=(200, 40, 40))

    def test_scene_classifier(self):
        tool = SceneClassifierTool()
        result = tool.analyze_scene(self.test_img)
        self.assertIn("lighting_condition", result)
        self.assertIn("orientation", result)
        self.assertIn("inferred_environment", result)
        self.assertEqual(result["orientation"], "landscape")

    def test_yolo_tool_detection(self):
        yolo = YOLOTool(model_name="yolov8n.pt")
        result = yolo.detect(self.test_img)
        self.assertIn("detections", result)
        self.assertIn("counts", result)
        self.assertIn("annotated_image_base64", result)
        self.assertTrue(result["annotated_image_base64"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(result["image_dimensions"]["width"], 320)
        self.assertEqual(result["image_dimensions"]["height"], 240)

    def test_decision_agent_reasoning_pipeline(self):
        agent = DecisionAgent(yolo_model="yolov8n.pt")
        res = agent.process_photo(self.test_img, user_goal="Identify what this is and make a decision.")
        
        # Verify required structured keys
        self.assertTrue(res["success"])
        self.assertIn("primary_identification", res)
        self.assertIn("decision", res)
        self.assertIn("action_recommendation", res)
        self.assertIn("category", res)
        self.assertIn("confidence", res)
        self.assertIn("reasoning_steps", res)
        self.assertIn("annotated_image", res)
        
        # Verify reasoning steps structure
        self.assertGreaterEqual(len(res["reasoning_steps"]), 4)
        for step in res["reasoning_steps"]:
            self.assertIn("step", step)
            self.assertIn("phase", step)
            self.assertIn("thought", step)

    def test_agent_human_tech_decision(self):
        # Test synthetic decision logic with mock detections
        agent = DecisionAgent()
        mock_detections = [
            {"label": "person", "confidence": 0.94, "bbox": [100, 50, 200, 250], "location": "center"},
            {"label": "laptop", "confidence": 0.91, "bbox": [120, 200, 220, 270], "location": "center"}
        ]
        mock_counts = {"person": 1, "laptop": 1}
        mock_scene = {"inferred_environment": "indoor_warm", "lighting_condition": "well_lit"}
        
        ident, cat, decision, action, conf = agent._synthesize_decision(
            detections=mock_detections,
            counts=mock_counts,
            scene_info=mock_scene,
            user_goal="Assess scene"
        )
        self.assertEqual(cat, "Workplace & Technology")
        self.assertIn("laptop", ident)
        self.assertGreaterEqual(conf, 0.85)


if __name__ == "__main__":
    unittest.main()
