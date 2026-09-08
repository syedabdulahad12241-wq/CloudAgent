"""
Scene Classifier & Visual Feature Extractor Tool
Analyzes holistic scene properties: environment type, lighting, color palette, and visual context.
"""

from typing import Dict, Any
from PIL import Image, ImageStat


class SceneClassifierTool:
    """
    Analyzes holistic image characteristics to infer the setting, environment,
    lighting condition, and overall composition. Also provides deep learning
    fine-grained classification (e.g. Dog breed, Car model, Object category).
    """

    def __init__(self):
        self._classifier = None
        self._weights = None

    def classify_subject(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Deep learning ImageNet classifier (1000 categories, including exact dog breeds).
        Runs pure PyTorch without external libraries.
        """
        try:
            import torch
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            if self._classifier is None:
                self._weights = MobileNet_V3_Small_Weights.DEFAULT
                self._classifier = mobilenet_v3_small(weights=self._weights).eval()

            transform = self._weights.transforms()
            batch = transform(pil_image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probs = self._classifier(batch).squeeze(0).softmax(0)
            top_prob, top_catid = torch.topk(probs, 1)
            cat_name = self._weights.meta["categories"][top_catid[0].item()]
            
            clean_name = cat_name.replace("_", " ").title()
            return {
                "specific_name": clean_name,
                "classifier_confidence": round(float(top_prob[0].item()), 3)
            }
        except Exception:
            return {"specific_name": "", "classifier_confidence": 0.0}

    def analyze_scene(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Extract scene features from image.
        """
        img = pil_image.convert("RGB")
        w, h = img.size
        aspect_ratio = round(w / max(1, h), 2)
        orientation = "landscape" if aspect_ratio > 1.15 else ("portrait" if aspect_ratio < 0.85 else "square")

        # Color and brightness analysis
        stat = ImageStat.Stat(img)
        r_mean, g_mean, b_mean = stat.mean[:3]
        brightness = (r_mean * 299 + g_mean * 587 + b_mean * 114) / 1000

        if brightness < 60:
            lighting = "low_light / night"
        elif brightness < 120:
            lighting = "subdued / indoor"
        elif brightness < 200:
            lighting = "well_lit / daylight"
        else:
            lighting = "high_exposure / bright"

        # Color temperature & environmental cues
        inferred_environment = self._infer_environment(r_mean, g_mean, b_mean, brightness)

        return {
            "orientation": orientation,
            "aspect_ratio": aspect_ratio,
            "lighting_condition": lighting,
            "brightness_score": round(brightness, 1),
            "dominant_color_bias": self._color_bias(r_mean, g_mean, b_mean),
            "inferred_environment": inferred_environment
        }

    def _color_bias(self, r: float, g: float, b: float) -> str:
        if r > g + 25 and r > b + 25:
            return "warm_red_tint"
        elif g > r + 20 and g > b + 20:
            return "natural_green_foliage"
        elif b > r + 20 and b > g + 20:
            return "cool_blue_sky_water"
        elif abs(r - g) < 15 and abs(g - b) < 15:
            return "neutral_monochrome"
        else:
            return "balanced_mixed"

    def _infer_environment(self, r: float, g: float, b: float, brightness: float) -> str:
        if g > r + 15 and g > b:
            return "natural_outdoor (vegetation, park, or forest)"
        elif b > r + 15 and brightness > 120:
            return "open_outdoor (sky, aquatic, or coastal)"
        elif r > 130 and g > 110 and b < 100:
            return "indoor_warm (living room, office, or dining)"
        elif brightness < 70:
            return "night_or_enclosed_space"
        else:
            return "general_indoor_or_urban"
