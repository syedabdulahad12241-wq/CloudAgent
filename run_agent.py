"""
CLI Runner for Deep Learning Vision Decision Agent
Run: python run_agent.py --image path/to/photo.jpg [--goal "Custom question"]
"""

import os
import sys
import argparse
import json
from PIL import Image

from agent.decision_agent import DecisionAgent


def main():
    parser = argparse.ArgumentParser(description="Autonomous Deep Learning Vision Decision Agent")
    parser.add_argument("--image", type=str, help="Path to input photo/image")
    parser.add_argument("--goal", type=str, default="What is this photo and what decision should be taken?", help="Agent objective or question")
    parser.add_argument("--sample", action="store_true", help="Run on a generated sample image if no image provided")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model checkpoint (default: yolov8n.pt)")
    parser.add_argument("--output-json", type=str, default=None, help="Save structured decision output to JSON file")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DEEP LEARNING VISION DECISION AGENT (YOLO + ReAct)")
    print("=" * 60)

    # Initialize agent
    print(f"[*] Initializing Decision Agent with model: {args.model}...")
    agent = DecisionAgent(yolo_model=args.model)

    image_source = None
    if args.image:
        if not os.path.exists(args.image):
            print(f"[-] Error: File not found at '{args.image}'")
            sys.exit(1)
        image_source = args.image
        print(f"[*] Loading photo: {args.image}")
    else:
        print("[*] No image path supplied, generating sample workspace image...")
        from PIL import ImageDraw
        img = Image.new("RGB", (640, 480), color=(235, 238, 240))
        draw = ImageDraw.Draw(img)
        # Draw desk & laptop
        draw.rectangle([0, 260, 640, 480], fill=(160, 130, 100))
        draw.rectangle([200, 160, 440, 300], fill=(40, 40, 40))
        draw.rectangle([210, 170, 430, 290], fill=(60, 120, 190))
        draw.rectangle([180, 300, 460, 320], fill=(80, 80, 80))
        image_source = img

    print(f"[*] Agent Goal: '{args.goal}'")
    print("[*] Processing photo through perception tools and reasoning loop...")

    result = agent.process_photo(image_source, user_goal=args.goal)

    print("\n" + "-" * 60)
    print("  AGENT REASONING TRACE")
    print("-" * 60)
    for step in result["reasoning_steps"]:
        tool_tag = f" [Tool: {step['tool']}]" if step.get("tool") else ""
        print(f"[{step['step']}] {step['phase']}{tool_tag}:")
        print(f"    {step['thought']}")

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AGENT DECISION RESULT")
    print("=" * 60)
    print(f"  WHAT IT IS:     {result['primary_identification']}")
    print(f"  CATEGORY:       {result['category']}")
    print(f"  CONFIDENCE:     {int(result['confidence'] * 100)}%")
    print(f"  DECISION:       {result['decision']}")
    print(f"  PRESCRIBED ACT: {result['action_recommendation']}")
    print(f"  OBJECT COUNTS:  {result['counts']}")
    print("=" * 60 + "\n")

    if args.output_json:
        # Exclude base64 image from file dump if requested to keep it concise
        clean_result = dict(result)
        clean_result["annotated_image"] = "<base64 omitted>"
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, indent=2)
        print(f"[+] Structured output saved to: {args.output_json}")


if __name__ == "__main__":
    main()
