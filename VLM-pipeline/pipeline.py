import os
import json
import time
import argparse
from pathlib import Path

import google.generativeai as genai
from PIL import Image


BODY_PARTS = [
    "Face & Jaw", "Neck", "Upper Chest", "Middle Chest", "Lower Chest",
    "Upper Abdomen", "Middle Abdomen", "Lower Abdomen", "Pelvis",
    "Hips", "Thighs"
]

ORGAN_LIST = [
    "Tongue", "Mandible", "Parotid", "Maxilla", "Nasal Air",
    "Cervical Spine", "Thyroid", "Airway", "Neck Muscles", "Neck Vessels",
    "Trachea", "Aortic Arch", "Lung Apex", "Clavicle", "Pectoralis Muscle",
    "Heart", "Descending Aorta", "Ribs", "Thoracic Spine", "Esophagus",
    "Diaphragm", "Lower Lungs", "IVC", "Upper Liver", "Liver", "Spleen",
    "Gallbladder", "Stomach", "Pancreas", "L-Kidney", "R-Kidney",
    "Abdominal Aorta", "Psoas", "Lumbar Spine", "Small Intestine",
    "Colon", "Mesentery", "Abdominal Wall", "Bladder", "Rectum",
    "Pelvic Bone", "Uterus/Prostate", "Sacrum", "Hip Joint",
    "Femoral Head", "Gluteus", "Groin", "Femur", "Quadriceps",
    "Sartorius", "Femoral Vessels", "Subcutaneous Fat"
]

SYSTEM_PROMPT = f"""You are an expert diagnostic radiologist. Your task is to analyze the provided T2-weighted MRI slice and output your analysis strictly in JSON format.

Your output must contain exactly three fields:
1. "Body Part": You must select exactly one value from the following list: {BODY_PARTS}.
2. "Organs Present": Identify only the clearly visible organs from the following list and return them as a list: {ORGAN_LIST}.
3. "Diseased Organs": List any organs from the "Organs Present" list that show obvious and significant pathology or disease. If the slice is healthy or no significant disease is detected, output ["None"].

Constraints:
- Only use organ names provided in the list above.
- Do not include any explanatory text, introductory remarks, or markdown formatting outside of the JSON object.
- Strictly adhere to the provided lists.
"""


def build_model(model_name: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable GEMINI_API_KEY is not set.")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )


def load_existing_results(output_path: Path) -> dict:
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_results(results: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def process_batch(input_dir: Path, output_path: Path, model_name: str) -> None:
    model = build_model(model_name)
    all_results = load_existing_results(output_path)

    image_extensions = {".png", ".jpg", ".jpeg"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions])

    print(f"Total files found: {len(files)}. Starting processing...")

    for idx, image_path in enumerate(files, start=1):
        filename = image_path.name

        if filename in all_results:
            continue

        try:
            img = Image.open(image_path)

            response = model.generate_content(img)

            try:
                analysis = json.loads(response.text)
                all_results[filename] = analysis
                print(f"[{idx}/{len(files)}] Processed: {filename}")
            except json.JSONDecodeError:
                print(f"[{idx}/{len(files)}] Error: Failed to parse JSON for {filename}")
                all_results[filename] = {
                    "error": "JSON_PARSE_FAILED",
                    "raw_output": response.text,
                }

            save_results(all_results, output_path)

        except Exception as e:
            print(f"[{idx}/{len(files)}] Critical error on {filename}: {e}")
            time.sleep(2)

    print(f"\nTask completed. Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch MRI slice analysis with Gemini.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the folder containing input MRI slice images.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("analysis_results.json"),
        help="Path to save the JSON results.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini model name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_batch(
        input_dir=args.input_dir,
        output_path=args.output_json,
        model_name=args.model_name,
    )