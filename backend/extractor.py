import os
import json
import cv2
import glob
import re
import sys
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
JSON_SUBFOLDER = "../json_output"
OUTPUT_FILENAME = "../batch_text_data.json"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_ocr_engine(lang_mode):
    """
    Loads the specific AI model based on user choice to save RAM.
    """
    if lang_mode == "ja":
        print("🇯🇵 Loading MangaOCR for Japanese...")
        try:
            from manga_ocr import MangaOcr
            return MangaOcr()
        except ImportError:
            print("Error: 'manga-ocr' not installed.")
            sys.exit(1)
            
    elif lang_mode == "ko":
        print("🇰🇷 Loading EasyOCR for Korean...")
        try:
            import easyocr
            # gpu=True is recommended if you have an Nvidia card
            return easyocr.Reader(['ko', 'en'], gpu=True) 
        except ImportError:
            print("Error: 'easyocr' not installed.")
            sys.exit(1)
            
    elif lang_mode == "zh":
        print("🇨🇳 Loading EasyOCR for Chinese (Simplified)...")
        try:
            import easyocr
            return easyocr.Reader(['ch_sim', 'en'], gpu=True)
        except ImportError:
            print("Error: 'easyocr' not installed.")
            sys.exit(1)

def run_extraction(project_folder):
    # 1. Ask for Language
    print("\nSelect the source language:")
    print("1. Japanese (Manga) -> Uses MangaOCR")
    print("2. Korean (Manhwa)  -> Uses EasyOCR")
    print("3. Chinese (Manhua) -> Uses EasyOCR")
    
    choice = input("Enter number (1-3): ").strip()
    
    lang_mode = "ja" # Default
    if choice == "2": lang_mode = "ko"
    if choice == "3": lang_mode = "zh"
    
    # 2. Setup Paths
    json_dir = os.path.join(project_folder, JSON_SUBFOLDER)
    output_path = os.path.join(project_folder, OUTPUT_FILENAME)

    if not os.path.exists(json_dir):
        print(f"ERROR: Could not find '{JSON_SUBFOLDER}' in {project_folder}")
        return

    # 3. Load the correct Engine
    ocr_engine = load_ocr_engine(lang_mode)

    # 4. Find Files
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    json_files.sort(key=natural_sort_key)
    
    if not json_files:
        print("No JSON files found.")
        return

    print(f"Found {len(json_files)} pages. Starting Extraction...")
    
    full_batch_data = []

    # 5. Extraction Loop
    for i, json_file in enumerate(json_files):
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        
        # Find Image
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            temp = os.path.join(project_folder, base_name + ext)
            if os.path.exists(temp):
                img_path = temp
                break
            temp = os.path.join(project_folder, base_name + ext.upper())
            if os.path.exists(temp):
                img_path = temp
                break
        
        if not img_path: continue

        print(f"[{i+1}/{len(json_files)}] OCR ({lang_mode}): {os.path.basename(img_path)}...", end=" ", flush=True)

        image_cv = cv2.imread(img_path)
        with open(json_file, "r", encoding="utf-8") as f:
            coords_data = json.load(f)

        page_entry = {
            "file_name": os.path.basename(img_path),
            "blocks": []
        }

        bubble_count = 0
        for block in coords_data:
            bbox = block.get("bbox", [])
            if len(bbox) != 4: continue

            x1, y1, x2, y2 = bbox
            h, w = image_cv.shape[:2]
            
            # Add padding for better OCR context
            pad = 5
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

            if x2 <= x1 or y2 <= y1: continue

            # Crop
            crop = image_cv[y1:y2, x1:x2]
            
            # --- STRATEGY SWITCH ---
            raw_text = ""
            
            # STRATEGY A: JAPANESE (MangaOCR)
            if lang_mode == "ja":
                # Convert to PIL
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                try:
                    raw_text = ocr_engine(pil_crop)
                except: pass

            # STRATEGY B: KOREAN/CHINESE (EasyOCR)
            else:
                try:
                    # EasyOCR takes numpy array directly
                    results = ocr_engine.readtext(crop, detail=0, paragraph=True)
                    raw_text = " ".join(results)
                except: pass

            # Save if text found
            if raw_text.strip():
                page_entry["blocks"].append({
                    "id": block["id"],
                    "bbox": block["bbox"], # Original coords
                    "text_raw": raw_text,
                    "text_translated": ""
                })
                bubble_count += 1

        full_batch_data.append(page_entry)
        print(f"Found {bubble_count} bubbles")

    # 6. Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_batch_data, f, indent=2, ensure_ascii=False)
    
    print("-" * 40)
    print(f"✅ Extraction Complete!")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        user_input = input("Enter FOLDER PATH (or 'q'):\n>> ").strip()
        if user_input.lower() == 'q': break
        path = user_input.replace('"', '').replace("'", "")
        if os.path.exists(path): run_extraction(path)
        else: print("❌ Invalid path.")