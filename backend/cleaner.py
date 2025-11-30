import sys
import os
import cv2
import numpy as np
import glob
import time
import re
import json

# --- CONFIGURATION ---
OUTPUT_CLEAN_DIR = "cleaned_output"
OUTPUT_JSON_DIR = "json_output"
DEBUG_DIR = "debug_boxes" 

# --- PATH SETUP ---
current_dir = os.getcwd()
repo_path = os.path.join(current_dir, 'comic-text-detector')
model_path = os.path.join(repo_path, 'data', 'comictextdetector.pt')

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from inference import TextDetector

def load_model():
    if not os.path.exists(model_path):
        print(f"CRITICAL: Model missing at {model_path}")
        sys.exit(1)
    print("Loading AI Model...")
    model = TextDetector(model_path=model_path, input_size=1024, device='cpu')
    print("Model Loaded.")
    return model

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def merge_close_boxes(boxes, distance_threshold=30):
    """
    Merges boxes that are close to each other or overlapping.
    boxes format: [[x1, y1, x2, y2], ...]
    """
    if not boxes: return []

    while True:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]: continue
            
            # Start with current box
            bx1, by1, bx2, by2 = boxes[i]
            used[i] = True
            
            # Check against all other unused boxes
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                
                ox1, oy1, ox2, oy2 = boxes[j]
                
                # Check proximity (expand box A by threshold and see if B intersects)
                # Horizontal check
                x_overlap = (bx1 - distance_threshold < ox2) and (bx2 + distance_threshold > ox1)
                # Vertical check
                y_overlap = (by1 - distance_threshold < oy2) and (by2 + distance_threshold > oy1)
                
                if x_overlap and y_overlap:
                    # Merge them!
                    bx1 = min(bx1, ox1)
                    by1 = min(by1, oy1)
                    bx2 = max(bx2, ox2)
                    by2 = max(by2, oy2)
                    used[j] = True
                    merged = True
            
            new_boxes.append([bx1, by1, bx2, by2])
        
        boxes = new_boxes
        if not merged: break # No changes this round, we are done

    return boxes

def process_folder(input_folder, detector):
    clean_output_path = os.path.join(input_folder, OUTPUT_CLEAN_DIR)
    json_output_path = os.path.join(input_folder, OUTPUT_JSON_DIR)
    debug_path = os.path.join(input_folder, DEBUG_DIR)
    
    os.makedirs(clean_output_path, exist_ok=True)
    os.makedirs(json_output_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_files = list(set(image_files))
    image_files.sort(key=natural_sort_key)

    if not image_files:
        print(f"No images found in: {input_folder}")
        return

    print(f"Found {len(image_files)} images. Starting pipeline...")
    print("-" * 30)

    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"[{i+1}/{len(image_files)}] Processing: {file_name}...", end=" ", flush=True)
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                print("FAILED (Read Error)")
                continue

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector(img_rgb)
            
            final_mask = None
            
            # 1. EXTRACT MASK
            if isinstance(results, tuple) and len(results) > 0:
                potential_mask = results[0]
                if isinstance(potential_mask, np.ndarray) and potential_mask.ndim >= 2:
                    final_mask = potential_mask
                
                if len(results) > 1 and isinstance(results[1], np.ndarray) and results[1].ndim >= 2:
                     if results[1].shape[0] > 100: 
                        final_mask = results[1]

            if final_mask is None:
                print("Skipped (No Mask)")
                continue

            # 2. PREPARE MASK
            if final_mask.dtype != np.uint8:
                final_mask = final_mask.astype(np.uint8)

            h, w = image.shape[:2]
            if final_mask.shape[:2] != (h, w):
                final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect letters
            kernel = np.ones((10, 10), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

            # 3. GENERATE & MERGE BOXES
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_boxes = []
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 10 or bh < 10: continue # Skip noise
                raw_boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
            
            # --- THE FIX: MERGE BOXES ---
            merged_boxes = merge_close_boxes(raw_boxes, distance_threshold=40)
            
            json_data = []
            debug_img = image.copy()
            
            for idx, box in enumerate(merged_boxes):
                x1, y1, x2, y2 = box
                
                block = {
                    "id": idx + 1,
                    "bbox": [x1, y1, x2, y2], 
                    "text_raw": "", 
                    "text_translated": "" 
                }
                json_data.append(block)
                
                # Debug Draw
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save JSON
            json_save_path = os.path.join(json_output_path, base_name + ".json")
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            # Save Debug
            cv2.imwrite(os.path.join(debug_path, file_name), debug_img)

            # 4. CLEAN IMAGE
            cleaned = cv2.inpaint(image, binary_mask, 5, cv2.INPAINT_TELEA)
            save_path = os.path.join(clean_output_path, file_name)
            cv2.imencode(os.path.splitext(file_name)[1], cleaned)[1].tofile(save_path)
            
            print(f"DONE (Merged {len(raw_boxes)} -> {len(merged_boxes)} bubbles)")

        except Exception as e:
            print(f"ERROR: {e}")

    print("-" * 30)
    print("Batch Complete! Check 'debug_boxes' folder to verify merges.")

if __name__ == "__main__":
    model = load_model()
    while True:
        print("\n" + "="*40)
        path = input("Enter FOLDER PATH (or 'q'):\n>> ").strip().replace('"', '').replace("'", "")
        if path.lower() == 'q': break
        if os.path.exists(path): process_folder(path, model)
        else: print("❌ Invalid path.")