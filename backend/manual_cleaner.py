import sys
import os
import cv2
import numpy as np
import glob
import time
import re
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# --- CONFIGURATION ---
OUTPUT_CLEAN_DIR = "../cleaned_output"
OUTPUT_JSON_DIR = "../json_output"

# --- PATH SETUP ---
current_dir = os.getcwd()
repo_path = os.path.join(current_dir, 'comic-text-detector')
model_path = os.path.join(repo_path, 'data', 'comictextdetector.pt')

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from inference import TextDetector

# --- STATE ---
current_boxes = []
current_ax = None
current_fig = None
rect_patches = []
selector = None

def load_model():
    if not os.path.exists(model_path):
        print(f"CRITICAL: Model missing at {model_path}")
        sys.exit(1)
    print("Loading AI Model...")
    model = TextDetector(model_path=model_path, input_size=1024, device='cpu')
    print("Model Loaded.")
    return model

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# --- GUI LOGIC ---
def on_select(eclick, erelease):
    global current_boxes
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    nx1, nx2 = sorted([x1, x2])
    ny1, ny2 = sorted([y1, y2])
    if abs(nx2 - nx1) > 5 and abs(ny2 - ny1) > 5:
        current_boxes.append([nx1, ny1, nx2, ny2])
        redraw_boxes()

def on_click(event):
    global current_boxes
    if event.button == 3: # Right Click
        if event.xdata is None or event.ydata is None: return
        cx, cy = int(event.xdata), int(event.ydata)
        for i in range(len(current_boxes) - 1, -1, -1):
            x1, y1, x2, y2 = current_boxes[i]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                current_boxes.pop(i)
                redraw_boxes()
                break

def redraw_boxes():
    global current_ax, rect_patches, current_fig
    for p in rect_patches: p.remove()
    rect_patches = []
    for box in current_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        current_ax.add_patch(rect)
        rect_patches.append(rect)
    current_fig.canvas.draw()

def manual_editor(image_rgb, auto_boxes, filename):
    global current_boxes, current_ax, current_fig, selector
    current_boxes = list(auto_boxes)
    
    current_fig, current_ax = plt.subplots(figsize=(12, 10))
    plt.title(f"EDITING: {filename}\nLeft Drag=ADD | Right Click=REMOVE | Close Window=SAVE")
    current_ax.imshow(image_rgb)
    redraw_boxes()
    
    selector = RectangleSelector(current_ax, on_select, useblit=True, button=[1], 
                                 minspanx=5, minspany=5, spancoords='data', interactive=False)
    current_fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()
    return current_boxes

def run_ai_detection(image, detector):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector(image_rgb)
    
    final_mask = None
    if isinstance(results, tuple) and len(results) > 0:
        if isinstance(results[0], np.ndarray) and results[0].ndim >= 2: final_mask = results[0]
        if len(results) > 1 and isinstance(results[1], np.ndarray) and results[1].ndim >= 2:
                if results[1].shape[0] > 100: final_mask = results[1]

    auto_boxes = []
    if final_mask is not None:
        if final_mask.dtype != np.uint8: final_mask = final_mask.astype(np.uint8)
        h, w = image.shape[:2]
        if final_mask.shape[:2] != (h, w):
            final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > 15 and bh > 15:
                auto_boxes.append([x, y, x+bw, y+bh])
    
    return auto_boxes

def process_single_image(img_path, detector, input_folder, open_editor=True):
    file_name = os.path.basename(img_path)
    base_name = os.path.splitext(file_name)[0]
    
    clean_out = os.path.join(input_folder, OUTPUT_CLEAN_DIR)
    json_out = os.path.join(input_folder, OUTPUT_JSON_DIR)
    
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(json_out, exist_ok=True)

    print(f"Loading {file_name}...")
    image = cv2.imread(img_path)
    if image is None: return

    json_path = os.path.join(json_out, base_name + ".json")
    boxes = []
    
    if os.path.exists(json_path):
        print(" -> Found existing data. Loading instantly...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                boxes.append(item['bbox'])
    else:
        print(" -> No data found. Running AI detection...")
        boxes = run_ai_detection(image, detector)

    if open_editor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = manual_editor(image_rgb, boxes, file_name)
    
    json_data = []
    user_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        block = {
            "id": idx + 1,
            "bbox": [x1, y1, x2, y2], 
            "text_raw": "", 
            "text_translated": "" 
        }
        json_data.append(block)
        cv2.rectangle(user_mask, (x1, y1), (x2, y2), 255, -1)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    user_mask = cv2.dilate(user_mask, np.ones((5,5), np.uint8), iterations=2)
    cleaned = cv2.inpaint(image, user_mask, 5, cv2.INPAINT_TELEA)
    
    save_path = os.path.join(clean_out, file_name)
    cv2.imencode(os.path.splitext(file_name)[1], cleaned)[1].tofile(save_path)
    print(f"✅ Processed {file_name}")

def parse_selection(selection_str, total_files):
    """
    Parses strings like "1, 3-5, 8" into a list of indices.
    """
    indices = set()
    parts = selection_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                # User is 1-based, we want 0-based
                for i in range(start, end + 1):
                    indices.add(i - 1)
            except ValueError:
                continue
        elif part.isdigit():
            indices.add(int(part) - 1)
            
    # Filter valid indices
    valid_indices = sorted([i for i in indices if 0 <= i < total_files])
    return valid_indices

def process_folder(input_folder, model):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_files = list(set(image_files))
    image_files.sort(key=natural_sort_key)

    if not image_files:
        print(f"No images found in {os.path.basename(input_folder)}.")
        return

    print(f"\n📂 Folder: {os.path.basename(input_folder)}")
    print(f"Found {len(image_files)} pages.")
    
    # --- PAGE SELECTION MENU ---
    print("\nHow would you like to proceed?")
    print(" [A] Process ALL pages automatically")
    print(" [S] Select specific pages (e.g., 1, 3-5)")
    
    mode = input(">> ").strip().upper()
    
    files_to_process = []
    
    if mode == 'S':
        print("\n--- Page List ---")
        for i, f in enumerate(image_files):
            print(f"{i+1}. {os.path.basename(f)}")
            
        sel_str = input("\nEnter page numbers (e.g. '1-3, 5'): ")
        indices = parse_selection(sel_str, len(image_files))
        
        if not indices:
            print("❌ No valid pages selected.")
            return
            
        files_to_process = [image_files[i] for i in indices]
        print(f"Selected {len(files_to_process)} pages.")
        
    else: # Default to All
        files_to_process = image_files
        print("Processing ALL pages.")

    # --- EXECUTION LOOP ---
    print("\n" + "="*40)
    print("🚀 STARTING EDITOR")
    print("="*40)
    
    for i, path in enumerate(files_to_process):
        print(f"\n[{i+1}/{len(files_to_process)}] Opening: {os.path.basename(path)}")
        process_single_image(path, model, input_folder, open_editor=True)
    
    print("\n✅ Queue complete!")

    # --- POST-COMPLETION MENU ---
    while True:
        print("\nOptions:")
        print(" [O] Open a specific page to re-edit")
        print(" [Q] Quit / Change Folder")
        
        choice = input(">> ").strip().upper()

        if choice == 'Q':
            break
        elif choice == 'O':
            print("\n--- File List ---")
            for i, path in enumerate(image_files):
                print(f"{i+1}. {os.path.basename(path)}")
            
            try:
                sel = input("\nEnter page number: ")
                idx = int(sel) - 1
                if 0 <= idx < len(image_files):
                    process_single_image(image_files[idx], model, input_folder, open_editor=True)
                else:
                    print("❌ Invalid number.")
            except ValueError:
                print("❌ Please enter a valid number.")

if __name__ == "__main__":
    model = load_model()
    
    while True:
        print("\n" + "="*50)
        print("   MANUAL CLEANING PIPELINE")
        print("="*50)
        path = input("Enter FOLDER PATH (or 'q' to quit):\n>> ").strip().replace('"', '').replace("'", "")
        
        if path.lower() == 'q':
            break
            
        if os.path.exists(path):
            process_folder(path, model)
        else:
            print("❌ Invalid path.")