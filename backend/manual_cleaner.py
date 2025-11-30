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
OUTPUT_CLEAN_DIR = "cleaned_output"
OUTPUT_JSON_DIR = "json_output"

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
    """Runs the AI model and returns a list of boxes"""
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

    # --- 1. SMART LOAD: CHECK IF JSON EXISTS ---
    json_path = os.path.join(json_out, base_name + ".json")
    boxes = []
    
    if os.path.exists(json_path):
        print(" -> Found existing data. Loading instantly...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract boxes from JSON structure
            for item in data:
                boxes.append(item['bbox'])
    else:
        print(" -> No data found. Running AI detection...")
        boxes = run_ai_detection(image, detector)

    # --- 2. OPEN EDITOR (OPTIONAL) ---
    if open_editor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = manual_editor(image_rgb, boxes, file_name)
    
    # --- 3. SAVE & CLEAN ---
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

def main_menu(input_folder, model):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_files = list(set(image_files))
    image_files.sort(key=natural_sort_key)

    if not image_files:
        print("No images found.")
        return

    while True:
        print("\n" + "="*40)
        print(f"📂 Folder: {os.path.basename(input_folder)}")
        print("Select an action:")
        print("-" * 20)
        
        for i, path in enumerate(image_files):
            clean_path = os.path.join(input_folder, OUTPUT_CLEAN_DIR, os.path.basename(path))
            json_path = os.path.join(input_folder, OUTPUT_JSON_DIR, os.path.splitext(os.path.basename(path))[0] + ".json")
            
            status = " "
            if os.path.exists(clean_path) and os.path.exists(json_path):
                status = "✅ Done"
            elif os.path.exists(json_path):
                status = "⚠️ Scanned" # AI run, but maybe not cleaned/checked
            
            print(f"{i+1:2d}. {os.path.basename(path)} \t{status}")
            
        print("-" * 20)
        print("P. PRE-SCAN ALL (Runs AI on everything now so editing is instant)")
        print("Q. Quit")
        
        choice = input("\n>> ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'p':
            print("🚀 Pre-scanning all images... (Go grab a coffee)")
            for path in image_files:
                # open_editor=False means just run AI and save JSON
                process_single_image(path, model, input_folder, open_editor=False)
            print("✨ Pre-scan complete!")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                process_single_image(image_files[idx], model, input_folder, open_editor=True)
            else:
                print("❌ Invalid number.")
        else:
            print("❌ Invalid input.")

if __name__ == "__main__":
    model = load_model()
    while True:
        print("\n" + "="*40)
        path = input("Enter FOLDER PATH (or 'q'):\n>> ").strip().replace('"', '').replace("'", "")
        if path.lower() == 'q': break
        if os.path.exists(path): main_menu(path, model)
        else: print("❌ Invalid path.")