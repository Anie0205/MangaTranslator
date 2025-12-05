import os
import json
import math
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
TRANSLATED_JSON = "batch_translated.json"
CLEANED_IMAGES_DIR = "cleaned_output"
FINAL_OUTPUT_DIR = "final_typeset_pages"
FONT_PATH = os.path.join("assets", "fonts", "comic.ttf") 

# --- SETTINGS ---
MAX_FONT_SIZE = 90
MIN_FONT_SIZE = 14
LINE_SPACING_RATIO = 0.15
GLOBAL_PADDING = 0.05       # 5% internal padding
EXPANSION_LIMIT = 60        # Max pixels to expand in any direction
NEIGHBOR_BUFFER = 15        # Minimum gap between text blocks

def get_smart_colors(original_img_cv, bbox):
    """K-Means Color Detection for Text & Stroke"""
    try:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = original_img_cv.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return (0, 0, 0), (255, 255, 255)
        
        crop = original_img_cv[y1:y2, x1:x2]
        if crop.size == 0: return (0, 0, 0), (255, 255, 255)

        data = crop.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        count_0 = np.count_nonzero(labels == 0)
        count_1 = np.count_nonzero(labels == 1)
        text_bgr = centers[0] if count_0 < count_1 else centers[1]
        
        b, g, r = text_bgr
        lum = 0.299*r + 0.587*g + 0.114*b
        
        if lum > 200: return (255, 255, 255), (0, 0, 0) # White Text -> Black Stroke
        elif lum < 50: return (0, 0, 0), (255, 255, 255) # Black Text -> White Stroke
        else:
            text_col = (int(r), int(g), int(b))
            stroke_col = (0, 0, 0) if lum > 180 else (255, 255, 255)
            return text_col, stroke_col
    except:
        return (0, 0, 0), (255, 255, 255)

def check_collision(box, all_boxes, self_idx, buffer=0):
    bx1, by1, bx2, by2 = box
    for i, other in enumerate(all_boxes):
        if i == self_idx: continue
        ox1, oy1, ox2, oy2 = other
        # Check if box overlaps other (with buffer)
        if (bx2 < ox1 - buffer) or (bx1 > ox2 + buffer): continue
        if (by2 < oy1 - buffer) or (by1 > oy2 + buffer): continue
        return True
    return False

def smart_inflate_box(bbox, all_boxes, self_idx, img_w, img_h):
    """
    Expands the text box outwards until it hits a neighbor, image edge, or limit.
    This reconstructs the 'Bubble' area from the 'Text' area.
    """
    x1, y1, x2, y2 = bbox
    step = 5
    
    # Expand LEFT
    for _ in range(0, EXPANSION_LIMIT, step):
        if x1 - step < 0: break
        if check_collision([x1 - step, y1, x2, y2], all_boxes, self_idx, NEIGHBOR_BUFFER): break
        x1 -= step
        
    # Expand RIGHT
    for _ in range(0, EXPANSION_LIMIT, step):
        if x2 + step > img_w: break
        if check_collision([x1, y1, x2 + step, y2], all_boxes, self_idx, NEIGHBOR_BUFFER): break
        x2 += step

    # Expand UP
    for _ in range(0, EXPANSION_LIMIT, step):
        if y1 - step < 0: break
        if check_collision([x1, y1 - step, x2, y2], all_boxes, self_idx, NEIGHBOR_BUFFER): break
        y1 -= step

    # Expand DOWN
    for _ in range(0, EXPANSION_LIMIT, step):
        if y2 + step > img_h: break
        if check_collision([x1, y1, x2, y2 + step], all_boxes, self_idx, NEIGHBOR_BUFFER): break
        y2 += step

    return [x1, y1, x2, y2]

def calculate_rounded_box_width(y_offset, w, h, corner_radius_percent=0.15):
    half_h = h / 2
    half_w = w / 2
    straight_h = half_h * (1.0 - corner_radius_percent)
    
    if abs(y_offset) <= straight_h: return w
    
    d_y = abs(y_offset) - straight_h
    max_d_y = half_h * corner_radius_percent
    if d_y >= max_d_y: return 0
    
    corner_h = max_d_y
    corner_w = half_w * 0.5 
    width_lost = corner_w * (1 - math.sqrt(1 - (d_y**2 / corner_h**2)))
    return w - (width_lost * 2)

def get_wrapped_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    current_w = 0
    space_w = font.getlength(" ")
    
    for word in words:
        word_w = font.getlength(word)
        if current_w + space_w + word_w <= max_width:
            current_line.append(word)
            current_w += space_w + word_w
        else:
            if current_line: lines.append(" ".join(current_line))
            current_line = [word]
            current_w = word_w
    if current_line: lines.append(" ".join(current_line))
    return lines

def try_fit_text(text, box_w, box_h, font_path, min_fs=MIN_FONT_SIZE):
    usable_w = box_w * (1.0 - GLOBAL_PADDING)
    usable_h = box_h * (1.0 - GLOBAL_PADDING)

    for size in range(MAX_FONT_SIZE, min_fs - 1, -2):
        try:
            font = ImageFont.truetype(font_path, size)
        except: return False, None, [], 0, 0, 0

        # Calculate stroke relative to font size
        stroke_width = max(2, int(size / 10))
        
        # Available space must accommodate text AND stroke
        safe_w = usable_w - (stroke_width * 2)
        safe_h = usable_h - (stroke_width * 2)
        
        if safe_w < size: continue

        lines = get_wrapped_text(text, font, safe_w)
        if not lines: continue
        
        ascent, descent = font.getmetrics()
        line_base_h = ascent + descent
        line_total_h = line_base_h * (1 + LINE_SPACING_RATIO)
        block_h = (line_total_h * len(lines)) - (line_base_h * LINE_SPACING_RATIO)

        # Check total height
        if block_h > safe_h: continue

        # Check corners (Roundness)
        fits = True
        current_y = -(block_h / 2) + (line_base_h / 2)
        
        for line in lines:
            line_w = font.getlength(line)
            allowed_w = calculate_rounded_box_width(current_y, safe_w, safe_h)
            if line_w > allowed_w:
                fits = False
                break
            current_y += line_total_h
            
        if fits:
            return True, font, lines, block_h, line_total_h, stroke_width

    return False, None, [], 0, 0, 0

def run_typesetting(project_folder):
    json_path = os.path.join(project_folder, TRANSLATED_JSON)
    clean_dir = os.path.join(project_folder, CLEANED_IMAGES_DIR)
    output_dir = os.path.join(project_folder, FINAL_OUTPUT_DIR)
    
    if not os.path.exists(json_path):
        print(f"ERROR: Cannot find {TRANSLATED_JSON}")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
        
    for page in batch_data:
        file_name = page["file_name"]
        clean_img_path = os.path.join(clean_dir, file_name)
        
        # Find Original Image (for colors)
        original_img_path = os.path.join(project_folder, file_name)
        if not os.path.exists(original_img_path):
            base_name = os.path.splitext(file_name)[0]
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                test_path = os.path.join(project_folder, base_name + ext)
                if os.path.exists(test_path):
                    original_img_path = test_path
                    break
        
        img_original_cv = None
        if os.path.exists(original_img_path):
            img_original_cv = cv2.imread(original_img_path)
            h, w_img = img_original_cv.shape[:2]
        else:
            w_img, h = 1000, 1000 # Fallback
        
        if not os.path.exists(clean_img_path): continue
            
        img_clean = Image.open(clean_img_path).convert("RGBA")
        draw = ImageDraw.Draw(img_clean)
        
        print(f"Typesetting {file_name}...", end=" ", flush=True)
        
        all_bboxes = [b["bbox"] for b in page["blocks"]]
            
        for i, block in enumerate(page["blocks"]):
            text = block.get("text_translated", "")
            if not text: continue

            # 1. Smart Inflate: Expand box to find available bubble space
            raw_bbox = block["bbox"]
            expanded_bbox = smart_inflate_box(raw_bbox, all_bboxes, i, w_img, h)
            
            x1, y1, x2, y2 = expanded_bbox
            w = x2 - x1
            h = y2 - y1
            cx, cy = x1 + w/2, y1 + h/2

            # 2. Get Colors (from original tight box to avoid background noise)
            text_color, stroke_color = (0,0,0), (255,255,255)
            if img_original_cv is not None:
                text_color, stroke_color = get_smart_colors(img_original_cv, raw_bbox)

            # 3. Fit Text
            success, font, lines, block_h, line_h, stroke_width = try_fit_text(text, w, h, FONT_PATH)
            
            # 4. Fallback (If too small, force minimal readable text)
            if not success:
                # Last resort: Force a small font and just wrap
                font = ImageFont.truetype(FONT_PATH, MIN_FONT_SIZE)
                stroke_width = 2
                lines = get_wrapped_text(text, font, w)
                ascent, descent = font.getmetrics()
                line_h = (ascent + descent) * (1 + LINE_SPACING_RATIO)
                block_h = line_h * len(lines)

            # 5. Draw
            current_y = cy - (block_h / 2)
            for line in lines:
                line_w = font.getlength(line)
                current_x = cx - (line_w / 2)
                
                # Stroke
                for adj_x in range(-stroke_width, stroke_width+1):
                    for adj_y in range(-stroke_width, stroke_width+1):
                        if adj_x**2 + adj_y**2 <= stroke_width**2:
                            draw.text((current_x+adj_x, current_y+adj_y), line, font=font, fill=stroke_color)
                # Fill
                draw.text((current_x, current_y), line, font=font, fill=text_color)
                current_y += line_h

        img_clean.convert("RGB").save(os.path.join(output_dir, file_name))
        print("DONE")

if __name__ == "__main__":
    while True:
        path = input("\nEnter FOLDER PATH (or 'q'):\n>> ").strip().replace('"', '').replace("'", "")
        if path.lower() == 'q': break
        if os.path.exists(path): run_typesetting(path)