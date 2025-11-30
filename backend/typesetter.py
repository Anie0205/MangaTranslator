import os
import json
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
TRANSLATED_JSON = "batch_translated.json"
CLEANED_IMAGES_DIR = "cleaned_output"
FINAL_OUTPUT_DIR = "final_typeset_pages"
FONT_PATH = os.path.join("assets", "fonts", "comic.ttf") 

# --- SETTINGS ---
MAX_FONT_SIZE = 80   # Start big
MIN_FONT_SIZE = 12   # Don't go smaller than unreadable
STROKE_WIDTH = 4
LINE_SPACING_RATIO = 0.2 # 20% of line height as gap
GLOBAL_PADDING = 0.05 # 5% padding from edges regardless of shape

def is_rectangle_shape(w, h):
    """Determines if box is rectangular (narration) or square-ish (bubble)"""
    ratio = w / h if h != 0 else 1
    # If aspect ratio is far from 1:1, treat as rectangle
    return (ratio > 1.5) or (ratio < 0.7)

def calculate_ellipse_width_at_y(y_offset_from_center, semi_axis_a, semi_axis_b):
    """
    Calculates available width in an ellipse at a specific vertical offset from center.
    Based on ellipse equation: x^2/a^2 + y^2/b^2 = 1
    """
    if abs(y_offset_from_center) >= semi_axis_b:
        return 0
    
    # x = a * sqrt(1 - y^2/b^2)
    x_half = semi_axis_a * math.sqrt(1 - (y_offset_from_center**2 / semi_axis_b**2))
    return x_half * 2

def get_wrapped_text(text, font, max_width):
    """Standard greedy wrapping"""
    words = text.split()
    if not words: return []
    lines = []
    current_line = []
    current_w = 0
    space_w = font.getlength(" ")
    
    for word in words:
        word_w = font.getlength(word)
        # Basic check to prevent single massive words breaking things
        if word_w > max_width and not current_line:
             lines.append(word) # Let it overflow, font size will shrink next iter
             continue

        if current_w + space_w + word_w <= max_width:
            current_line.append(word)
            current_w += space_w + word_w
        else:
            if current_line: lines.append(" ".join(current_line))
            current_line = [word]
            current_w = word_w
            
    if current_line: lines.append(" ".join(current_line))
    return lines

def fit_text_precise(text, box_w, box_h, font_path):
    """
    Iteratively finds the largest font that fits perfectly inside the shape geometry.
    """
    is_rect = is_rectangle_shape(box_w, box_h)
    
    # Apply global padding
    usable_w = box_w * (1.0 - GLOBAL_PADDING*2)
    usable_h = box_h * (1.0 - GLOBAL_PADDING*2)
    
    # Ellipse parameters
    semi_a = usable_w / 2
    semi_b = usable_h / 2

    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        try:
            font = ImageFont.truetype(font_path, size)
        except: sys.exit(1)

        # 1. Initial Wrap (using max available width)
        lines = get_wrapped_text(text, font, usable_w)
        if not lines: continue
        
        # 2. Calculate Metrics
        ascent, descent = font.getmetrics()
        line_base_h = ascent + descent
        line_spacing = line_base_h * LINE_SPACING_RATIO
        total_text_h = (line_base_h * len(lines)) + (line_spacing * (len(lines) - 1))

        # 3. Basic Vertical Fit Check
        if total_text_h > usable_h:
            continue # Too tall, try smaller font

        # 4. GEOMETRY COLLISION CHECK
        if is_rect:
            # For rectangles, if it passed vertical fit, it's good.
            return font, lines, total_text_h, line_base_h, line_spacing
        else:
            # For Ellipses, check every line against the curve
            fits_geometry = True
            start_y_offset = -total_text_h / 2 + line_base_h / 2

            for i, line in enumerate(lines):
                line_w = font.getlength(line)
                # Calculate vertical center of this specific line
                current_line_y_offset = start_y_offset + i * (line_base_h + line_spacing)
                
                # Get available width at this Y position
                allowed_width = calculate_ellipse_width_at_y(current_line_y_offset, semi_a, semi_b)
                
                if line_w > allowed_width:
                    fits_geometry = False
                    break
            
            if fits_geometry:
                 return font, lines, total_text_h, line_base_h, line_spacing

    # Fallback: Min size
    font = ImageFont.truetype(font_path, MIN_FONT_SIZE)
    lines = get_wrapped_text(text, font, usable_w)
    ascent, descent = font.getmetrics()
    line_base_h = ascent + descent
    line_spacing = line_base_h * LINE_SPACING_RATIO
    total_text_h = (line_base_h * len(lines)) + (line_spacing * (len(lines) - 1))
    return font, lines, total_text_h, line_base_h, line_spacing

def run_typesetting(project_folder):
    json_path = os.path.join(project_folder, TRANSLATED_JSON)
    clean_dir = os.path.join(project_folder, CLEANED_IMAGES_DIR)
    output_dir = os.path.join(project_folder, FINAL_OUTPUT_DIR)
    
    if not os.path.exists(json_path):
        print("JSON missing.")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
        
    print(f"Typesetting {len(batch_data)} pages with Precise Mode...")
    
    for page in batch_data:
        file_name = page["file_name"]
        clean_img_path = os.path.join(clean_dir, file_name)
        if not os.path.exists(clean_img_path): continue
        
        img = Image.open(clean_img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        print(f"Rendering: {file_name}...", end=" ", flush=True)
        
        for block in page["blocks"]:
            text = block.get("text_translated", "")
            if not text or "[TR]" in text: continue

            x1, y1, x2, y2 = block["bbox"]
            w = x2 - x1
            h = y2 - y1
            if w < 20 or h < 20: continue

            # Center of box
            cx, cy = x1 + w / 2, y1 + h / 2

            # --- PRECISE FITTING ---
            font, lines, total_h, line_h, line_spacing = fit_text_precise(text, w, h, FONT_PATH)
            
            # Calculate start Y to center the block vertically
            current_y = cy - (total_h / 2)
            
            for line in lines:
                line_w = font.getlength(line)
                current_x = cx - (line_w / 2)
                
                # Stroke & Fill
                for adj_x in range(-STROKE_WIDTH, STROKE_WIDTH+1):
                    for adj_y in range(-STROKE_WIDTH, STROKE_WIDTH+1):
                        draw.text((current_x+adj_x, current_y+adj_y), line, font=font, fill="white")
                draw.text((current_x, current_y), line, font=font, fill="black")
                
                current_y += line_h + line_spacing

        img.convert("RGB").save(os.path.join(output_dir, file_name))
        print("DONE")

if __name__ == "__main__":
    while True:
        path = input("\nEnter FOLDER PATH (or 'q'):\n>> ").strip().replace('"', '').replace("'", "")
        if path.lower() == 'q': break
        if os.path.exists(path): run_typesetting(path)