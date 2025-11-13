import pytesseract
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import textwrap
from io import BytesIO
from sklearn.cluster import DBSCAN
import random

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

LANG_MAP = {
    "japanese": "jpn",
    "korean": "kor",
    "chinese": "chi_sim",
}

def extract_boxes_and_text(image_bytes: bytes, lang_key: str):
    lang = LANG_MAP.get(lang_key.lower(), "jpn")
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(img)
    
    # Convert RGB to BGR for OpenCV compatibility
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # OCR with bounding boxes (use RGB for tesseract)
    data = pytesseract.image_to_data(np_img, lang=lang, output_type=pytesseract.Output.DICT)

    boxes_and_text = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes_and_text.append({"word": word, "box": (x, y, w, h)})

    return np_img_bgr, boxes_and_text

def detect_bubble_boundaries(boxes_and_text, image_np):
    """Detect the overall bubble boundary by finding white/light areas around text"""
    if not boxes_and_text:
        return None
    
    # Convert BGR to grayscale for better bubble detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    image_shape = image_np.shape
    
    # Find the bounding box that encompasses all text
    min_x = min(box["box"][0] for box in boxes_and_text)
    min_y = min(box["box"][1] for box in boxes_and_text)
    max_x = max(box["box"][0] + box["box"][2] for box in boxes_and_text)
    max_y = max(box["box"][1] + box["box"][3] for box in boxes_and_text)
    
    # Expand search area around text (make it larger)
    search_padding = 80  # Increased from 50
    search_x1 = max(0, min_x - search_padding)
    search_y1 = max(0, min_y - search_padding)
    search_x2 = min(image_shape[1], max_x + search_padding)
    search_y2 = min(image_shape[0], max_y + search_padding)
    
    # Extract the search region
    search_region = gray[search_y1:search_y2, search_x1:search_x2]
    
    # Try multiple threshold values to detect different bubble colors
    thresholds = [180, 200, 220, 240]  # Different brightness levels
    
    for threshold in thresholds:
        # Threshold to find light areas (bubble background)
        _, thresh = cv2.threshold(search_region, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of light areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest light area (likely the bubble)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Only use if the contour is reasonably large
            if contour_area > 1000:  # Minimum area threshold
                # Get bounding rectangle of the bubble
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Adjust coordinates back to original image space
                bubble_x = search_x1 + x
                bubble_y = search_y1 + y
                bubble_w = w
                bubble_h = h
                
                return (bubble_x, bubble_y, bubble_w, bubble_h)
    
    # Fallback: use text bounding box with larger padding
    padding = 50  # Increased from 30
    bubble_x = max(0, min_x - padding)
    bubble_y = max(0, min_y - padding)
    bubble_w = min(image_shape[1] - bubble_x, max_x - min_x + 2 * padding)
    bubble_h = min(image_shape[0] - bubble_y, max_y - min_y + 2 * padding)
    
    return (bubble_x, bubble_y, bubble_w, bubble_h)

def justify_text_line(text, target_width, font_scale, thickness):
    """Calculate character spacing to justify text across target width"""
    if len(text) <= 1:
        return text
    
    # Get text width
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    if text_width >= target_width:
        return text  # Can't justify if text is too wide
    
    # Calculate spacing needed
    total_spacing = target_width - text_width
    spaces_needed = len(text) - 1
    
    if spaces_needed == 0:
        return text
    
    # Use a more conservative spacing approach
    spacing_per_char = min(total_spacing / spaces_needed, 3)  # Limit max spacing
    
    # Create justified text with controlled spacing
    justified_text = ""
    for i, char in enumerate(text):
        justified_text += char
        if i < len(text) - 1:  # Don't add spacing after last character
            # Add a small fixed amount of spacing
            justified_text += " "
    
    return justified_text

def overlay_translations(image_np, boxes_and_text, full_translation):
    # Ensure full_translation is a string
    if not isinstance(full_translation, str):
        full_translation = str(full_translation) if full_translation is not None else "[Translation error]"
    
    # Limit translation length
    if len(full_translation) > 500:
        full_translation = full_translation[:500] + "..."
    
    output_img = image_np.copy()
    h, w = output_img.shape[:2]

    # Check image dimensions
    if h < 50 or w < 50:
        return {"error": "Image too small"}
    if h > 4000 or w > 4000:
        return {"error": "Image too large"}

    # Detect the original bubble boundary
    bubble_boundary = detect_bubble_boundaries(boxes_and_text, image_np)
    
    if bubble_boundary:
        bubble_x, bubble_y, bubble_w, bubble_h = bubble_boundary
        
        # Calculate text wrapping based on bubble width
        max_chars_per_line = int(bubble_w / 15)  # Characters per line for text wrapping
        lines = textwrap.wrap(full_translation, width=max_chars_per_line)
        
        # Adjust bubble height based on number of lines
        line_height = 25
        padding = 15
        required_height = len(lines) * line_height + 2 * padding
        bubble_h = max(bubble_h, required_height)
        
        # Create bubble with rounded corners effect
        # Add shadow effect
        shadow_offset = 3
        cv2.rectangle(output_img, 
                      (bubble_x + shadow_offset, bubble_y + shadow_offset), 
                      (bubble_x + bubble_w + shadow_offset, bubble_y + bubble_h + shadow_offset), 
                      (100, 100, 100), -1)
        
        # Draw main bubble
        cv2.rectangle(output_img, 
                      (bubble_x, bubble_y), 
                      (bubble_x + bubble_w, bubble_y + bubble_h), 
                      (255, 255, 255), -1)
        
        # Add border
        cv2.rectangle(output_img, 
                      (bubble_x, bubble_y), 
                      (bubble_x + bubble_w, bubble_y + bubble_h), 
                      (0, 0, 0), 2)
        
        # Note: Removed speech bubble tail to match original manga style
        
        # Add text with better styling and center alignment
        text_x = bubble_x + padding
        text_y = bubble_y + padding + line_height
        font_scale = 0.6
        thickness = 1
        
        for i, line in enumerate(lines):
            # Get text width for center alignment
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Calculate center position for this line
            center_x = bubble_x + bubble_w // 2
            line_x = center_x - text_width // 2
            
            # Add text shadow for better readability
            cv2.putText(output_img, line, (line_x + 1, text_y + i * line_height + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), thickness, lineType=cv2.LINE_AA)
            # Main text
            cv2.putText(output_img, line, (line_x, text_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
    else:
        # Fallback to top-center bubble if no text detected
        max_chars_per_line = int(w / 20)
        lines = textwrap.wrap(full_translation, width=max_chars_per_line)
        
        line_height = 30
        padding = 20
        bubble_height = len(lines) * line_height + 2 * padding
        bubble_width = min(w - 40, max(len(line) * 15 for line in lines) + 2 * padding)
        
        bubble_x = (w - bubble_width) // 2
        bubble_y = 20
        
        # Create fallback bubble
        cv2.rectangle(output_img, 
                      (bubble_x, bubble_y), 
                      (bubble_x + bubble_width, bubble_y + bubble_height), 
                      (255, 255, 255), -1)
        cv2.rectangle(output_img, 
                      (bubble_x, bubble_y), 
                      (bubble_x + bubble_width, bubble_y + bubble_height), 
                      (0, 0, 0), 2)
        
        text_x = bubble_x + padding
        text_y = bubble_y + padding + line_height
        
        for i, line in enumerate(lines):
            cv2.putText(output_img, line, (text_x + 1, text_y + i * line_height + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, lineType=cv2.LINE_AA)
            cv2.putText(output_img, line, (text_x, text_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return output_img


# Optional: basic OCR without boxes
def preprocess(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("L")
    np_img = np.array(img)
    _, thresh = cv2.threshold(np_img, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text(image_bytes: bytes, lang_key: str) -> str:
    lang = LANG_MAP.get(lang_key.lower(), "jpn")
    processed_img = preprocess(image_bytes)
    text = pytesseract.image_to_string(processed_img, lang=lang)
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return clean or "[No text detected]"

def group_boxes_into_bubbles(boxes_and_text, distance_threshold=60, min_samples=1):
    """Group OCR boxes into bubbles using DBSCAN clustering for accurate grouping."""
    if not boxes_and_text:
        return []
    # Use the center of each box for clustering
    centers = [((box["box"][0] + box["box"][2] // 2), (box["box"][1] + box["box"][3] // 2)) for box in boxes_and_text]
    X = np.array(centers)
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    bubbles = []
    for label in set(labels):
        group = [boxes_and_text[i] for i, l in enumerate(labels) if l == label]
        if group:
            bubbles.append(group)
    return bubbles

def debug_overlay_bubbles(image_np, bubbles):
    """Draw colored rectangles for each bubble group for visual debugging."""
    output_img = image_np.copy()
    colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(bubbles))]
    for idx, bubble in enumerate(bubbles):
        for item in bubble:
            x, y, w, h = item["box"]
            cv2.rectangle(output_img, (x, y), (x + w, y + h), colors[idx], 2)
    return output_img
