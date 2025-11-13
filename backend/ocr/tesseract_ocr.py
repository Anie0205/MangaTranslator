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


def group_boxes_into_bubbles(boxes_and_text, distance_threshold=80, min_samples=1):
    """Group OCR boxes into speech bubbles using spatial clustering."""
    if not boxes_and_text:
        return []
    
    # Use the center of each box for clustering
    centers = [
        (box["box"][0] + box["box"][2] // 2, box["box"][1] + box["box"][3] // 2) 
        for box in boxes_and_text
    ]
    X = np.array(centers)
    
    # Use DBSCAN clustering to group nearby text boxes
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    
    # Group boxes by cluster label
    bubbles = []
    for label in set(labels):
        if label == -1:  # Skip noise points
            continue
        group = [boxes_and_text[i] for i, l in enumerate(labels) if l == label]
        if group:
            bubbles.append(group)
    
    return bubbles


def get_bubble_boundary(bubble_boxes, image_shape, padding=30):
    """Calculate the bounding box for a group of text boxes."""
    if not bubble_boxes:
        return None
    
    min_x = min(box["box"][0] for box in bubble_boxes)
    min_y = min(box["box"][1] for box in bubble_boxes)
    max_x = max(box["box"][0] + box["box"][2] for box in bubble_boxes)
    max_y = max(box["box"][1] + box["box"][3] for box in bubble_boxes)
    
    # Add padding
    x = max(0, min_x - padding)
    y = max(0, min_y - padding)
    w = min(image_shape[1] - x, max_x - min_x + 2 * padding)
    h = min(image_shape[0] - y, max_y - min_y + 2 * padding)
    
    return (x, y, w, h)


def detect_bubble_mask(image_np, bubble_boundary):
    """Detect light/white areas (speech bubble background) within the boundary."""
    x, y, w, h = bubble_boundary
    
    # Extract region of interest
    roi = image_np[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find light areas (typical speech bubble backgrounds)
    _, mask = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def draw_text_in_bubble(image, bubble_boundary, text, font_scale=0.6, color=(0, 0, 0)):
    """Draw text centered in a bubble with proper wrapping."""
    x, y, w, h = bubble_boundary
    
    # Calculate max characters per line based on bubble width
    avg_char_width = 12  # Approximate width per character
    max_chars = max(10, int(w / avg_char_width))
    
    # Wrap text
    lines = textwrap.wrap(text, width=max_chars)
    
    # Calculate line height
    line_height = 25
    total_text_height = len(lines) * line_height
    
    # Calculate starting Y position (center vertically)
    start_y = y + (h - total_text_height) // 2 + line_height
    
    # Draw each line centered horizontally
    for i, line in enumerate(lines):
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Center horizontally
        text_x = x + (w - text_width) // 2
        text_y = start_y + i * line_height
        
        # Draw text shadow for better readability
        cv2.putText(
            image, line, (text_x + 1, text_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1, cv2.LINE_AA
        )
        
        # Draw main text
        cv2.putText(
            image, line, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA
        )


def overlay_translations_multi_bubble(image_np, bubbles, translations):
    """
    Overlay translations for multiple bubbles.
    
    Args:
        image_np: Original image
        bubbles: List of bubble groups (each is a list of text boxes)
        translations: List of translated text strings (one per bubble)
    """
    output_img = image_np.copy()
    h, w = output_img.shape[:2]
    
    if len(bubbles) != len(translations):
        print(f"Warning: {len(bubbles)} bubbles but {len(translations)} translations")
        translations = translations[:len(bubbles)]  # Truncate if needed
    
    for bubble_boxes, translation in zip(bubbles, translations):
        # Get bubble boundary
        boundary = get_bubble_boundary(bubble_boxes, output_img.shape)
        if not boundary:
            continue
        
        x, y, w, h = boundary
        
        # White out the original text area
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
        # Draw border
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # Draw translated text
        draw_text_in_bubble(output_img, boundary, translation)
    
    return output_img


def overlay_translations(image_np, boxes_and_text, full_translation):
    """
    Original single-bubble overlay function (kept for backward compatibility).
    For better results, use overlay_translations_multi_bubble with grouped bubbles.
    """
    if not isinstance(full_translation, str):
        full_translation = str(full_translation) if full_translation is not None else "[Translation error]"
    
    if len(full_translation) > 500:
        full_translation = full_translation[:500] + "..."
    
    output_img = image_np.copy()
    h, w = output_img.shape[:2]

    if h < 50 or w < 50 or h > 4000 or w > 4000:
        return output_img
    
    if not boxes_and_text:
        return output_img
    
    # Get overall boundary
    boundary = get_bubble_boundary(boxes_and_text, output_img.shape, padding=40)
    
    if boundary:
        x, y, w, h = boundary
        
        # White out the area
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
        # Draw border
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # Draw text
        draw_text_in_bubble(output_img, boundary, full_translation)
    
    return output_img


def debug_overlay_bubbles(image_np, bubbles):
    """Draw colored rectangles for each bubble group for visual debugging."""
    output_img = image_np.copy()
    colors = [tuple(random.randint(50, 255) for _ in range(3)) for _ in range(len(bubbles))]
    
    for idx, bubble in enumerate(bubbles):
        # Get bubble boundary
        boundary = get_bubble_boundary(bubble, output_img.shape)
        if boundary:
            x, y, w, h = boundary
            cv2.rectangle(output_img, (x, y), (x + w, y + h), colors[idx], 3)
            
            # Add bubble number
            cv2.putText(
                output_img, f"Bubble {idx + 1}", (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2
            )
    
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