from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from detection.bubble_detector import detect_bubbles, merge_boxes
from ocr.paddle_ocr import extract_text_from_crop
from translator.deepseek_translator import translate_text
from overlay.text_overlay import overlay_text_on_bubble

app = FastAPI()


@app.post("/translate-manga/")
async def translate_manga(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    boxes = merge_boxes(detect_bubbles(img))

    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]

        jp_text = extract_text_from_crop(crop)
        if not jp_text.strip():
            continue

        en_text = translate_text(jp_text)
        img = overlay_text_on_bubble(img, box, en_text)

    _, encoded_img = cv2.imencode('.jpg', img)
    return {"image": encoded_img.tobytes()}
