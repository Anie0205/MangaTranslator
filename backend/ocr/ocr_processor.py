from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64

from .tesseract_ocr import (
    extract_boxes_and_text,
    group_boxes_into_bubbles,
    debug_overlay_bubbles,
    overlay_translations
)
from translation.translator import translate_text

app = FastAPI(title="OCR Processor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    source_lang: str = Form("japanese"),
    target_lang: str = Form("english")
):
    """Extract text and bubbles from manga image"""
    try:
        image_bytes = await file.read()
        
        # Extract boxes and text
        np_img_bgr, boxes_and_text = extract_boxes_and_text(image_bytes, source_lang)
        
        # Group into bubbles
        bubbles = group_boxes_into_bubbles(boxes_and_text)
        
        # Create debug overlay
        debug_img = debug_overlay_bubbles(np_img_bgr, bubbles)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', debug_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "original_text": " ".join([item["word"] for item in boxes_and_text]),
            "bubbles_count": len(bubbles),
            "image_base64": image_base64,
            "ocr_boxes": boxes_and_text
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
