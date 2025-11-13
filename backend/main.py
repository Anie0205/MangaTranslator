from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64

from ocr.tesseract_ocr import (
    extract_boxes_and_text,
    group_boxes_into_bubbles,
    overlay_translations_multi_bubble,
    debug_overlay_bubbles
)
from translation.translator import translate_text

app = FastAPI(title="Manga Translator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/translate/")
async def translate_image(
    file: UploadFile = File(...),
    source_lang: str = Form("japanese"),
    target_lang: str = Form("english"),
    debug: bool = Form(False)
):
    """Translate manga image with multi-bubble support"""
    try:
        image_bytes = await file.read()
        
        # Extract text boxes using OCR
        np_img_bgr, boxes_and_text = extract_boxes_and_text(image_bytes, source_lang)
        
        if not boxes_and_text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "No text detected in image"}
            )
        
        # Group boxes into separate speech bubbles
        bubbles = group_boxes_into_bubbles(boxes_and_text)
        
        if not bubbles:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Could not group text into bubbles"}
            )
        
        # If debug mode, return image with bubble boundaries highlighted
        if debug:
            debug_image = debug_overlay_bubbles(np_img_bgr, bubbles)
            _, buffer = cv2.imencode('.png', debug_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            bubble_texts = []
            for i, bubble in enumerate(bubbles):
                text = " ".join([item["word"] for item in bubble])
                bubble_texts.append({"bubble_id": i + 1, "text": text})
            
            return {
                "status": "success",
                "debug_mode": True,
                "bubbles_detected": len(bubbles),
                "bubble_texts": bubble_texts,
                "image_base64": image_base64
            }
        
        # Translate each bubble separately
        translations = []
        original_texts = []
        
        for bubble in bubbles:
            # Combine text from all boxes in this bubble
            bubble_text = " ".join([item["word"] for item in bubble])
            original_texts.append(bubble_text)
            
            # Translate this bubble's text
            translated = translate_text(bubble_text, source_lang, target_lang)
            translations.append(translated)
        
        # Overlay all translations on the image
        result_image = overlay_translations_multi_bubble(np_img_bgr, bubbles, translations)
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.png', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "bubbles_count": len(bubbles),
            "original_texts": original_texts,
            "translated_texts": translations,
            "image_base64": image_base64
        }
    
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "manga-translator"}

@app.get("/")
async def root():
    return {
        "message": "Manga Translator API",
        "version": "2.0 - Multi-Bubble Support",
        "endpoints": {
            "translate": "/translate/",
            "docs": "/docs",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)