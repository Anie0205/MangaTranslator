from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64

from ocr.tesseract_ocr import (
    extract_boxes_and_text,
    group_boxes_into_bubbles,
    overlay_translations
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
    target_lang: str = Form("english")
):
    """Translate manga image"""
    try:
        image_bytes = await file.read()
        
        # Extract text boxes using OCR
        np_img_bgr, boxes_and_text = extract_boxes_and_text(image_bytes, source_lang)
        
        if not boxes_and_text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "No text detected in image"}
            )
        
        # Extract original text
        original_text = " ".join([item["word"] for item in boxes_and_text])
        
        # Translate the extracted text
        translated_text = translate_text(original_text, source_lang, target_lang)
        
        # Overlay translation on image
        result_image = overlay_translations(np_img_bgr, boxes_and_text, translated_text)
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.png', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "original_text": original_text,
            "translated_text": translated_text,
            "image_base64": image_base64
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "manga-translator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
