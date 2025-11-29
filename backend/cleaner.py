import sys
import os
import cv2
import numpy as np
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

# --- PATH SETUP ---
current_dir = os.getcwd()
repo_path = os.path.join(current_dir, 'comic-text-detector')
model_path = os.path.join(repo_path, 'data', 'comictextdetector.pt')

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from inference import TextDetector

app = FastAPI()

# Check Model
if not os.path.exists(model_path):
    print(f"CRITICAL: Model missing at {model_path}")
    sys.exit(1)

# Initialize Detector
# input_size=1024 is standard
detector = TextDetector(model_path=model_path, input_size=1024, device='cpu')
print("Model loaded.")

@app.post("/clean_image")
async def clean_image(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4()}.png"
    
    try:
        # 1. SAVE & LOAD (Avoid memory corruption)
        with open(filename, "wb") as buffer:
            buffer.write(await file.read())
        
        image = cv2.imread(filename)
        if image is None: raise ValueError("Could not read saved image.")
        
        # 2. RUN DETECTION
        # The detector expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Running detection...")
        
        # results is typically (mask, mask_refined, boxes)
        results = detector(img_rgb)
        
        # 3. EXTRACT THE MASK DIRECTLY
        # We stop looking for boxes. We look for the image mask (Index 0 or 1)
        final_mask = None
        
        # Check if Index 0 is a Mask (Numpy array with same HxW as input, or close)
        if isinstance(results, tuple) and len(results) > 0:
            potential_mask = results[0]
            
            # If it's a numpy array and looks like an image (H, W)
            if isinstance(potential_mask, np.ndarray) and potential_mask.ndim >= 2:
                print("DEBUG: Found RAW MASK at Index 0. Using it directly.")
                final_mask = potential_mask
            
            # Sometimes the better mask is at Index 1
            if len(results) > 1 and isinstance(results[1], np.ndarray) and results[1].ndim >= 2:
                 # If Index 1 is also an image, it's usually the 'refined' mask. Prefer this.
                 if results[1].shape[0] > 100: # Sanity check it's not a box list
                    print("DEBUG: Found REFINED MASK at Index 1. Using it.")
                    final_mask = results[1]

        # 4. PROCESS THE MASK
        if final_mask is not None:
            # Ensure mask is uint8 (0-255)
            if final_mask.dtype != np.uint8:
                final_mask = final_mask.astype(np.uint8)

            # Resize mask to match original image exactly (Model sometimes downscales)
            h, w = image.shape[:2]
            if final_mask.shape[:2] != (h, w):
                final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Threshold to binary (Make sure text is White 255, Background is Black 0)
            # The model output might be grayscale probability.
            _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Dilation (Thicken the white text area to cover edges)
            kernel = np.ones((10, 10), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # 5. INPAINT
            cleaned = cv2.inpaint(image, binary_mask, 5, cv2.INPAINT_TELEA)
            
        else:
            print("WARNING: Could not find a valid mask map. Inpainting skipped.")
            cleaned = image

        # 6. ENCODE & CLEANUP
        success, png_img = cv2.imencode(".png", cleaned)
        if os.path.exists(filename): os.remove(filename)
        
        return Response(content=png_img.tobytes(), media_type="image/png")

    except Exception as e:
        if os.path.exists(filename): os.remove(filename)
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)