from PIL import Image
import uuid, os

def load_img(upload):
    return Image.open(upload.file).convert("RGB")

def save_img(img, suffix="output"):
    filename = f"out_{suffix}_{uuid.uuid4().hex}.png"
    path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    img.save(path)
    return path
