from paddleocr import PaddleOCR

# Load Japanese OCR model once
ocr = PaddleOCR(lang='japan', use_angle_cls=True)


def extract_text_from_crop(crop):
    result = ocr.ocr(crop, det=False, cls=False)
    if not result:
        return ""
    return "\n".join([line[1][0] for line in result])
