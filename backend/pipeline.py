import cleaner, ocr, translator, typesetter

def run(upload):
    cleaned = cleaner.run(upload)
    ocr_text = ocr.run(upload)
    translated = translator.run(ocr_text["raw_text"])
    final = typesetter.run(upload, translated["translation"])

    return {
        "cleaned": cleaned,
        "ocr": ocr_text,
        "translation": translated,
        "final_typeset": final
    }
