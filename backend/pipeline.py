import cleaner, backend.extractor as extractor, backend.translator_proofreader as translator_proofreader, typesetter

def run(upload):
    cleaned = cleaner.run(upload)
    ocr_text = extractor.run(upload)
    translated = translator_proofreader.run(ocr_text["raw_text"])
    final = typesetter.run(upload, translated["translation"])

    return {
        "cleaned": cleaned,
        "ocr": ocr_text,
        "translation": translated,
        "final_typeset": final
    }
