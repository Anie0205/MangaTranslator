import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using Google Gemini API
    """
    try:
        prompt = f"""Translate the following {source_lang} text to {target_lang}.
Only provide the translation, no explanations or additional text.

Text: {text}

Translation:"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"[Translation error: {str(e)}]"