import json
import os
import re
import sys
from dotenv import load_dotenv

# --- CONFIGURATION ---
INPUT_FILE = "batch_text_data.json"
OUTPUT_FILE = "batch_translated.json"
MODEL_NAME = "gemini-2.5-flash" 

# --- SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️  WARNING: No GOOGLE_API_KEY found in .env file.")
    DUMMY_MODE = True
else:
    DUMMY_MODE = False
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
    except ImportError:
        print("CRITICAL: Run 'pip install google-generativeai'")
        sys.exit(1)

def clean_json_text(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

def run_translation(project_folder):
    input_path = os.path.join(project_folder, INPUT_FILE)
    output_path = os.path.join(project_folder, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"❌ Error: Cannot find {INPUT_FILE}")
        return

    # 1. Load Data
    with open(input_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
    print(f"Loaded {len(batch_data)} pages. Starting Context-Aware Translation...")

    # 2. Build ONE Big Prompt (Gives the AI the full story context)
    prompt_script = "Translate the following story. The format is [Page Name] -> [Block ID] -> Text.\n"
    for page in batch_data:
        prompt_script += f"\n--- Page: {page['file_name']} ---\n"
        for block in page["blocks"]:
            # We include the ID so the AI knows where to put the translation
            prompt_script += f"[ID: {block['id']}] {block['text_raw']}\n"

    # 3. The "Anti-Machine-Translation" System Prompt
    system_instruction = """
    You are an expert Manhwa/Manga Localizer (NOT just a translator).
    Your goal is to convert Korean/Japanese text into **Natural, Flowing English Dialogue**.

    ### CRITICAL RULES FOR PRONOUNS & GENDER:
    1. **Fix Dropped Subjects**: Asian languages often drop "I", "You", "He", "She". You MUST infer who is speaking based on the conversation flow.
       - If Character A asks a question, Character B's reply is likely about "I" (themselves) or "You" (A).
       - Do not flip-flop. If a character was "She" on the previous page, she remains "She".
    
    2. **No "Machine" Phrasing**: 
       - BAD: "My body is moving on its own!" (Robotic)
       - GOOD: "What?! Why can't I control my body?!" (Natural)
       - BAD: "Do not disturb me."
       - GOOD: "Get out of my way!" or "Leave me alone!"

    3. **Context Awareness**: 
       - If text is short and in a box (like "Inventory"), translate it as UI.
       - If text looks like a sound (e.g., "Kwang!"), write "Boom!" or "Crash!".
    
    4. **Garbage**: If text is random OCR noise (e.g. "@@", ";;"), return "[DELETE]".

    ### OUTPUT FORMAT:
    Return strictly a JSON object mapping filenames to IDs.
    {
      "page1.jpg": {
        "1": "Translation here",
        "2": "Translation here"
      }
    }
    """

    # 4. Call AI
    translated_map = {}
    
    if DUMMY_MODE:
        print("⚠️ DUMMY MODE: Skipping AI.")
    else:
        print("⚡ Sending full story context to Gemini...")
        try:
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=system_instruction,
                generation_config={"response_mime_type": "application/json"}
            )
            # We assume the context fits in the window (Flash has 1M context, usually plenty)
            response = model.generate_content(prompt_script)
            clean_text = clean_json_text(response.text)
            translated_map = json.loads(clean_text)
        except Exception as e:
            print(f"❌ AI Error: {e}")
            # print(response.text) # Uncomment to debug if JSON fails
            return

    # 5. Apply Translations
    fill_count = 0
    for page in batch_data:
        fname = page["file_name"]
        page_map = translated_map.get(fname, {})
        
        valid_blocks = []
        for block in page["blocks"]:
            b_id = str(block["id"])
            trans_text = page_map.get(b_id, "")
            
            # Cleaning
            if trans_text in ["[DELETE]", ""] or "SFX:" in trans_text:
                 trans_text = trans_text.replace("SFX:", "").strip()
            
            if not trans_text: continue

            block["text_translated"] = trans_text
            valid_blocks.append(block)
            fill_count += 1
            
        page["blocks"] = valid_blocks

    # 6. Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print("-" * 40)
    print(f"✅ Localization Complete! ({fill_count} blocks)")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    while True:
        path = input("\nEnter FOLDER PATH (or 'q'):\n>> ").strip().replace('"', '').replace("'", "")
        if path.lower() == 'q': break
        if os.path.exists(path): run_translation(path)