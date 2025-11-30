import json
import os
import re
import sys
from dotenv import load_dotenv

# --- CONFIGURATION ---
INPUT_FILE = "batch_text_data.json"
OUTPUT_FILE = "batch_translated.json"

# --- LOAD ENVIRONMENT VARIABLES ---
# This looks for a .env file in the current folder and loads it
load_dotenv()

# Get key from environment (returns None if not found)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set DUMMY_MODE automatically based on whether the key exists
DUMMY_MODE = False
if not GOOGLE_API_KEY:
    print("⚠️  WARNING: No GOOGLE_API_KEY found in .env file.")
    print("   Running in DUMMY MODE (Placeholder translations only).")
    DUMMY_MODE = True

try:
    import google.generativeai as genai
except ImportError:
    if not DUMMY_MODE:
        print("CRITICAL: 'google-generativeai' not found. Run: pip install google-generativeai")
        sys.exit(1)

def clean_json_response(response_text):
    """Removes markdown code blocks if the AI adds them"""
    clean_text = re.sub(r"```json\s*", "", response_text)
    clean_text = re.sub(r"```\s*$", "", clean_text)
    return clean_text.strip()

def run_translation(project_folder):
    input_path = os.path.join(project_folder, INPUT_FILE)
    output_path = os.path.join(project_folder, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {INPUT_FILE} in {project_folder}")
        return

    # 1. Load Data
    with open(input_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)

    print(f"Loaded {len(batch_data)} pages. Preparing translation...")

    # 2. Build Prompt (Context-Aware)
    prompt_script = ""
    for page in batch_data:
        prompt_script += f"\n--- Page: {page['file_name']} ---\n"
        for block in page["blocks"]:
            prompt_script += f"[ID: {block['id']}] {block['text_raw']}\n"

    system_instruction = """
    You are a professional Manga Translator.
    1. Translate the text to natural, punchy English suitable for comics.
    2. Context Awareness: Use the full chapter context to determine who is speaking.
    3. Output Format: Return a JSON object where keys are Filenames, and values are objects mapping ID to Translation.
    
    Example Structure:
    {
      "1.jpg": { "1": "Hello", "2": "World" },
      "2.jpg": { "1": "Goodbye" }
    }
    """

    # 3. Call Gemini
    translated_map = {}

    if DUMMY_MODE:
        print("⚠️ DUMMY MODE: Generating placeholder translations...")
        for page in batch_data:
            page_map = {}
            for block in page["blocks"]:
                page_map[str(block["id"])] = f"[TR] {block['text_raw'][:15]}..."
            translated_map[page["file_name"]] = page_map
    else:
        print("⚡ Sending to Gemini 2.5 Flash...")
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Use 'gemini-2.5-flash' for speed and cost efficiency
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_instruction,
                generation_config={"response_mime_type": "application/json"}
            )
            
            response = model.generate_content(prompt_script)
            
            # Parse JSON
            translated_map = json.loads(response.text)
            
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            return

    # 4. Merge Data
    translation_count = 0
    for page in batch_data:
        fname = page["file_name"]
        if fname in translated_map:
            page_translations = translated_map[fname]
            for block in page["blocks"]:
                b_id = str(block["id"])
                if b_id in page_translations:
                    block["text_translated"] = page_translations[b_id]
                    translation_count += 1

    # 5. Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print("-" * 40)
    print(f"✅ Translation Complete!")
    print(f"Filled {translation_count} bubbles.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        user_input = input("Enter FOLDER PATH (or 'q'):\n>> ").strip()
        if user_input.lower() == 'q': break
        
        folder_path = user_input.replace('"', '').replace("'", "")
        if os.path.exists(folder_path):
            run_translation(folder_path)
        else:
            print("❌ Invalid path.")