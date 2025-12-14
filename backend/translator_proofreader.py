import json
import os
import sys
import copy
from dotenv import load_dotenv

# --- CONFIGURATION ---
INPUT_FILE = "batch_text_data.json"
OUTPUT_FILE = "batch_translated.json"
MODEL_NAME = "gemini-2.5-flash"  # ⚠️ NOTE: If this model doesn't exist, translation will fail.

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

# ==========================================
# STAGE 1: THE TRANSLATOR (SMART MERGE FIX)
# ==========================================
def run_translation(project_folder):
    input_path = os.path.join(project_folder, INPUT_FILE)
    output_path = os.path.join(project_folder, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"❌ Error: Cannot find {INPUT_FILE}")
        return

    # 1. ALWAYS Load the Fresh Source Data (This is the Source of Truth)
    # We start with the input file so we catch any NEW pages/blocks you added.
    with open(input_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
    
    # 2. Smart Resume: Merge existing translations into the fresh data
    # Instead of replacing batch_data, we fill in the blanks using the old file.
    needs_translation = False
    
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_translated_data = json.load(f)
            
            # Create a map: "Filename|BlockID" -> "Translated Text"
            existing_translations = {}
            for page in old_translated_data:
                fname = page['file_name']
                for block in page.get('blocks', []):
                    # Only save if there is actual translated text
                    if block.get('text_translated'):
                        key = f"{fname}|{block['id']}"
                        existing_translations[key] = block['text_translated']
            
            # Apply these old translations to our FRESH batch_data
            mapped_count = 0
            for page in batch_data:
                fname = page['file_name']
                for block in page['blocks']:
                    key = f"{fname}|{block['id']}"
                    
                    # A. If we have a translation from before, keep it
                    if key in existing_translations:
                        block['text_translated'] = existing_translations[key]
                        mapped_count += 1
                    
                    # B. If we DON'T, check if we need one (ignore empty raw blocks)
                    elif block.get('text_raw', '').strip():
                        # We found a block with raw text but NO translation -> We must run AI
                        needs_translation = True
            
            print(f"🔄 Merged {mapped_count} existing translations into fresh input data.")
            
        except Exception as e:
            print(f"⚠️ Old translation file corrupt ({e}). Starting fresh.")
            needs_translation = True
    else:
        needs_translation = True

    # 3. Run Translation only if we found gaps
    if needs_translation:
        print(f"⚡ Found untranslated blocks. Starting Translation Agent...")
        batch_data = execute_translation_ai(batch_data)
        
        # Save Draft
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
    else:
        print("✅ All blocks are already translated. Skipping to Proofreader.")

    # 4. Run Proofreader
    run_proofreading_agent(batch_data, project_folder)

def execute_translation_ai(batch_data):
    if DUMMY_MODE: return batch_data

    # Build Prompt
    # OPTIMIZATION: Only ask AI to translate blocks that are currently EMPTY
    prompt_blocks_count = 0
    prompt_script = "Translate the following Manga/Manhwa text blocks.\n"
    
    for page in batch_data:
        # Check if page has any untranslated blocks
        page_has_work = False
        for block in page["blocks"]:
            if not block.get("text_translated") and block.get("text_raw", "").strip():
                page_has_work = True
                break
        
        if page_has_work:
            prompt_script += f"\n--- Page: {page['file_name']} ---\n"
            for block in page["blocks"]:
                # Only include block in prompt if it lacks translation
                if not block.get("text_translated") and block.get("text_raw", "").strip():
                    prompt_script += f"[ID: {block['id']}] {block['text_raw']}\n"
                    prompt_blocks_count += 1
    
    if prompt_blocks_count == 0:
        return batch_data

    system_instruction = """
    You are an elite Manga/Manhwa Translator.
    
    ### CRITICAL RULES:
    1. **1:1 Mapping**: You MUST output exactly one translation for every [ID] provided. DO NOT merge IDs.
       - If ID:1 and ID:2 are parts of one sentence, translate ID:1 as the first half and ID:2 as the second half.
    
    2. **Natural Dialogue**: 
       - Fix dropped subjects (I, You, He, She) based on context.
       - Use contractions (I'm, It's, Don't) for speech.
       - Avoid robotic phrasing like "My body moved on its own." -> "My body... it moved by itself!"
    
    3. **SFX & UI**: 
       - Sound effects -> English equivalents (e.g. "Kwang!" -> "BOOM!").
       - UI/Status Windows -> Keep it literal and concise.
    
    4. **Garbage Handling**:
       - If the text is random OCR noise (e.g., "l|l;", "no_text"), return "[DELETE]".
    
    ### OUTPUT FORMAT:
    Return strictly a JSON object:
    {
      "page_filename": {
        "block_id": "Translation string"
      }
    }
    """

    print(f"   ...Sending {prompt_blocks_count} blocks to Translator Agent...")
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt_script)
        translated_map = json.loads(clean_json_text(response.text))
    except Exception as e:
        print(f"❌ Translator Failed (Check MODEL_NAME): {e}")
        return batch_data

    # Apply Translations
    for page in batch_data:
        fname = page["file_name"]
        page_map = translated_map.get(fname, {})
        
        for block in page["blocks"]:
            # Only update if we don't have a translation yet
            if block.get("text_translated"): continue

            b_id = str(block["id"])
            trans_text = page_map.get(b_id, "")
            
            # Basic Cleaning
            if trans_text.strip() == "[DELETE]": continue
            if "SFX:" in trans_text: trans_text = trans_text.replace("SFX:", "").strip()
            
            if trans_text:
                block["text_translated"] = trans_text
            elif block['text_raw'].strip():
                # Only mark missing if raw text existed
                block["text_translated"] = "[MISSING TRANSLATION]"
    
    return batch_data

# ==========================================
# STAGE 2: THE AGGRESSIVE PROOFREADER (FIXED WITH RETRY)
# ==========================================
def run_proofreading_agent(batch_data, project_folder):
    print("\n" + "="*50)
    print("🕵️  STARTING AGGRESSIVE PROOFREADER AGENT")
    print("="*50)
    
    if DUMMY_MODE: 
        print("Skipping Proofreader (No API Key)")
        return

    # 1. Prepare Context for Proofreader
    proof_prompt = "Review this translation for natural English flow and logic.\n"
    lookup_map = {} 

    for page in batch_data:
        proof_prompt += f"\n=== FILE: {page['file_name']} ===\n"
        for block in page["blocks"]:
            key = f"{page['file_name']}|{block['id']}"
            lookup_map[key] = block
            safe_tl = block.get('text_translated', '[EMPTY]')
            proof_prompt += f"[ID: {block['id']}]\n   RAW: {block['text_raw']}\n   TL:  {safe_tl}\n"

    system_instruction = """
    You are a strict, aggressive Manga Editor and Proofreader.
    Your job is to read the dialogue flow and flag ANY lines that sound unnatural, robotic, or confusing.

    ### HOW TO JUDGE:
    1. **Flow Check**: Does the conversation make sense? 
    2. **Tone Check**: Do characters sound like humans?
    3. **Consistency**: Are pronouns consistent?

    ### OUTPUT FORMAT:
    Return a strictly valid JSON LIST. 
    **CRITICAL**: Escape all double quotes inside strings (e.g. "He said \\"Hello\\"") or the JSON will break.
    
    [
      {
        "file_name": "exact_filename.jpg",
        "block_id": 1,
        "reason": "Sounds too robotic.",
        "suggestion": "Better phrasing."
      }
    ]
    """

    print("   ...Sending to Proofreader Agent (This analyzes the whole chapter flow)...")
    
    suggestions = []
    MAX_RETRIES = 3
    
    # --- RETRY LOOP START ---
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=system_instruction,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(proof_prompt)
            
            # Clean and Parse
            text_cleaned = clean_json_text(response.text)
            suggestions = json.loads(text_cleaned)
            
            # If we get here, it worked!
            break
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Attempt {attempt}/{MAX_RETRIES} Failed: AI returned invalid JSON. Retrying...")
            if attempt == MAX_RETRIES:
                print(f"❌ Proofreader failed after {MAX_RETRIES} attempts. Skipping proofreading step.")
                print(f"   (Error details: {e})")
                return
        except Exception as e:
            print(f"❌ Critical API Error: {e}")
            return
    # --- RETRY LOOP END ---

    if not suggestions:
        print("✅ Proofreader found 0 issues! (Or returned empty list)")
        return

    # 2. Interactive Review Loop
    print(f"\n📢 The Proofreader flagged {len(suggestions)} issues.")
    print("Type 'y' to accept, 'n' to reject, 'e' to edit manually, or 'a' to accept ALL remaining.\n")

    auto_accept = False
    
    for i, item in enumerate(suggestions):
        fname = item.get('file_name')
        bid = item.get('block_id')
        reason = item.get('reason', 'No reason provided')
        better = item.get('suggestion', '')
        
        # Find original block data
        key = f"{fname}|{bid}"
        if key not in lookup_map: continue
        current_block = lookup_map[key]
        current_text = current_block.get('text_translated', '')

        if auto_accept:
            current_block['text_translated'] = better
            print(f"[{i+1}/{len(suggestions)}] Auto-Accepted fix for {fname} (ID {bid})")
            continue

        print("-" * 60)
        print(f"🚩 ISSUE [{i+1}/{len(suggestions)}] in {fname} (ID {bid})")
        print(f"   RAW:      {current_block['text_raw']}")
        print(f"   CURRENT:  {current_text}")
        print(f"   REASON:   {reason}")
        print(f"   PROPOSED: \033[92m{better}\033[0m") 
        print("-" * 60)

        while True:
            choice = input(">> [y]es / [n]o / [e]dit / [a]ccept-all: ").strip().lower()
            
            if choice == 'y':
                current_block['text_translated'] = better
                print("   -> Updated.")
                break
            elif choice == 'n':
                print("   -> Ignored.")
                break
            elif choice == 'e':
                new_input = input("   Enter your custom translation: ").strip()
                if new_input:
                    current_block['text_translated'] = new_input
                    print("   -> Updated with custom text.")
                break
            elif choice == 'a':
                auto_accept = True
                current_block['text_translated'] = better
                print("   -> Updated. (Auto-accepting rest...)")
                break

    # 3. Final Save
    output_path = os.path.join(project_folder, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print(f"✅ Proofreading Complete! Saved to {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    while True:
        path = input("\nEnter FOLDER PATH (or 'q'):\n>> ").strip().replace('"', '').replace("'", "")
        if path.lower() == 'q': break
        if os.path.exists(path): run_translation(path)
        else: print("❌ Invalid path")