import os
import sys
import argparse
from unittest.mock import patch

# --- 1. SETUP & IMPORTS ---
try:
    import cleaner
    import manual_cleaner
    import extractor
    import translator_proofreader
    import typesetter
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you are running this script inside the 'backend' folder.")
    sys.exit(1)

def get_user_inputs():
    """Gets Folder, Language, and Starting Point."""
    parser = argparse.ArgumentParser(description="Manga Translation Pipeline")
    parser.add_argument("folder", nargs="?", help="Project folder path")
    parser.add_argument("--lang", choices=["ja", "ko", "zh"], help="Source language")
    parser.add_argument("--skip-manual", action="store_true", help="Skip the manual GUI step")
    args = parser.parse_args()

    # 1. Folder Selection
    folder_path = args.folder
    if not folder_path:
        print("\n" + "="*50)
        print("   MANGA PIPELINE: SETUP")
        print("="*50)
        folder_path = input("📂 Enter the Chapter Folder Path:\n>> ").strip().replace('"', '').replace("'", "")
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: The folder '{folder_path}' does not exist.")
        sys.exit(1)

    # 2. Language Selection
    lang_code = args.lang
    if not lang_code:
        print("\n🗣️  Select Source Language:")
        print("   1. Japanese (ja)")
        print("   2. Korean (ko)")
        print("   3. Chinese (zh)")
        choice = input(">> Enter number (1-3) or code: ").strip().lower()
        map_choice = {'1': 'ja', '2': 'ko', '3': 'zh', 'japanese': 'ja', 'korean': 'ko', 'chinese': 'zh'}
        lang_code = map_choice.get(choice, choice)

    if lang_code not in ['ja', 'ko', 'zh']:
        print(f"⚠️  Unknown language '{lang_code}'. Defaulting to Japanese (ja).")
        lang_code = 'ja'

    # 3. Start Point Selection
    print("\n🚀 Select Starting Point:")
    print("   1. Full Pipeline (Auto Clean -> Manual -> Extract -> Translate -> Typeset)")
    print("   2. Manual Cleaner (Skip Auto Clean)")
    print("   3. Text Extractor (Skip Cleaning)")
    print("   4. Translator (Skip Extraction)")
    print("   5. Typesetter Only")
    
    start_input = input(">> Enter number (1-5) [Default: 1]: ").strip()
    try:
        start_step = int(start_input) if start_input else 1
    except ValueError:
        start_step = 1

    return folder_path, lang_code, args.skip_manual, start_step

def configure_paths(chapter_folder):
    """
    Configures ABSOLUTE paths and creates necessary directories.
    """
    # 1. Resolve absolute path
    chapter_folder = os.path.abspath(chapter_folder)
    
    # 2. Check for 'raws'
    raws_path = os.path.join(chapter_folder, "raws")
    
    if os.path.isdir(raws_path):
        print(f"✅ Detected 'raws' folder. Reading from: {raws_path}")
        working_folder = raws_path
        output_root = chapter_folder 
    else:
        print(f"ℹ️  No 'raws' folder found. Treating '{chapter_folder}' as the main folder.")
        working_folder = chapter_folder
        output_root = chapter_folder

    print(f"   Outputs will be saved to root: {output_root}")

    # 3. Define Clean/Absolute Paths using os.path.normpath
    # This resolves 'Folder/../Other' into just 'Other'
    
    clean_dir = os.path.normpath(os.path.join(output_root, "../cleaned_output"))
    json_dir  = os.path.normpath(os.path.join(output_root, "../json_output"))
    final_dir = os.path.normpath(os.path.join(output_root, "../final_typeset_pages"))
    
    batch_text_file = os.path.normpath(os.path.join(output_root, "../batch_text_data.json"))
    batch_trans_file = os.path.normpath(os.path.join(output_root, "../batch_translated.json"))

    # 4. CRITICAL: Create directories immediately
    # This prevents 'Folder Not Found' errors in modules if steps are skipped
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # 5. Assign to Modules
    cleaner.OUTPUT_CLEAN_DIR = clean_dir
    cleaner.OUTPUT_JSON_DIR = json_dir
    
    manual_cleaner.OUTPUT_CLEAN_DIR = clean_dir
    manual_cleaner.OUTPUT_JSON_DIR = json_dir

    extractor.JSON_SUBFOLDER = json_dir
    extractor.OUTPUT_FILENAME = batch_text_file

    translator_proofreader.INPUT_FILE = batch_text_file
    translator_proofreader.OUTPUT_FILE = batch_trans_file

    typesetter.TRANSLATED_JSON = batch_trans_file
    typesetter.CLEANED_IMAGES_DIR = clean_dir
    typesetter.FINAL_OUTPUT_DIR = final_dir

    return working_folder, batch_text_file, batch_trans_file

def run_pipeline():
    # 1. Get Inputs
    chapter_folder, lang_code, skip_manual, start_step = get_user_inputs()
    
    # 2. Configure Paths and Create Folders
    working_folder, batch_text_file, batch_trans_file = configure_paths(chapter_folder)
    
    print("="*50)

    # --- STEP 1: AUTO CLEANER ---
    if start_step == 1:
        print("\n[1/5] Running Automatic Cleaner...")
        try:
            detector_model = cleaner.load_model()
            cleaner.process_folder(working_folder, detector_model)
        except Exception as e:
            print(f"❌ Cleaner failed: {e}")
            return
    else:
        print("\n[1/5] Skipping Auto Cleaner...")

    # --- STEP 2: MANUAL CLEANER ---
    should_run_manual = (start_step == 1 and not skip_manual) or (start_step == 2)
    
    if should_run_manual:
        print("\n[2/5] Running Manual Cleaner (GUI)...")
        print("   👉 Opening pages sequentially. Close the window to save and go to next.")
        try:
            if 'detector_model' not in locals():
                detector_model = cleaner.load_model()
            
            # CHECK: Does manual_cleaner actually have 'main_menu'?
            if hasattr(manual_cleaner, 'main_menu'):
                manual_cleaner.main_menu(working_folder, detector_model)
            elif hasattr(manual_cleaner, 'process_folder'):
                 manual_cleaner.process_folder(working_folder, detector_model)
            else:
                print("❌ Error: Could not find function 'main_menu' in manual_cleaner.py")
                print("   Please check your manual_cleaner.py for the correct function name.")
        except Exception as e:
            print(f"❌ Manual Cleaner error: {e}")
            # Non-critical, can continue
    else:
        print("\n[2/5] Skipping Manual Cleaner...")

    # --- STEP 3: EXTRACTOR ---
    if start_step <= 3:
        print("\n[3/5] Running Text Extractor...")
        try:
            ocr_input_map = {'ja': '1', 'ko': '2', 'zh': '3'}
            simulated_input = ocr_input_map.get(lang_code, '1')
            
            with patch('builtins.input', return_value=simulated_input):
                extractor.run_extraction(working_folder)
        except Exception as e:
            print(f"❌ Extractor failed: {e}")
            return
    else:
        print("\n[3/5] Skipping Extractor...")

    # --- STEP 4: TRANSLATOR ---
    if start_step <= 4:
        print("\n[4/5] Running Translator (Gemini)...")
        
        # Check if input file exists before running
        if not os.path.exists(batch_text_file):
            print(f"❌ Error: Input file for translator not found: {batch_text_file}")
            print("   Did the Extractor (Step 3) run successfully?")
            return

        try:
            translator_proofreader.run_translation(working_folder)
        except Exception as e:
            print(f"❌ Translator failed: {e}")
            return
    else:
        print("\n[4/5] Skipping Translator...")

    # --- STEP 5: TYPESETTER ---
    if start_step <= 5:
        print("\n[5/5] Running Typesetter...")

        # Check if input file exists before running
        if not os.path.exists(batch_trans_file):
            print(f"❌ Error: Input file for typesetter not found: {batch_trans_file}")
            print("   Did the Translator (Step 4) run successfully?")
            return

        try:
            typesetter.run_typesetting(working_folder)
        except Exception as e:
            print(f"❌ Typesetter failed: {e}")
            return

    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETE!")
    final_path = os.path.normpath(typesetter.FINAL_OUTPUT_DIR)
    print(f"   Output: {final_path}")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()