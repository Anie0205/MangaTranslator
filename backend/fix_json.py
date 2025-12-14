import json

# Load the dirty json
with open("../Manhwa/MagicalGirl78/Chapter3/batch_translated.json", "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0
for page in data:
    # Filter out blocks that have "[DELETE]" as the translation
    valid_blocks = []
    for block in page["blocks"]:
        text = block.get("text_translated", "").strip()
        if text != "[DELETE]" and text != "":
            valid_blocks.append(block)
        else:
            count += 1
    page["blocks"] = valid_blocks

# Save the clean json
with open("batch_translated.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Fixed! Removed {count} garbage blocks.")