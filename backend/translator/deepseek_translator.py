import subprocess

def translate_text(text: str) -> str:
    prompt = (
        "Translate the following Japanese manga text into natural English. "
        "Keep the tone casual and manga-like, not literal.\n\n"
        f"{text}\n"
    )

    result = subprocess.run(
        ["ollama", "run", "deepseek-r1:1.5b"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )

    return result.stdout.decode().strip()
