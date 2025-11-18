from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

FONT_PATH = "fonts/AnimeAce.ttf"  # make sure this exists

def overlay_text_on_bubble(img, box, text):
    x1, y1, x2, y2 = box

    pil_img = Image.fromarray(img)
    bubble = pil_img.crop((x1, y1, x2, y2))

    draw = ImageDraw.Draw(bubble)
    font = ImageFont.truetype(FONT_PATH, 26)

    wrapped = textwrap.fill(text, width=12)
    w, h = draw.textsize(wrapped, font=font)

    draw.text(
        ((bubble.width - w) / 2, (bubble.height - h) / 2),
        wrapped,
        fill="black",
        font=font
    )

    pil_img.paste(bubble, (x1, y1))
    return np.array(pil_img)
