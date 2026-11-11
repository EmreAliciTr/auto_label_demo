"""Create a synthetic OCR sample."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main():
    from PIL import Image, ImageDraw, ImageFont
    output = Path('inputs/etiketleme_ornegi.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (800, 300), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 770, 270), outline='#333333', width=3)
    draw.text((80, 80), 'HELLO 123', font=ImageFont.load_default(size=70), fill='black')
    draw.text((80, 185), 'TEST PRODUCT', font=ImageFont.load_default(size=32), fill='black')
    image.save(output, 'PNG')
    print('Synthetic sample created in inputs/etiketleme_ornegi.png')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Sample generation failed.')
        raise SystemExit(1) from None
