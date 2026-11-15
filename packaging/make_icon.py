"""Build the app's icons from a logo image.

    python packaging/make_icon.py path/to/logo.png

Pads the logo onto a transparent square (so a wide logo isn't stretched), then writes
  owlfluids/app/icons/owlfluids.png  256x256, the window and taskbar icon
  packaging/owlfluids.ico            16-256 px, the exe, installer and shortcut icon
Commit both. Needs Pillow (pip install pillow).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]


def square(img: Image.Image, size: int = 256, margin: float = 0.04) -> Image.Image:
    img = img.convert("RGBA")
    bbox = img.getbbox()  # trim transparent edges
    if bbox:
        img = img.crop(bbox)
    inner = round(size * (1 - 2 * margin))
    img.thumbnail((inner, inner), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    icon = square(Image.open(argv[0]))
    png = ROOT / "owlfluids" / "app" / "icons" / "owlfluids.png"
    ico = ROOT / "packaging" / "owlfluids.ico"
    png.parent.mkdir(parents=True, exist_ok=True)
    icon.save(png)
    icon.save(ico, sizes=[(s, s) for s in SIZES])
    print(f"wrote {png.relative_to(ROOT)} and {ico.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
