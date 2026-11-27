"""Generate synthetic Soil Health Card images for the app's "Try a sample" feature.

The cards are made up: no farmer, location or card number. They copy only the layout of the
"Soil Test Results" table so the OCR step can read them. Run from the repo root:

    python tools/make_sample_cards.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "samples"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/Library/Fonts/Times New Roman.ttf",
]

# name -> (N, P, K) in kg/ha
CARDS = {
    "sample_card_synthetic_maize.png": (160.00, 20.00, 150.00),
    "sample_card_synthetic_rice.png": (200.00, 8.00, 280.00),
}
# Full-layout cards (table + made-up farmer details panel)
FULL_CARDS = {
    "sample_card_synthetic_full.png": (240.00, 45.00, 210.00),
}

BANNER = "SYNTHETIC SAMPLE - not a real card"
BANNER_BG = (255, 235, 235)
BANNER_FG = (180, 0, 0)

COLUMNS = [("Sr.No.", 50), ("Parameter", 240), ("Test Value", 90), ("Unit", 80)]
ROW_H = 30
HEADER_BG = (153, 204, 102)
TITLE_BG = (255, 255, 0)
GRID = (60, 60, 60)


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def rows_for(n, p, k):
    return [
        ("1", "pH", "7.10", ""),
        ("2", "EC", "0.42", "dS/m"),
        ("3", "Organic Carbon (OC)", "0.55", "%"),
        ("4", "Available Nitrogen (N)", f"{n:.2f}", "kg/ha"),
        ("5", "Available Phosphorus (P)", f"{p:.2f}", "kg/ha"),
        ("6", "Available Potassium (K)", f"{k:.2f}", "kg/ha"),
        ("7", "Available Sulphur (S)", "18.00", "ppm"),
        ("8", "Available Zinc (Zn)", "1.10", "ppm"),
    ]


def make_card(path, n, p, k):
    """A sample card containing only the soil test table."""
    make_table(n, p, k).save(path)


def make_table(n, p, k):
    """The "Soil Test Results" table with a SYNTHETIC SAMPLE banner on top."""
    font, bold = load_font(18), load_font(19)
    width = sum(w for _, w in COLUMNS) + 2
    rows = rows_for(n, p, k)
    height = ROW_H * (len(rows) + 3) + 2
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, width - 1, ROW_H], fill=BANNER_BG)
    d.text((width / 2, ROW_H / 2), BANNER, font=bold, fill=BANNER_FG, anchor="mm")
    d.rectangle([0, ROW_H, width - 1, 2 * ROW_H], fill=TITLE_BG)
    d.text((width / 2, 1.5 * ROW_H), "Soil Test Results", font=bold, fill="black", anchor="mm")

    y = 2 * ROW_H
    d.rectangle([0, y, width - 1, y + ROW_H], fill=HEADER_BG)
    for cells in [[c for c, _ in COLUMNS]] + [list(r) for r in rows]:
        x = 0
        for text, (_, col_w) in zip(cells, COLUMNS):
            d.rectangle([x, y, x + col_w, y + ROW_H], outline=GRID, width=1)
            numeric = text.replace(".", "", 1).isdigit()
            if numeric:
                d.text((x + col_w - 6, y + ROW_H / 2), text, font=font, fill="black", anchor="rm")
            else:
                d.text((x + 6, y + ROW_H / 2), text, font=font, fill="black", anchor="lm")
            x += col_w
        y += ROW_H
    return img


# Right-hand panel of a full card. Every detail is made up.
FULL_CARD_PANEL = [
    ("title", "Soil Health Card"),
    ("row", "Card Number", "XX/0000-00/0000000/0"),
    ("title", "Farmer's Details"),
    ("row", "Farmer Name", "Sample Farmer"),
    ("row", "Address", "XXXXX"),
    ("row", "Mobile No.", "XXXX-XXXX-XX"),
    ("title", "Soil Sample Details"),
    ("row", "Date of Collection", "01-01-2026"),
    ("row", "Farm Size", "1.00 Acre"),
    ("row", "Soil Type", "Red soil"),
]


def make_full_card(path, n, p, k):
    """A full card: the soil test table on the left, made-up farmer details on the right."""
    font, bold = load_font(17), load_font(19)
    table = make_table(n, p, k)
    panel_w, gap, label_w = 400, 30, 170
    width = table.width + gap + panel_w + 20
    height = max(table.height, ROW_H * (len(FULL_CARD_PANEL) + 1)) + 20
    img = Image.new("RGB", (width, height), (232, 245, 220))
    img.paste(table, (10, 10))

    d = ImageDraw.Draw(img)
    x0, y = 10 + table.width + gap, 10
    d.rectangle([x0, y, x0 + panel_w, y + ROW_H], fill=BANNER_BG)
    d.text((x0 + panel_w / 2, y + ROW_H / 2), BANNER, font=bold, fill=BANNER_FG, anchor="mm")
    y += ROW_H
    for item in FULL_CARD_PANEL:
        if item[0] == "title":
            d.rectangle([x0, y, x0 + panel_w, y + ROW_H], fill=TITLE_BG, outline=GRID)
            d.text((x0 + panel_w / 2, y + ROW_H / 2), item[1], font=bold, fill="black", anchor="mm")
        else:
            d.rectangle([x0, y, x0 + label_w, y + ROW_H], fill="white", outline=GRID)
            d.rectangle([x0 + label_w, y, x0 + panel_w, y + ROW_H], fill="white", outline=GRID)
            d.text((x0 + 6, y + ROW_H / 2), item[1], font=font, fill="black", anchor="lm")
            d.text((x0 + label_w + 6, y + ROW_H / 2), item[2], font=font, fill="black", anchor="lm")
        y += ROW_H
    img.save(path)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name, npk in CARDS.items():
        make_card(OUT_DIR / name, *npk)
        print("wrote", OUT_DIR / name)
    for name, npk in FULL_CARDS.items():
        make_full_card(OUT_DIR / name, *npk)
        print("wrote", OUT_DIR / name)
