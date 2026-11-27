"""Benchmark the Soil Health Card reader on every sample card, at three image sizes.

Prints the Tesseract version and, for each card and size, which values were read correctly, left
blank, or read WRONG. With --debug it also prints what Tesseract saw around each value, which is
how a misread such as K 69.00 -> 65.00 can be traced on a particular Tesseract build.

Exit code 1 if any value is read wrong (a blank is acceptable: the app asks the user to type it).

    python tools/ocr_benchmark.py [--debug]
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402

EXPECTED = {
    "sample_card_real_full.jpg": {"N": 305.0, "P": 36.0, "K": 69.0},
    "sample_card_real_cropped.jpg": {"N": 305.0, "P": 36.0, "K": 69.0},
    "sample_card_synthetic_full.png": {"N": 240.0, "P": 45.0, "K": 210.0},
    "sample_card_synthetic_maize.png": {"N": 160.0, "P": 20.0, "K": 150.0},
    "sample_card_synthetic_rice.png": {"N": 200.0, "P": 8.0, "K": 280.0},
}
SCALES = (1.0, 0.75, 1.5)   # phone screenshots and photos arrive at different resolutions


def load(name):
    return cv2.imdecode(np.fromfile(str(core.SAMPLES_DIR / name), dtype=np.uint8), cv2.IMREAD_COLOR)


def resize(img, scale):
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def debug(img):
    """Show the OCR lines that mention a nutrient, and what the digits-only re-read returns, per view."""
    views = {"enlarged gray": core._gray(core._enlarge(img)), "grid lines removed": core._without_table_lines(img)}
    for view_name, view in views.items():
        print(f"    [{view_name}]")
        for words in core._ocr_lines(view):
            texts = [word[0] for word in words]
            line = " ".join(texts)
            for key, label in core.NUTRIENT_LABELS.items():
                at = next((i for i, text in enumerate(texts) if re.search(label, text, re.IGNORECASE)), None)
                if at is None:
                    continue
                value_word = next((w for w in words[at + 1:]
                                   if re.search(r"\d", w[0]) and not core.RANGE_PATTERN.search(w[0])), None)
                reread = core._reread_digits(view, value_word) if value_word else []
                print(f"      {key}: line={line!r} value_word={value_word[0] if value_word else None!r} "
                      f"digit_reread={reread}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--debug", action="store_true", help="print what Tesseract saw around each value")
    args = parser.parse_args()

    core.configure_tesseract()
    print("Tesseract", core.pytesseract.get_tesseract_version())
    total = right = blank = wrong = 0
    for name, expected in EXPECTED.items():
        img = load(name)
        for scale in SCALES:
            found = core.read_npk(resize(img, scale))
            marks = []
            for key, value in expected.items():
                total += 1
                got = found.get(key)
                if got == value:
                    right += 1
                    marks.append(f"{key} ok")
                elif got is None:
                    blank += 1
                    marks.append(f"{key} blank")
                else:
                    wrong += 1
                    marks.append(f"{key} WRONG ({got} instead of {value})")
            print(f"{name:34s} x{scale:<4} " + ", ".join(marks))
            if args.debug and (scale == 1.0 or any("WRONG" in m for m in marks)):
                debug(resize(img, scale))
    print(f"\n{right}/{total} right, {blank} blank, {wrong} wrong")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
