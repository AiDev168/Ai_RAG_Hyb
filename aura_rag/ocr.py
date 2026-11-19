"""OCR text helpers (cleaning, image pre-processing, language-pack resolution)."""

import re

from .dependencies import Pytesseract


OCR_LANG_INSTALL_HINT = (
    "Persian pack: 'apt install tesseract-ocr-fas' (Debian/Ubuntu), "
    "'brew install tesseract-lang' (macOS), or tick 'Persian' in the Windows "
    "tesseract installer (or download fas.traineddata into the tessdata folder)."
)

_OCR_BIDI_RE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")
_ARABIC_TO_PERSIAN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹")


def _clean_ocr_text(text: str, persian: bool) -> str:
    """Tidy raw tesseract output. Always: drop invisible bidi marks / soft
    hyphens and trim whitespace. For Persian: map Arabic-form letters and
    digits to their Persian forms and drop tatweel, so the text matches what
    users type and what the FTS5 keyword index normalizes to. ZWNJ is kept
    (it is meaningful inside Persian words)."""
    if not text:
        return ""
    text = _OCR_BIDI_RE.sub("", text)
    if persian:
        text = text.replace("\u0640", "")                                   # tatweel
        text = (text.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc")  # ي ى -> ی
                    .replace("\u0643", "\u06a9"))                           # ك -> ک
        text = text.translate(_ARABIC_TO_PERSIAN_DIGITS)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _prepare_ocr_image(image):
    """Grayscale + autocontrast, and upscale small images: tesseract is much
    more accurate on Persian's small dots/joins at roughly 300 DPI."""
    from PIL import Image, ImageOps
    gray = ImageOps.autocontrast(image.convert("L"))
    longest = max(gray.size)
    if 0 < longest < 1800:
        scale = min(3.0, 1800 / longest)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), resample)
    return gray


def _resolve_ocr_languages(requested: str):
    """Split a 'fas+eng' style string and keep only the language packs that
    tesseract actually has. Returns (usable '+'-joined string, missing list).
    If the installed packs can't be listed (e.g. tesseract itself is missing)
    the request is passed through so OCR reports the real error."""
    wanted = [l.strip() for l in (requested or "fas+eng").split("+") if l.strip()] or ["eng"]
    try:
        installed = set(Pytesseract.get_languages(config=""))
    except Exception:
        return "+".join(wanted), []
    usable = [l for l in wanted if l in installed]
    missing = [l for l in wanted if l not in installed]
    return "+".join(usable), missing
