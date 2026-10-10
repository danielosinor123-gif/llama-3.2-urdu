import re
import unicodedata

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
PRESENTATION_FORMS_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]")
ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
LATIN_RE = re.compile(r"[A-Za-z]")

HINDI_ISM_BLOCKLIST = [
    "ورش",
    "پرکرتی",
    "جیون",
    "سوریہ",
    "منوش",
    "لوک منش",
    "شریدا",
    "پرماتما",
    "درگا ماتا",
    "پنجاب کے بارے",
]

MIN_ARABIC_RATIO = 0.70
MAX_LATIN_RATIO = 0.40


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = unicodedata.normalize("NFKC", text)
    text = PRESENTATION_FORMS_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def arabic_script_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P") and not c.isdigit()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if ARABIC_LETTER_RE.match(c))
    return arabic / len(letters)


def latin_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if LATIN_RE.match(c)) / len(letters)


def find_hindi_isms(text: str) -> list:
    found = []
    for term in HINDI_ISM_BLOCKLIST:
        if term in text:
            found.append(term)
    return found


def check_text(text: str, min_arabic_ratio: float = MIN_ARABIC_RATIO) -> dict:
    reasons = []
    if not text or not text.strip():
        reasons.append("empty")
    if has_devanagari(text):
        reasons.append("devanagari_contamination")
    ratio = arabic_script_ratio(text)
    if text.strip() and ratio < min_arabic_ratio:
        reasons.append(f"low_arabic_script_ratio({ratio:.2f})")
    lr = latin_ratio(text)
    if text.strip() and lr > MAX_LATIN_RATIO:
        reasons.append(f"high_latin_ratio({lr:.2f})")
    isms = find_hindi_isms(text)
    if isms:
        reasons.append("hindi_ism:" + "|".join(isms))
    return {"ok": not reasons, "reasons": reasons, "arabic_ratio": ratio, "latin_ratio": lr}


def normalize_and_check(text: str) -> tuple:
    text = normalize_text(text)
    result = check_text(text)
    return text, result
