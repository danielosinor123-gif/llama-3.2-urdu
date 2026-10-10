from filter_urdu import check_text, has_devanagari, normalize_and_check, normalize_text


def test_devanagari_rejection():
    text = "یہ متن ہے और یہ بھی۔"
    assert has_devanagari(text)
    result = check_text(text)
    assert not result["ok"]
    assert "devanagari_contamination" in result["reasons"]


def test_clean_urdu_passes():
    text = "یہ ایک صاف اردو جملہ ہے۔ آج موسم اچھا ہے۔"
    result = check_text(text)
    assert result["ok"], result["reasons"]


def test_empty_rejected():
    result = check_text("   ")
    assert not result["ok"]
    assert "empty" in result["reasons"]


def test_english_heavy_rejected():
    result = check_text("The weather is nice today and we will go out to play cricket with friends.")
    assert not result["ok"]
    assert any("low_arabic_script_ratio" in r for r in result["reasons"])


def test_normalization_removes_zero_width():
    text = "اردو\u200bمتن\u200cہے"
    normalized = normalize_text(text)
    assert "\u200b" not in normalized
    assert "\u200c" not in normalized


def test_normalization_collapses_whitespace():
    normalized = normalize_text("اردو   متن\n\nہے")
    assert normalized == "اردو متن ہے"


def test_normalize_and_check_round_trip():
    text, result = normalize_and_check("  یہ ایک جملہ ہے۔  ")
    assert result["ok"]
    assert text == "یہ ایک جملہ ہے۔"


def test_hindi_ism_blocklist():
    result = check_text("یہ پرکرتی کی بات ہے۔")
    assert not result["ok"]
    assert any(r.startswith("hindi_ism:") for r in result["reasons"])
