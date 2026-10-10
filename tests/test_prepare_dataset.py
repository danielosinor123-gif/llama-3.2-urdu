import json
from pathlib import Path

from prepare_dataset import parse_csv, parse_jsonl, parse_plain_text


def test_parse_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"prompt": "سوال", "response": "جواب"}, ensure_ascii=False) + "\n"
        + json.dumps({"instruction": "سوال 2", "completion": "جواب 2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = parse_jsonl(path)
    assert len(rows) == 2
    assert rows[0] == {"prompt": "سوال", "response": "جواب"}
    assert rows[1] == {"prompt": "سوال 2", "response": "جواب 2"}


def test_parse_jsonl_invalid_row(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prompt": "صرف سوال"}\n', encoding="utf-8")
    try:
        parse_jsonl(path)
        assert False, "should have raised"
    except ValueError:
        pass


def test_parse_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("prompt,response\nسوال,جواب\n", encoding="utf-8")
    rows = parse_csv(path)
    assert rows == [{"prompt": "سوال", "response": "جواب"}]


def test_parse_plain_text(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("اردو کی ایک سطر۔\n\nدوسری سطر۔\n", encoding="utf-8")
    rows = parse_plain_text(path)
    assert len(rows) == 2
    assert rows[0]["response"] == "اردو کی ایک سطر۔"


def test_seed_scripts_are_valid_jsonl():
    seed_dir = Path(__file__).parent.parent / "data" / "seed"
    total = 0
    for path in sorted(seed_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                assert "scenes" in row and row["scenes"], path.name
                total += 1
    assert total >= 20
