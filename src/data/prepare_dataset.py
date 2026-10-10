import argparse
import csv
import json
from pathlib import Path


def parse_plain_text(path: Path):
    lines = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if text:
            lines.append({
                "prompt": "اردو میں ایک مختصر جواب لکھیں۔",
                "response": text,
            })
    return lines


def parse_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question")
            response = obj.get("response") or obj.get("answer") or obj.get("completion")
            if prompt is None or response is None:
                raise ValueError(f"Invalid JSONL row: {line}")
            rows.append({"prompt": str(prompt), "response": str(response)})
    return rows


def parse_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt") or row.get("instruction") or row.get("question")
            response = row.get("response") or row.get("answer") or row.get("completion")
            if prompt is None or response is None:
                raise ValueError(f"CSV row missing prompt/response fields: {row}")
            rows.append({"prompt": str(prompt), "response": str(response)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare Urdu instruction data for Llama fine-tuning.")
    parser.add_argument("--input", type=str, required=True, help="Input file: JSONL, CSV, or plain text.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path for prompt-response pairs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".jsonl":
        rows = parse_jsonl(input_path)
    elif input_path.suffix.lower() == ".csv":
        rows = parse_csv(input_path)
    elif input_path.suffix.lower() in {".txt", ".md"}:
        rows = parse_plain_text(input_path)
    else:
        raise ValueError("Unsupported input format. Use JSONL, CSV, or plain text.")

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"Prepared {len(rows)} prompt-response pairs and wrote them to {output_path}")


if __name__ == "__main__":
    main()
