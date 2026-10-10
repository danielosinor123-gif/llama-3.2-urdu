import argparse
import json
from pathlib import Path

from datasets import load_dataset


SOURCES = {
    "opus100_ur_en": {
        "loader": lambda max_rows: load_dataset("Helsinki-NLP/opus-100", "ur-en", split=f"train[:{max_rows}]"),
        "extract": lambda row: {"ur": row["translation"]["ur"], "en": row["translation"]["en"]},
    },
    "makhzan_urdu": {
        "loader": lambda max_rows: load_dataset("ReySajju742/makhzan-urdu", split=f"train[:{max_rows}]"),
        "extract": None,
    },
    "urdu_shers": {
        "loader": lambda max_rows: load_dataset("keplersystems/UrduShers-10k", split=f"train[:{max_rows}]"),
        "extract": None,
    },
}


def resolve_text_fields(sample: dict) -> list:
    texts = []
    for value in sample.values():
        if isinstance(value, str) and value.strip():
            texts.append(value)
        elif isinstance(value, dict):
            texts.extend(v for v in value.values() if isinstance(v, str) and v.strip())
    return texts


def ingest_source(name: str, spec: dict, max_rows: int, output_dir: Path):
    ds = spec["loader"](max_rows)
    out_path = output_dir / f"{name}.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if spec["extract"] is not None:
                try:
                    record = spec["extract"](row)
                except (KeyError, TypeError):
                    continue
                payload = {"source": name, "source_id": str(i), "fields": record}
            else:
                texts = resolve_text_fields(row)
                if not texts:
                    continue
                payload = {"source": name, "source_id": str(i), "fields": {"text": " ".join(texts)}}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    print(f"[{name}] wrote {count} rows -> {out_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest public Urdu datasets from Hugging Face with provenance.")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--max_rows", type=int, default=20000, help="Max rows per source.")
    parser.add_argument("--sources", type=str, nargs="*", default=list(SOURCES.keys()))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for name in args.sources:
        if name not in SOURCES:
            print(f"Unknown source: {name}, skipping")
            continue
        try:
            total += ingest_source(name, SOURCES[name], args.max_rows, output_dir)
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}")

    manifest = {
        "sources": args.sources,
        "max_rows_per_source": args.max_rows,
        "total_rows": total,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Total ingested rows: {total}")


if __name__ == "__main__":
    main()
