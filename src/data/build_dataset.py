import argparse
import hashlib
import json
import re
from pathlib import Path

from filter_urdu import normalize_and_check


PUNCT_RE = re.compile(r"[\u0600-\u06FF\w]+")
ARABIC_PUNCT_RE = re.compile(r"[\u06D4\u060C\u061B\u061F\u066A-\u066D\u204F\u2E41]")


def text_fingerprint(text: str) -> str:
    text = ARABIC_PUNCT_RE.sub(" ", text)
    tokens = PUNCT_RE.findall(text)
    return hashlib.md5(" ".join(sorted(tokens)).encode("utf-8")).hexdigest()


def exact_fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def jaccard_shingle_sim(a: str, b: str) -> float:
    sa = set(PUNCT_RE.findall(a))
    sb = set(PUNCT_RE.findall(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def stable_hash(key: str, total: int) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % total


def split_by_hash(key: str, train: float = 0.7, val: float = 0.15) -> str:
    roll = stable_hash(key, 10000) / 10000.0
    if roll < train:
        return "train"
    if roll < train + val:
        return "val"
    return "test"


def load_seed_episodes(seed_dir: Path):
    episodes = []
    for path in sorted(seed_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                episodes.append(json.loads(line))
    return episodes


def scene_to_instruction(episode: dict, scene: dict) -> dict:
    characters = "، ".join(episode.get("characters", []))
    prompt = (
        f"ایک {episode['genre'].replace('_', ' ')} ڈرامے کا منظر لکھیں۔ "
        f"عنوان: {episode['title']}۔ کردار: {characters}۔ "
        f"منظر نمبر {scene['scene_number']}۔"
    )
    parts = []
    if scene.get("setting"):
        parts.append(f"منظر: {scene['setting']}")
    for line in scene.get("lines", []):
        speaker = line.get("speaker", "شخص")
        action = line.get("action")
        text = line.get("text", "")
        if action:
            parts.append(f"{speaker} ({action}): {text}")
        else:
            parts.append(f"{speaker}: {text}")
    return {"prompt": prompt, "response": "\n".join(parts)}


def episode_to_continuation(episode: dict) -> dict:
    scenes = episode.get("scenes", [])
    if len(scenes) < 2:
        return None
    first = scene_to_instruction(episode, scenes[0])
    second = scene_to_instruction(episode, scenes[1])
    prompt = (
        f"ڈرامہ: {episode['title']}۔ منظر نمبر 1 نیچے دیا گیا ہے۔ "
        f"کردار: {'، '.join(episode.get('characters', []))}۔ "
        "منظر نمبر 2 لکھیں، پچھلے منظر کی کہانی، کرداروں کی آواز اور جذباتی انداز برقرار رکھتے ہوئے۔\n\n"
        f"منظر 1:\n{first['response']}"
    )
    return {"prompt": prompt, "response": second["response"]}


def episode_to_chat(episode: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": "آپ ایک مہارت مند اردو ڈرامہ نگار ہیں۔ آپ فطری، رواں اردو مکالمہ لکھتے ہیں، کرداروں کی آواز اور کہانی کے تسلسل کو برقرار رکھتے ہیں۔",
        }
    ]
    for scene in episode.get("scenes", []):
        pair = scene_to_instruction(episode, scene)
        messages.append({"role": "user", "content": pair["prompt"]})
        messages.append({"role": "assistant", "content": pair["response"]})
    return {"messages": messages}


def process_seed_episodes(episodes: list):
    records = {"train": [], "val": [], "test": []}
    seen_fingerprints = set()
    stats = {"episodes": len(episodes), "scenes": 0, "duplicates_dropped": 0, "filtered_dropped": 0}

    for episode in episodes:
        ep_split = split_by_hash(episode["episode_id"])
        scene_pairs = []
        for scene in episode.get("scenes", []):
            stats["scenes"] += 1
            pair = scene_to_instruction(episode, scene)
            text, result = normalize_and_check(pair["response"])
            if not result["ok"]:
                stats["filtered_dropped"] += 1
                continue
            fp = exact_fingerprint(text)
            if fp in seen_fingerprints:
                stats["duplicates_dropped"] += 1
                continue
            seen_fingerprints.add(fp)
            scene_pairs.append({"prompt": pair["prompt"], "response": text, "source": "seed", "episode_id": episode["episode_id"], "genre": episode["genre"]})

        for pair in scene_pairs:
            records[ep_split].append(pair)

        continuation = episode_to_continuation(episode)
        if continuation and ep_split == "test":
            records["test"].append({"prompt": continuation["prompt"], "response": continuation["response"], "source": "seed_continuation", "episode_id": episode["episode_id"], "genre": episode["genre"]})

        chat = episode_to_chat(episode)
        records[ep_split].append({"messages": chat["messages"], "source": "seed_chat", "episode_id": episode["episode_id"], "genre": episode["genre"]})

    return records, stats


def route_row(row: dict, records: dict, seen: set, stats: dict) -> str:
    response = row.get("response", "")
    text, result = normalize_and_check(response)
    if not result["ok"] or not text:
        stats["raw_filtered_dropped"] += 1
        return "filtered"
    fp = text_fingerprint(text)
    if fp in seen:
        stats["raw_duplicates_dropped"] += 1
        return "duplicate"
    seen.add(fp)
    key = f"{row.get('source', 'unknown')}:{row.get('episode_id', '')}:{fp[:12]}"
    split = split_by_hash(key)
    records[split].append({"prompt": row.get("prompt", ""), "response": text, "source": row.get("source", "unknown"), "episode_id": row.get("episode_id", ""), "genre": row.get("genre", "generic")})
    return "kept"


def process_raw_source_rows(raw_dir: Path):
    records = {"train": [], "val": [], "test": []}
    seen = set()
    stats = {"raw_rows": 0, "raw_kept": 0, "raw_duplicates_dropped": 0, "raw_filtered_dropped": 0}

    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.name == "pseudo_scripts.jsonl":
            with path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["source"] = "pseudo_script"
                    row["episode_id"] = row.get("source_id", "")
                    row["genre"] = "drama_dialogue"
                    if route_row(row, records, seen, stats) == "kept":
                        stats["raw_kept"] += 1
            continue

        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                stats["raw_rows"] += 1
                fields = raw.get("fields", {})
                source = raw.get("source", "unknown")
                source_id = raw.get("source_id", "")
                candidates = []
                if "ur" in fields and "en" in fields:
                    candidates.append({"prompt": f"اس انگریزی جملے کا اردو میں ترجمہ کریں: '{fields['en']}'", "response": fields["ur"]})
                else:
                    for key in ("text", "sentence", "sher", "content"):
                        if key in fields and isinstance(fields[key], str) and fields[key].strip():
                            candidates.append({"prompt": "اردو میں ایک رواں جملہ لکھیں جو اسی انداز کا ہو:", "response": fields[key]})

                for cand in candidates:
                    cand["source"] = source
                    cand["episode_id"] = source_id
                    cand["genre"] = "generic"
                    if route_row(cand, records, seen, stats) == "kept":
                        stats["raw_kept"] += 1

    return records, stats


def split_no_leakage(records: dict) -> dict:
    leaks = []
    train_ids = {r.get("episode_id") for r in records["train"] if r.get("episode_id")}
    test_ids = {r.get("episode_id") for r in records["test"] if r.get("episode_id")}
    val_ids = {r.get("episode_id") for r in records["val"] if r.get("episode_id")}
    seed_test = {i for i in test_ids if not str(i).startswith("pseudo-") and i not in ("", None)}
    seed_train = {i for i in train_ids if not str(i).startswith("pseudo-") and i not in ("", None)}
    seed_val = {i for i in val_ids if not str(i).startswith("pseudo-") and i not in ("", None)}
    overlaps = (seed_test & seed_train) | (seed_test & seed_val) | (seed_train & seed_val)
    if overlaps:
        leaks.append({"type": "seed_episode_leakage", "episodes": sorted(overlaps)})
    return {"ok": not leaks, "leaks": leaks}


def main():
    parser = argparse.ArgumentParser(description="Build final Urdu drama dataset: filters, dedup, leakage-free splits.")
    parser.add_argument("--seed_dir", type=str, default="data/seed")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/final")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_seed_episodes(seed_dir)
    seed_records, seed_stats = process_seed_episodes(episodes)

    raw_records = {"train": [], "val": [], "test": []}
    raw_stats = {"raw_rows": 0, "raw_kept": 0, "raw_duplicates_dropped": 0, "raw_filtered_dropped": 0}
    if raw_dir.exists() and any(raw_dir.glob("*.jsonl")):
        raw_records, raw_stats = process_raw_source_rows(raw_dir)

    merged = {split: seed_records[split] + raw_records[split] for split in ("train", "val", "test")}

    leak_check = split_no_leakage(merged)
    if not leak_check["ok"]:
        print(f"LEAKAGE DETECTED: {leak_check['leaks']}")
        raise SystemExit(1)

    for split in ("train", "val", "test"):
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in merged[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{split}] wrote {len(merged[split])} rows -> {out_path}")

    per_source = {}
    for split in ("train", "val", "test"):
        for row in merged[split]:
            per_source.setdefault(row.get("source", "unknown"), 0)
            per_source[row.get("source", "unknown")] += 1

    report = {
        "seed_stats": seed_stats,
        "raw_stats": raw_stats,
        "split_sizes": {split: len(merged[split]) for split in ("train", "val", "test")},
        "per_source": per_source,
        "leakage_check": leak_check,
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Build report ->", output_dir / "build_report.json")


if __name__ == "__main__":
    main()
