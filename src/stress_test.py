import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data"))

from filter_urdu import check_text


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_text(item: dict) -> str:
    if "messages" in item:
        return " ".join(m.get("content", "") for m in item["messages"])
    return item.get("response", "") + " " + item.get("prompt", "")


def check_split_leakage(data_dir: Path):
    train = load_rows(data_dir / "train.jsonl")
    test = load_rows(data_dir / "test.jsonl")
    val = load_rows(data_dir / "val.jsonl")

    train_eps = {r.get("episode_id") for r in train if r.get("episode_id") and not str(r.get("episode_id")).startswith("pseudo-")}
    test_eps = {r.get("episode_id") for r in test if r.get("episode_id") and not str(r.get("episode_id")).startswith("pseudo-")}
    val_eps = {r.get("episode_id") for r in val if r.get("episode_id") and not str(r.get("episode_id")).startswith("pseudo-")}

    issues = []
    overlap_tt = train_eps & test_eps
    overlap_tv = train_eps & val_eps
    if overlap_tt:
        issues.append(f"train/test episode overlap: {sorted(overlap_tt)}")
    if overlap_tv:
        issues.append(f"train/val episode overlap: {sorted(overlap_tv)}")
    return issues


def check_contamination(data_dir: Path):
    issues = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_rows(path)
        for i, row in enumerate(rows):
            text = get_text(row)
            result = check_text(text)
            if not result["ok"]:
                issues.append(f"{split}.jsonl row {i}: {result['reasons']}")
    return issues


def check_duplicates(data_dir: Path):
    from build_dataset import text_fingerprint

    issues = []
    seen = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_rows(path)
        for i, row in enumerate(rows):
            fp = text_fingerprint(get_text(row))
            if fp in seen:
                issues.append(f"duplicate: {split}.jsonl row {i} == {seen[fp]}")
            else:
                seen[fp] = f"{split}.jsonl row {i}"
    return issues


def check_length_outliers(data_dir: Path, min_len: int = 20, max_len: int = 20000):
    issues = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_rows(path)
        for i, row in enumerate(rows):
            text = get_text(row)
            if len(text) < min_len:
                issues.append(f"{split}.jsonl row {i}: too short ({len(text)} chars)")
            if len(text) > max_len:
                issues.append(f"{split}.jsonl row {i}: too long ({len(text)} chars)")
    return issues


def check_prompt_repetition(data_dir: Path, max_repeats: int = 10):
    issues = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_rows(path)
        prompt_counts = {}
        for row in rows:
            if "messages" in row:
                user_msgs = [m.get("content", "") for m in row["messages"] if m.get("role") == "user"]
                prompt = (user_msgs[0] if user_msgs else "")[:100]
            else:
                prompt = row.get("prompt", "")[:100]
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        for prompt, count in prompt_counts.items():
            if count > max_repeats:
                issues.append(f"{split}.jsonl: template repeated {count}x: '{prompt}'")
    return issues


def check_seed_scripts(seed_dir: Path):
    issues = []
    required_fields = ["episode_id", "genre", "title", "characters", "character_sheet", "scenes"]
    genres = set()
    for path in sorted(seed_dir.glob("*.jsonl")):
        rows = load_rows(path)
        if not rows:
            issues.append(f"{path.name}: empty seed file")
        for i, row in enumerate(rows):
            for field in required_fields:
                if field not in row or not row[field]:
                    issues.append(f"{path.name} row {i}: missing field '{field}'")
            for scene in row.get("scenes", []):
                for line in scene.get("lines", []):
                    text = check_text(line.get("text", ""))
                    if not text["ok"]:
                        issues.append(f"{path.name} row {i}: {text['reasons']}")
            genres.add(row.get("genre", ""))
    return issues, genres


def main():
    parser = argparse.ArgumentParser(description="Stress test the Urdu drama dataset and pipeline for holes and leaks.")
    parser.add_argument("--data_dir", type=str, default="data/final")
    parser.add_argument("--seed_dir", type=str, default="data/seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seed_dir = Path(args.seed_dir)

    all_issues = []
    failures = 0

    if data_dir.exists():
        for name, fn in [
            ("SPLIT LEAKAGE", check_split_leakage),
            ("CONTAMINATION", check_contamination),
            ("DUPLICATES", check_duplicates),
            ("LENGTH OUTLIERS", check_length_outliers),
            ("TEMPLATE REPETITION", check_prompt_repetition),
        ]:
            issues = fn(data_dir)
            status = "PASS" if not issues else f"FAIL ({len(issues)})"
            print(f"[{status}] {name}")
            for issue in issues[:10]:
                print(f"    {issue}")
            if issues:
                failures += 1
                all_issues.extend(issues)
    else:
        print(f"[SKIP] {data_dir} does not exist (run build_dataset.py first)")

    seed_issues, genres = check_seed_scripts(seed_dir)
    status = "PASS" if not seed_issues else f"FAIL ({len(seed_issues)})"
    print(f"[{status}] SEED SCRIPTS")
    for issue in seed_issues[:10]:
        print(f"    {issue}")
    print(f"    genres: {sorted(genres)}")
    if seed_issues:
        failures += 1
        all_issues.extend(seed_issues)

    print(f"\n{'=' * 50}")
    if failures == 0:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"{failures} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
