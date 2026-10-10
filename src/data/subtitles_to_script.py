import argparse
import json
import random
from pathlib import Path


SPEAKERS = ["شخص اول", "شخص دوم", "شخص سوم"]
SCENE_BREAK_MARKERS = ["۔", "؟", "！"]
MIN_TURNS_PER_SCENE = 4
MAX_TURNS_PER_SCENE = 10


def segment_dialogue(lines: list, rng: random.Random) -> list:
    scenes = []
    current = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        current.append(line)
        if len(current) >= rng.randint(MIN_TURNS_PER_SCENE, MAX_TURNS_PER_SCENE):
            scenes.append(current)
            current = []
    if len(current) >= MIN_TURNS_PER_SCENE:
        scenes.append(current)
    return scenes


def format_scene(lines: list, scene_number: int, speaker_names: list) -> str:
    parts = [f"منظر {scene_number}:"]
    for i, line in enumerate(lines):
        speaker = speaker_names[i % len(speaker_names)]
        parts.append(f"{speaker}: {line}")
    return "\n".join(parts)


def convert_lines(lines: list, seed: int = 42) -> list:
    rng = random.Random(seed)
    scenes = segment_dialogue(lines, rng)
    speaker_names = SPEAKERS[: min(3, max(2, len(SPEAKERS)))]
    return [
        {
            "scene_number": idx + 1,
            "text": format_scene(scene, idx + 1, speaker_names),
        }
        for idx, scene in enumerate(scenes)
    ]


def convert_opus_to_pseudo_scripts(input_path: Path, output_path: Path, max_scripts: int, context_turns: int = 2, seed: int = 42):
    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)

    buffers = []
    scripts = []
    for row in rows:
        ur = row.get("fields", {}).get("ur", "").strip()
        if not ur:
            continue
        buffers.append(ur)
        if len(buffers) == context_turns:
            scripts.append(list(buffers))
            buffers = []
        if len(scripts) >= max_scripts:
            break

    converted = []
    for idx, dialogue_lines in enumerate(scripts):
        scenes = convert_lines(dialogue_lines, seed=seed + idx)
        if not scenes:
            continue
        script_text = "\n\n".join(s["text"] for s in scenes)
        converted.append(
            {
                "source": "opus100_ur_en",
                "source_id": f"pseudo-{idx}",
                "prompt": "مندرجہ ذیل مکالمے کو ڈرامہ اسکرپٹ کی شکل میں لکھیں (منظر اور کردار کے نام کے ساتھ):",
                "response": script_text,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} pseudo-scripts -> {output_path}")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert Urdu subtitle dialogue into pseudo drama scripts.")
    parser.add_argument("--input", type=str, default="data/raw/opus100_ur_en.jsonl")
    parser.add_argument("--output", type=str, default="data/raw/pseudo_scripts.jsonl")
    parser.add_argument("--max_scripts", type=int, default=5000)
    parser.add_argument("--context_turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_opus_to_pseudo_scripts(Path(args.input), Path(args.output), args.max_scripts, args.context_turns, args.seed)


if __name__ == "__main__":
    main()
