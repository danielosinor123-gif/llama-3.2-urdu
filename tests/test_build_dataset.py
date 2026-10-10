import json

from build_dataset import (
    episode_to_continuation,
    episode_to_chat,
    jaccard_shingle_sim,
    scene_to_instruction,
    split_by_hash,
    text_fingerprint,
)


def make_episode(episode_id="family-99", genre="family_drama"):
    return {
        "episode_id": episode_id,
        "genre": genre,
        "title": "ٹیسٹ ڈرامہ",
        "characters": ["امی", "ابو"],
        "character_sheet": {"امی": "ماں", "ابو": "باپ"},
        "scenes": [
            {
                "scene_number": 1,
                "setting": "صحن میں چائے کی ٹری۔",
                "lines": [
                    {"speaker": "ابو", "action": "چائے لیتے ہوئے", "text": "کھانا کھا لیا؟"},
                    {"speaker": "امی", "action": None, "text": "جی، آپ بھی کھا لیں۔"},
                ],
            },
            {
                "scene_number": 2,
                "setting": "رات کا وقت۔",
                "lines": [
                    {"speaker": "ابو", "action": None, "text": "چلیں سو جائیں۔"},
                    {"speaker": "امی", "action": None, "text": "جی، اللہ حافظ۔"},
                ],
            },
        ],
    }


def test_split_by_hash_is_deterministic():
    assert split_by_hash("family-01") == split_by_hash("family-01")
    assert split_by_hash("family-01") in {"train", "val", "test"}


def test_split_by_hash_covers_all_buckets():
    splits = {split_by_hash(f"ep-{i}") for i in range(200)}
    assert {"train", "val", "test"}.issubset(splits)


def test_scene_to_instruction_format():
    pair = scene_to_instruction(make_episode(), make_episode()["scenes"][0])
    assert "منظر" in pair["prompt"]
    assert "ابو (" in pair["response"]
    assert "امی:" in pair["response"]


def test_episode_to_continuation():
    cont = episode_to_continuation(make_episode())
    assert cont is not None
    assert "منظر 1:" in cont["prompt"] or "منظر نمبر 1" in cont["prompt"]
    assert "منظر" in cont["response"]


def test_episode_to_chat_has_system_and_turns():
    chat = episode_to_chat(make_episode())
    messages = chat["messages"]
    assert messages[0]["role"] == "system"
    roles = [m["role"] for m in messages[1:]]
    assert roles[0] == "user" and roles[1] == "assistant"
    assert roles[-1] == "assistant"


def test_fingerprint_dedup():
    a = "یہ ایک جملہ ہے۔"
    b = "جملہ ایک یہ ہے"
    assert text_fingerprint(a) == text_fingerprint(b)
    c = "بالکل مختلف متن ہے۔"
    assert text_fingerprint(a) != text_fingerprint(c)


def test_jaccard_similarity():
    assert jaccard_shingle_sim("یہ جملہ ہے", "یہ جملہ ہے") == 1.0
    assert jaccard_shingle_sim("یہ جملہ ہے", "کوئی اور متن") < 0.5


def test_jsonl_round_trip(tmp_path):
    episode = make_episode()
    path = tmp_path / "ep.jsonl"
    path.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == episode
