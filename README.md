# Llama 3.2 Urdu Drama

A fine-tuning project that produces an Urdu drama-script writer: a Llama 3.2 model fine-tuned to write natural, idiomatic Urdu drama scripts — not machine-translated, not Hindi-leaning.

<p align="center">
  <img src="https://img.shields.io/badge/Model-Llama_3.2-8A2BE2" alt="Llama 3.2" />
  <img src="https://img.shields.io/badge/Method-QLoRA-00BFA6" alt="QLoRA" />
  <img src="https://img.shields.io/badge/Language-Urdu_Nastaliq-0066CC" alt="Urdu" />
  <img src="https://img.shields.io/badge/Deploy-Ollama-FF6B6B" alt="Ollama" />
</p>

## What the model does

- Writes natural, idiomatic Urdu dialogue (native Urdu script, Nastaliq-compatible Unicode)
- Generates full scripts in proper drama format: scene headings, character dialogue, action lines
- Maintains character voice, emotional tone, and story continuity across scenes
- Handles genres: family drama, romance, social issues, thriller

## Pipeline

```text
data/seed/     ← hand-written formatted drama scripts (the format layer)
data/raw/      ← pulled from Hugging Face (the dialogue/fluency layer)
src/data/ingest_hf.py          ← pulls opus-100 ur-en, makhzan-urdu, UrduShers + provenance
src/data/filter_urdu.py        ← Devanagari rejection, NFKC normalize, Hindi-ism blocklist
src/data/subtitles_to_script.py← subtitle dialogue → pseudo drama scripts
src/data/build_dataset.py      ← dedup, filters, episode-level leakage-free splits
src/train.py                   ← QLoRA fine-tuning (fixed: pad token, seeds, eval split)
src/evaluate.py                ← ROUGE/BLEU/chrF + script structure score + baseline compare
src/stress_test.py             ← holes/leaks/contamination checks (CI gate)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 1) Ingest public Urdu data from Hugging Face (optional, adds scale)
python src/data/ingest_hf.py --output_dir data/raw --max_rows 20000
python src/data/subtitles_to_script.py --input data/raw/opus100_ur_en.jsonl --output data/raw/pseudo_scripts.jsonl

# 2) Build the final dataset (filters + dedup + leakage-free splits)
python src/data/build_dataset.py --seed_dir data/seed --raw_dir data/raw --output_dir data/final

# 3) Stress test before training (exit 1 on any failure)
python src/stress_test.py --data_dir data/final --seed_dir data/seed

# 4) Fine-tune with QLoRA
python src/train.py \
  --train_data data/final/train.jsonl \
  --eval_data data/final/val.jsonl \
  --output_dir models/urdu-drama-3.2-qLoRA \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --epochs 2

# 5) Evaluate fine-tuned vs baseline, same prompts
python src/evaluate.py \
  --model_dir models/urdu-drama-3.2-qLoRA \
  --eval_data data/final/test.jsonl \
  --baseline_model meta-llama/Llama-3.2-3B-Instruct \
  --output reports/eval.json

# 6) Inference
python src/inference.py --model_dir models/urdu-drama-3.2-qLoRA \
  --prompt "ایک family drama ڈرامے کا منظر لکھیں۔ کردار: امجد صاحب، شازیہ۔"

# 7) Unit tests
python -m pytest tests/ -q
```

## Why a hybrid dataset (no public drama-script dataset exists)

| Layer | Source | Teaches |
|---|---|---|
| Format | `data/seed/` — hand-written formatted drama scripts | Scene structure, speaker labels, action lines |
| Dialogue | `Helsinki-NLP/opus-100` (ur-en, subtitles) | Natural, colloquial spoken Urdu |
| Grammar | `ReySajju742/makhzan-urdu` | Formal register, editorial-quality prose |
| Flavour | `keplersystems/UrduShers-10k` | Poetic and emotional register |

Every pulled source is filtered (Devanagari rejection, Hindi-ism screening, script-ratio checks) and every row keeps provenance, so benchmarks can report per-source quality.

## Data format

Seed episodes (`data/seed/*.jsonl`):

```json
{
  "episode_id": "family-01",
  "genre": "family_drama",
  "title": "بڑے گھر کی بیٹھک",
  "characters": ["امجد صاحب", "شازیہ"],
  "character_sheet": {"امجد صاحب": "۶۵ سالہ بزرگ، روایتی مزاج"},
  "scenes": [{"scene_number": 1, "setting": "...", "lines": [{"speaker": "امجد صاحب", "action": "چائے لیتے ہوئے", "text": "..."}]}]
}
```

Training rows (`data/final/*.jsonl`): `{"prompt", "response", "source", "episode_id", "genre"}` and multi-turn `{"messages", ...}` episode conversations.

## Documentation

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — benchmark protocol, required results table, stress test spec
- [docs/EVALUATION.md](docs/EVALUATION.md) — metric definitions, failure modes, human eval protocol

## License

MIT

## Acknowledgements

- Meta Llama, Hugging Face Transformers, PEFT, TRL, Ollama
- Helsinki-NLP opus-100, makhzan-urdu, UrduShers-10k contributors
