# Llama 3.2 Urdu

A practical fine-tuning project for building a high-quality Urdu-capable LLM using Meta's Llama 3.2, QLoRA, and Ollama deployment.

This repository demonstrates how to:
- prepare Urdu instruction data
- fine-tune Llama 3.2 with QLoRA
- evaluate response quality with metrics
- run the model locally with `ollama`
- generate a clean portfolio-ready project structure

## Why this project matters

Most public LLMs are trained primarily for English, and Urdu quality often suffers from:
- limited instruction tuning data
- weak alignment for Urdu-specific tasks
- poor handling of dialectal nuance and local context

This project addresses that by creating a simple, reproducible pipeline for Urdu instruction tuning using a memory-efficient fine-tuning strategy.

## Project goals

- Fine-tune a Llama 3.2 model for Urdu text generation and QA
- Use QLoRA to make memory usage manageable
- Keep the setup understandable for researchers and builders
- Make deployment easy with Ollama

## Repo structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── urdu_demo.jsonl
│   └── README.md
├── src/
│   ├── data/
│   │   └── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── ollama/
│   └── Modelfile
├── scripts/
│   └── setup_ollama.sh
└── models/
    └── .gitkeep
```

## Quick start

1. Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Prepare a dataset:

```bash
python src/data/prepare_dataset.py \
  --input data/urdu_demo.jsonl \
  --output data/train.jsonl
```

3. Fine-tune using QLoRA:

```bash
python src/train.py \
  --train_data data/train.jsonl \
  --output_dir models/urdu-llama-3.2-qLoRA \
  --model_name meta-llama/Llama-3.2-3B-Instruct
```

4. Evaluate:

```bash
python src/evaluate.py \
  --model_dir models/urdu-llama-3.2-qLoRA \
  --eval_data data/urdu_demo.jsonl
```

5. Run inference:

```bash
python src/inference.py \
  --model_dir models/urdu-llama-3.2-qLoRA \
  --prompt "اردو میں ایک مختصر جواب لکھیں کہ موسم کیسی ہے؟"
```

6. Deploy with Ollama:

```bash
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
```

## Dataset requirements

The training code expects prompt-response pairs in JSONL format:

```json
{"prompt": "اردو میں ترجمہ کریں: 'The weather is nice today.'", "response": "آج موسم اچھا ہے۔"}
```

You can also use CSV or a plain text file, then convert it with `prepare_dataset.py`.

## Training details

This project uses:
- Llama 3.2 instruct model as the base model
- QLoRA for PEFT-based fine-tuning
- `bitsandbytes` for 4-bit quantization
- `trl.SFTTrainer` for efficient fine-tuning
- memory-conscious settings suitable for local GPU workloads

## Metrics included

The evaluation flow tracks:
- training loss
- validation loss
- perplexity-like practical quality tracking
- Rouge-L for instruction-response quality
- sample generation outputs for qualitative review

## Ollama deployment

The project includes a Modelfile that creates a local Ollama model wrapper. After you have a trained model or a converted exported checkpoint, run:

```bash
ollama create urdu-llama -f ollama/Modelfile
ollama run urdu-llama
```

## Notes about model access

Llama 3.2 may require access to the Hugging Face model hub, depending on your account and license status. You may need to authenticate:

```bash
huggingface-cli login
```

## Recommended next steps

- add more Urdu instruction tuning data
- benchmark against a baseline model
- expand to conversation-style examples
- deploy a small web UI or API around the model
- publish a more polished version of this repo for GitHub portfolio use

## License

MIT

## Acknowledgements

- Meta Llama
- Hugging Face Transformers
- PEFT
- TRL
- Ollama

This project is designed to be educational, reproducible, and suitable as a real-world portfolio repository.
