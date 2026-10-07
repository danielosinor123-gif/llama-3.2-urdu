# Llama 3.2 Urdu

<p align="center">
  <img src="https://img.shields.io/badge/Model-Llama_3.2-8A2BE2" alt="Llama 3.2" />
  <img src="https://img.shields.io/badge/Method-QLoRA-00BFA6" alt="QLoRA" />
  <img src="https://img.shields.io/badge/Language-Urdu-0066CC" alt="Urdu" />
  <img src="https://img.shields.io/badge/Deploy-Ollama-FF6B6B" alt="Ollama" />
</p>

A portfolio-ready project for fine-tuning a Llama 3.2 model for Urdu instruction-following, translation, and conversational generation. The repository uses QLoRA for memory-efficient training and includes a clean evaluation and deployment workflow for local inference with Ollama.

## Why this project

Many open-weight models perform well in English but are weaker in Urdu due to limited instruction tuning and smaller Urdu-specific data coverage. This project demonstrates a practical workflow for:

- preparing Urdu prompt-response datasets
- fine-tuning Llama 3.2 with QLoRA
- measuring quality with validation metrics and sample generation
- deploying the result locally through Ollama

## What is included

- Urdu dataset preparation pipeline
- QLoRA fine-tuning script for Llama 3.2
- evaluation script with sample outputs
- local inference CLI
- Ollama deployment config
- clear repo structure for a polished GitHub project

## Repository layout

```text
.
├── README.md
├── Makefile
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── urdu_demo.jsonl
│   └── train.jsonl   # generated after dataset prep
├── src/
│   ├── data/
│   │   └── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── __init__.py
├── ollama/
│   └── Modelfile
├── scripts/
│   └── setup_ollama.sh
├── models/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```

## Quick start

### 1) Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Prepare the dataset

```bash
python src/data/prepare_dataset.py \
  --input data/urdu_demo.jsonl \
  --output data/train.jsonl
```

### 3) Fine-tune with QLoRA

```bash
python src/train.py \
  --train_data data/train.jsonl \
  --output_dir models/urdu-llama-3.2-qLoRA \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 2e-4
```

### 4) Evaluate the model

```bash
python src/evaluate.py \
  --model_dir models/urdu-llama-3.2-qLoRA \
  --eval_data data/urdu_demo.jsonl
```

### 5) Run inference interactively

```bash
python src/inference.py \
  --model_dir models/urdu-llama-3.2-qLoRA
```

### 6) Deploy locally with Ollama

```bash
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
ollama run urdu-llama
```

## Data format

The training pipeline expects prompt-response pairs in JSONL format.

```json
{"prompt": "اردو میں ترجمہ کریں: 'The weather is nice today.'", "response": "آج موسم اچھا ہے۔"}
```

You can also supply CSV or plain text and convert it with the helper script in `src/data/prepare_dataset.py`.

## Training stack

This project is built around:

- Llama 3.2 instruct model
- QLoRA for efficient parameter-efficient fine-tuning
- `bitsandbytes` for 4-bit quantization
- Hugging Face `transformers`, `peft`, and `trl`
- custom evaluation and inference utilities

## Metrics and evaluation

The evaluation flow focuses on:

- training and validation loss
- sample generation quality
- qualitative review of Urdu outputs
- Rouge-like scoring for generated text comparison

This gives a practical benchmark for whether the model is becoming more useful for Urdu tasks.

## Ollama deployment

The project includes an Ollama `Modelfile` to run the model locally with a simple assistant prompt style.

```bash
ollama create urdu-llama -f ollama/Modelfile
ollama run urdu-llama
```

## Model access note

If you are using a gated model like Llama 3.2, make sure your Hugging Face access is configured:

```bash
huggingface-cli login
```

## Recommended workflow

For a serious implementation, the best workflow is:

1. collect a larger Urdu instruction dataset
2. clean and filter low-quality examples
3. train using QLoRA
4. test on validation prompts
5. evaluate with human judgement and automated metrics
6. export and deploy through Ollama or a local API

## Makefile shortcuts

```bash
make install
make data
make train
make eval
make chat
make ollama
```

## Roadmap

- add a larger Urdu corpus dataset pipeline
- improve validation metrics and reporting
- add a small Streamlit or FastAPI demo UI
- benchmark against a baseline model
- publish a cleaner portfolio-ready version with screenshots and results

## License

MIT

## Acknowledgements

- Meta Llama
- Hugging Face Transformers
- PEFT
- TRL
- Ollama

This project is designed as an educational, reproducible, and portfolio-friendly implementation of Urdu fine-tuning with Llama 3.2.
