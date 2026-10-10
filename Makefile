PYTHON ?= python
VENV ?= .venv

install:
	@$(PYTHON) -m venv $(VENV)
	@. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

prepare:
	@. $(VENV)/bin/activate && python src/data/prepare_dataset.py --input data/urdu_demo.jsonl --output data/train.jsonl

ingest:
	@. $(VENV)/bin/activate && python src/data/ingest_hf.py --output_dir data/raw --max_rows 20000

build:
	@. $(VENV)/bin/activate && python src/data/build_dataset.py --seed_dir data/seed --raw_dir data/raw --output_dir data/final

train:
	@. $(VENV)/bin/activate && python src/train.py --train_data data/final/train.jsonl --eval_data data/final/val.jsonl --output_dir models/urdu-drama-3.2-qLoRA --model_name meta-llama/Llama-3.2-3B-Instruct --epochs 2

eval:
	@. $(VENV)/bin/activate && python src/evaluate.py --model_dir models/urdu-drama-3.2-qLoRA --eval_data data/final/test.jsonl --output reports/eval.json

baseline:
	@. $(VENV)/bin/activate && python src/evaluate.py --model_dir meta-llama/Llama-3.2-3B-Instruct --eval_data data/final/test.jsonl --output reports/baseline.json

stress:
	@. $(VENV)/bin/activate && python src/stress_test.py --data_dir data/final --seed_dir data/seed

test:
	@. $(VENV)/bin/activate && python -m pytest tests/ -q

chat:
	@. $(VENV)/bin/activate && python src/inference.py --model_dir models/urdu-drama-3.2-qLoRA

ollama:
	@chmod +x scripts/setup_ollama.sh && ./scripts/setup_ollama.sh

clean:
	@rm -rf models/* reports/* .cache
