import argparse
import json
from pathlib import Path

from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_PROMPTS = [
    "اردو میں ایک مختصر اور دوستانہ جواب لکھیں۔",
    "اردو میں ایک رسمی ای میل لکھیں۔",
    "اردو میں ایک مختصر تعریف لکھیں کہ جدید ٹیکنالوجی کیا ہے۔",
]


def build_prompt_response_pairs(data_path: str):
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({"prompt": item["prompt"], "response": item["response"]})
    return rows


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def compute_rouge(predictions, references):
    metric = load_metric("rouge")
    result = metric.compute(predictions=predictions, references=references)
    return {key: round(float(value), 4) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Urdu Llama model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory with adapter or merged weights.")
    parser.add_argument("--eval_data", type=str, default=None, help="JSONL file with prompt/response pairs for validation.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate per response.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    if args.eval_data:
        rows = build_prompt_response_pairs(args.eval_data)
        preds = []
        refs = []
        for row in rows[:20]:
            prediction = run_generation(model, tokenizer, row["prompt"], max_new_tokens=args.max_new_tokens)
            preds.append(prediction)
            refs.append(row["response"])
        metrics = compute_rouge(preds, refs)
        print("Evaluation metrics:")
        for key, val in metrics.items():
            print(f"{key}: {val}")

    print("\nSample outputs:")
    for prompt in DEFAULT_PROMPTS:
        answer = run_generation(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        print(f"Prompt: {prompt}")
        print(f"Response: {answer}\n")


if __name__ == "__main__":
    main()
