import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "ایک family drama ڈرامے کا منظر لکھیں۔ عنوان: بڑے گھر کی بیٹھک۔ کردار: امجد صاحب، شازیہ۔ منظر نمبر 1۔",
    "ایک thriller ڈرامے کا منظر لکھیں۔ عنوان: اندھیری رات۔ کردار: افسر کمال، سیمی۔ منظر نمبر 1۔",
    "ایک romance ڈرامے کا منظر لکھیں۔ کردار: عرفان، مہرین۔ منظر نمبر 1۔",
]


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_messages(item: dict):
    # schema unification: absent fields arrive as None - check truthiness
    if item.get("messages"):
        return item["messages"]
    return [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response"]}]


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int = 300):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def compute_bleu(predictions, references):
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {"bleu": round(bleu.score, 4), "chrf": round(chrf.score, 4)}


def compute_rouge(predictions, references):
    from evaluate import load as load_metric

    metric = load_metric("rouge")
    result = metric.compute(predictions=predictions, references=references)
    return {key: round(float(value), 4) for key, value in result.items()}


def script_structure_score(outputs):
    import re

    scores = []
    for text in outputs:
        has_scene = bool(re.search(r"منظر\s*\d?", text))
        has_speaker = bool(re.search(r"[\u0600-\u06FF][\u0600-\u06FF\s]*:", text))
        has_action = bool(re.search(r"\([^)]+\)", text))
        scores.append(((1 if has_scene else 0) + (1 if has_speaker else 0) + (1 if has_action else 0)) / 3)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def evaluate_model(model_name: str, eval_data: str, max_new_tokens: int, limit: int = 50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False)
    model.eval()

    rows = load_rows(eval_data)[:limit]
    preds = []
    refs = []
    prompts = []
    for row in rows:
        messages = build_messages(row)
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), row.get("prompt", ""))
        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), row.get("response", ""))
        prompts.append(user_msg)
        prediction = run_generation(model, tokenizer, user_msg, max_new_tokens=max_new_tokens)
        preds.append(prediction)
        refs.append(asst_msg)

    results = {"model": model_name, "num_examples": len(preds)}
    results.update(compute_rouge(preds, refs))
    results.update(compute_bleu(preds, refs))
    results["script_structure_score"] = script_structure_score(preds)
    results["avg_prediction_length"] = round(sum(len(p) for p in preds) / len(preds), 2) if preds else 0
    results["samples"] = [
        {"prompt": p[:200], "prediction": pred[:400], "reference": ref[:400]}
        for p, pred, ref in zip(prompts[:5], preds[:5], refs[:5])
    ]

    del model
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Urdu drama model with ROUGE/BLEU/chrF and structure scoring.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None, help="Held-out test JSONL.")
    parser.add_argument("--baseline_model", type=str, default=None, help="Base model for comparison.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="reports/eval.json")
    args = parser.parse_args()

    report = {}
    if args.eval_data:
        report["fine_tuned"] = evaluate_model(args.model_dir, args.eval_data, args.max_new_tokens, args.limit)
        if args.baseline_model:
            report["baseline"] = evaluate_model(args.baseline_model, args.eval_data, args.max_new_tokens, args.limit)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", trust_remote_code=False)
        model.eval()
        samples = []
        for prompt in DEFAULT_PROMPTS:
            answer = run_generation(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            samples.append({"prompt": prompt, "response": answer})
        report["samples"] = samples

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2, ensure_ascii=False))
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
