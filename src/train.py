import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_text(item: dict, tokenizer) -> dict:
    if "messages" in item:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
    else:
        messages = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def set_pad_token(tokenizer):
    if tokenizer.pad_token is not None:
        return
    for cand in ["<|finetune_right_pad_id|>", "<|reserved_special_token_0|>"]:
        if cand in tokenizer.get_vocab():
            tokenizer.pad_token = cand
            return
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def build_sft_trainer(model, tokenizer, train_dataset, eval_dataset, args_ns):
    common = dict(
        output_dir=args_ns.output_dir,
        per_device_train_batch_size=args_ns.batch_size,
        gradient_accumulation_steps=args_ns.gradient_accumulation_steps,
        learning_rate=args_ns.learning_rate,
        fp16=False,
        bf16=True,
        num_train_epochs=args_ns.epochs,
        logging_steps=10,
        save_steps=100,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="none",
        seed=args_ns.seed,
        save_total_limit=2,
    )
    if eval_dataset is not None:
        common["eval_strategy"] = "steps"
        common["eval_steps"] = 50
        common["per_device_eval_batch_size"] = args_ns.batch_size

    from trl import SFTTrainer
    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    if SFTConfig is not None:
        sft_kwargs = dict(
            max_length=args_ns.max_seq_length,
            packing=False,
            dataset_text_field="text",
        )
        trainer_kwargs = {}
        try:
            cfg = SFTConfig(**common, **sft_kwargs)
            trainer_kwargs = {"args": cfg, "processing_class": tokenizer}
        except TypeError as exc:
            if "eval_strategy" in str(exc):
                common.pop("eval_strategy", None)
                common["evaluation_strategy"] = "steps" if eval_dataset is not None else "no"
                cfg = SFTConfig(**common, **sft_kwargs)
                trainer_kwargs = {"args": cfg, "processing_class": tokenizer}
            else:
                raise
        trainer = SFTTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, **trainer_kwargs)
        return trainer

    from transformers import TrainingArguments
    legacy = dict(common)
    legacy.pop("eval_strategy", None)
    legacy["evaluation_strategy"] = "steps" if eval_dataset is not None else "no"
    try:
        targs = TrainingArguments(**legacy)
    except TypeError:
        legacy.pop("evaluation_strategy", None)
        legacy["eval_strategy"] = "steps" if eval_dataset is not None else "no"
        targs = TrainingArguments(**legacy)
    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=args_ns.max_seq_length,
        args=targs,
        packing=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 for Urdu drama scripts using QLoRA.")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    train_rows = load_rows(args.train_data)
    if len(train_rows) < 2:
        raise ValueError("Training dataset is too small. Add more examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    set_pad_token(tokenizer)

    train_dataset = load_dataset("json", data_files={"train": args.train_data}, split="train")
    train_dataset = train_dataset.map(lambda x: to_text(x, tokenizer), remove_columns=[c for c in train_dataset.column_names if c != "text"])

    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_dataset("json", data_files={"eval": args.eval_data}, split="eval")
        eval_dataset = eval_dataset.map(lambda x: to_text(x, tokenizer), remove_columns=[c for c in eval_dataset.column_names if c != "text"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = build_sft_trainer(model, tokenizer, train_dataset, eval_dataset, args)
    trainer.train()

    history = trainer.state.log_history
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
