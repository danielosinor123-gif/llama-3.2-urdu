import argparse
import json
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def load_prompt_response_dataset(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item["prompt"]
            response = item["response"]
            rows.append({"prompt": prompt, "response": response})
    return rows


def format_chat(example):
    prompt = example["prompt"]
    response = example["response"]
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{response}<|eot_id|>"
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 for Urdu using QLoRA.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to a JSONL training file with prompt/response pairs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Train batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Fraction of data used for validation.")
    args = parser.parse_args()

    rows = load_prompt_response_dataset(args.train_data)
    if len(rows) < 2:
        raise ValueError("Training dataset is too small. Add more examples.")

    dataset = load_dataset("json", data_files={"train": args.train_data}, split="train")
    dataset = dataset.map(lambda x: {"prompt": x["prompt"], "response": x["response"]})

    split = dataset.train_test_split(test_size=args.eval_ratio, seed=42)
    train_dataset = split["train"].map(format_chat)
    eval_dataset = split["test"].map(format_chat)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=50 if len(eval_dataset) > 0 else None,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="tensorboard",
        remove_unused_columns=False,
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
