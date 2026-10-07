import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
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


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Urdu model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to send to the model.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of generated tokens.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    if args.prompt is None:
        while True:
            user_input = input("You (Urdu assistant): ")
            if user_input.lower() in {"exit", "quit", "q"}:
                break
            answer = generate_response(model, tokenizer, user_input, max_new_tokens=args.max_new_tokens)
            print(f"Assistant: {answer}\n")
    else:
        answer = generate_response(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
        print(answer)


if __name__ == "__main__":
    main()
