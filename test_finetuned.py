#!/usr/bin/env python3
"""Test the fine-tuned Gemma adapter.

Usage:
    python3 test_finetuned.py "Your question here"
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_PATH = "./serac-gemma-adapter"


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 test_finetuned.py "Your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    print(f"Device: {device}")
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map={"": device},
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nQuestion: {question}\n")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
