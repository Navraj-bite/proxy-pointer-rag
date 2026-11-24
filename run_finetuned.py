#!/usr/bin/env python3
"""Task 1: Run all 20 evaluation instructions through the fine-tuned Gemma model.

Loads base google/gemma-3-1b-it + serac-gemma-adapter LoRA weights ONCE,
then generates an answer for every instruction in evaluation_prompts.jsonl.
Saves finetuned_outputs.csv with columns: id, question, generated_answer.

Config: float32, device=mps, no quantization.
"""
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_PATH = "./serac-gemma-adapter"
PROMPTS_FILE = "evaluation_prompts.jsonl"
OUTPUT_FILE = "finetuned_outputs.csv"
DEVICE = "mps"
MAX_NEW_TOKENS = 256


def load_prompts(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} evaluation prompts")

    # --- Load base model + adapter ONCE, outside the loop ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading base model (float32, mps, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": DEVICE},
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    rows = []
    for p in prompts:
        question = p["instruction"]
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        rows.append(
            {"id": p["id"], "question": question, "generated_answer": answer}
        )
        print(f"[{p['id']:>2}/{len(prompts)}] done ({len(answer)} chars)")

    df = pd.DataFrame(rows, columns=["id", "question", "generated_answer"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
