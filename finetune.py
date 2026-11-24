#!/usr/bin/env python3
import json
import sys
import os

MODEL_ID = "google/gemma-3-1b-it"
DATA_PATH = "train.jsonl"
OUTPUT_DIR = "./serac-gemma-adapter"

COLAB_TRAINING_CODE = '''\
import json, os, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "google/gemma-3-1b-it"
DATA_PATH = "train.jsonl"   # upload via the Files panel (left sidebar)
OUTPUT_DIR = "./serac-gemma-adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

records = [json.loads(l) for l in open(DATA_PATH)]

def format_example(r):
    messages = [
        {"role": "user",      "content": r["instruction"]},
        {"role": "assistant", "content": r["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list([format_example(r) for r in records])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Step {state.global_step:4d} | loss={logs['loss']:.4f}", flush=True)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=False,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    callbacks=[PrintLossCallback()],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"\\nAdapter saved to {OUTPUT_DIR}")
'''


def generate_colab_notebook():
    def cell(cell_type, source, cell_id):
        base = {"cell_type": cell_type, "id": cell_id, "metadata": {}, "source": source}
        if cell_type == "code":
            base["outputs"] = []
            base["execution_count"] = None
        return base

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": [
            cell("markdown", (
                "# Fine-tune Gemma-3-1b-it with LoRA (4-bit, T4 GPU)\n\n"
                "> **Before running:** set runtime to T4 GPU via `Runtime > Change runtime type > T4 GPU`\n\n"
                "Upload `train.jsonl` using the Files panel (folder icon, left sidebar)."
            ), "md_title"),
            cell("code", "!pip install -q trl peft bitsandbytes datasets transformers accelerate", "install"),
            cell("code", COLAB_TRAINING_CODE, "train"),
        ],
    }

    with open("finetune_colab.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Device: {device}  |  dtype: {dtype}")

    # Load and format dataset
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} training examples.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def format_example(r):
        messages = [
            {"role": "user",      "content": r["instruction"]},
            {"role": "assistant", "content": r["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = Dataset.from_list([format_example(r) for r in records])

    print("Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map={"": device},
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    class PrintLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                print(f"Step {state.global_step:4d} | loss={logs['loss']:.4f}", flush=True)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=512,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            callbacks=[PrintLossCallback()],
        )

        print("\n--- Training started ---")
        trainer.train()
        print("--- Training complete ---\n")

        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Adapter saved to {OUTPUT_DIR}")

    except (NotImplementedError, RuntimeError) as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("mps", "not implemented", "not supported", "aten::")):
            print(f"\n[MPS error] {exc}\n")
            generate_colab_notebook()
            sep = "=" * 62
            print(sep)
            print("  MPS IS NOT COMPATIBLE WITH THIS TRAINING CONFIGURATION")
            print(sep)
            print("  A Colab notebook was saved as: finetune_colab.ipynb")
            print("  Open it in Google Colab with a T4 GPU runtime:")
            print("    Runtime > Change runtime type > T4 GPU")
            print("  Upload train.jsonl via the Files panel, then run all cells.")
            print(sep)
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
