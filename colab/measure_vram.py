# -*- coding: utf-8 -*-
"""
Measure peak GPU memory (VRAM) on a Colab T4 for the QLoRA training arm and the
inference arms. This has NOT been run yet: the repo's other numbers come from an
Apple M4 (unified memory, no CUDA), where VRAM does not exist. See
docs/latency-memory.md.

How to run
----------
1. Colab: Runtime > Change runtime type > T4 GPU.
2. Gemma-3 is gated. Accept the licence on huggingface.co/google/gemma-3-1b-it,
   then add a Colab secret named HF_TOKEN (key icon, left sidebar) or run
   `from huggingface_hub import login; login()` before cell 3.
3. Optional: upload your own train.jsonl ({"instruction", "output"} per line).
   Without it, cell 2 builds training pairs from questions/public_seed.jsonl
   (public, CC BY 4.0). Never upload proprietary data to a shared notebook.
4. Clone the repo and work from its root:
     !git clone https://github.com/Navraj-bite/proxy-pointer-rag.git
     %cd proxy-pointer-rag
   then run the whole file with `!python colab/measure_vram.py`, or paste the
   "# %%" blocks into Colab cells and run them top to bottom.

What is measured
----------------
torch.cuda.max_memory_allocated() (peak tensor memory) and
torch.cuda.max_memory_reserved() (peak held by the caching allocator, closer to
what nvidia-smi shows), reset before each arm:

  train_qlora_data      finetune.py's QLoRA config on the training pairs
  train_qlora_512       same, every example at the full max_length=512 tokens
                        (upper bound for this config: batch 1, grad-accum 4)
  infer_gemma_base      google/gemma-3-1b-it, dtype as in run_finetuned.py
  infer_gemma_adapter   same + the LoRA adapter trained in cell 3
  infer_tinyllama       TinyLlama-1.1B-Chat fp32 (the RAG/router generator)

Each inference arm runs a short prompt (bare question) and a long prompt
(~1,800 tokens of context, the size an uncapped proxy-pointer context reaches),
greedy, max_new_tokens 256 for Gemma (run_finetuned.py) and 150 for TinyLlama
(run_day10_benchmark.py).

Training runs MAX_STEPS optimizer steps rather than 3 full epochs: memory peaks
in the first steps (weights, LoRA optimizer state and activations are all
allocated by step 1), so a short run gives the same peak. Set FULL_EPOCHS=True
to train the full 3 epochs instead.

Output: a table and vram_results.json.
"""

# %% [1] install (Colab) -----------------------------------------------------
import subprocess
import sys

if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl", "peft",
                    "bitsandbytes", "datasets", "transformers", "accelerate"], check=True)

# %% [2] config (reused from finetune.py) and training data ------------------
import datetime
import gc
import json
import os
import platform
import time

import torch

assert torch.cuda.is_available(), "needs a CUDA GPU (Colab T4)"

MODEL_ID = "google/gemma-3-1b-it"            # finetune.py
TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./vram-gemma-adapter"           # throwaway adapter for the inference arm
LORA = dict(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")   # finetune.py
TRAIN = dict(per_device_train_batch_size=1, gradient_accumulation_steps=4,
             learning_rate=2e-4, max_length=512, num_train_epochs=3)  # finetune.py
MAX_STEPS = 20
FULL_EPOCHS = False
GEMMA_INFER_DTYPE = torch.float32             # run_finetuned.py loads float32
TINYLLAMA_DTYPE = torch.float32               # run_day10_benchmark.py / hybrid_intent_router.py
LONG_PROMPT_TOKENS = 1800
SEED = 0

if "google.colab" in sys.modules and not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        pass


def load_training_records():
    if os.path.exists("train.jsonl"):
        src = "train.jsonl (user-supplied)"
        recs = [json.loads(l) for l in open("train.jsonl") if l.strip()]
    else:
        path = "questions/public_seed.jsonl"
        if not os.path.exists(path):
            raise SystemExit(
                "no train.jsonl and no questions/public_seed.jsonl: run from the repo root, e.g.\n"
                "  !git clone https://github.com/Navraj-bite/proxy-pointer-rag.git\n"
                "  %cd proxy-pointer-rag\n"
                "or upload a train.jsonl")
        src = "questions/public_seed.jsonl (public)"
        recs = [{"instruction": q["question"], "output": q["reference_answer"]}
                for q in (json.loads(l) for l in open(path) if l.strip())]
    return recs, src


RECORDS, DATA_SOURCE = load_training_records()
print(f"{len(RECORDS)} training pairs from {DATA_SOURCE}")

RESULTS = {}


def gpu_reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def gpu_peak():
    return {"max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved())}


# %% [3] training arm: QLoRA exactly as finetune.py ---------------------------
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

BF16 = torch.cuda.is_bf16_supported()   # finetune.py's Colab code uses bf16=True


def train_arm(name, records, pad_to_max):
    set_seed(SEED)
    gpu_reset()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def fmt(r):
        msgs = [{"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["output"]}]
        return tok.apply_chat_template(msgs, tokenize=False)

    texts = [fmt(r) for r in records]
    if pad_to_max:   # repeat each example until it exceeds max_length, SFT truncates to 512
        texts = [(t + "\n") * (1 + TRAIN["max_length"] // max(1, len(tok(t).input_ids)))
                 for t in texts]
    ds = Dataset.from_list([{"text": t} for t in texts])
    compute = torch.bfloat16 if BF16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=compute))
    after_load = gpu_peak()
    cfg = SFTConfig(output_dir=OUTPUT_DIR, logging_steps=1, save_strategy="no", report_to="none",
                    bf16=BF16, fp16=not BF16, seed=SEED,
                    max_steps=-1 if FULL_EPOCHS else MAX_STEPS, **TRAIN)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=LoraConfig(**LORA), processing_class=tok)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    RESULTS[name] = {**gpu_peak(), "after_model_load": after_load, "train_s": elapsed,
                     "steps": trainer.state.global_step, "examples": len(texts),
                     "pad_to_max_length": pad_to_max, "precision": "bf16" if BF16 else "fp16",
                     "data_source": DATA_SOURCE}
    if not pad_to_max:
        trainer.save_model(OUTPUT_DIR)
    del trainer, model
    gpu_reset()
    print(name, RESULTS[name])


train_arm("train_qlora_data", RECORDS, pad_to_max=False)
train_arm("train_qlora_512", RECORDS, pad_to_max=True)

# %% [4] inference arms ------------------------------------------------------
from peft import PeftModel

FILLER = ("The status LED blinks while the vehicle initialises its sensors and then "
          "turns solid once the estimator has converged. ")


def prompts(tok):
    long_ctx = FILLER
    while len(tok(long_ctx).input_ids) < LONG_PROMPT_TOKENS:
        long_ctx += FILLER
    q = "What does the status LED indicate after initialisation?"
    return {"short": [{"role": "user", "content": q}],
            "long": [{"role": "user", "content": f"Context:\n{long_ctx}\n\nQuestion: {q}"}]}


def infer_arm(name, model_id, dtype, max_new_tokens, adapter=None):
    gpu_reset()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": "cuda"})
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    res = {"after_model_load": gpu_peak(), "dtype": str(dtype), "max_new_tokens": max_new_tokens}
    for label, msgs in prompts(tok).items():
        torch.cuda.reset_peak_memory_stats()
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        res[label] = {**gpu_peak(), "prompt_tokens": int(ids.shape[1]),
                      "new_tokens": int(out.shape[1] - ids.shape[1]),
                      "generate_s": time.time() - t0}
    RESULTS[name] = res
    del model
    gpu_reset()
    print(name, res)


infer_arm("infer_gemma_base", MODEL_ID, GEMMA_INFER_DTYPE, 256)
infer_arm("infer_gemma_adapter", MODEL_ID, GEMMA_INFER_DTYPE, 256, adapter=OUTPUT_DIR)
infer_arm("infer_tinyllama", TINYLLAMA_ID, TINYLLAMA_DTYPE, 150)

# %% [5] report --------------------------------------------------------------
from importlib.metadata import version


def _v(m):
    try:
        return version(m)
    except Exception:
        return None


env = {
    "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    "cuda": torch.version.cuda,
    "bf16_supported": BF16,
    "python": platform.python_version(),
    "libraries": {m: _v(m) for m in ("torch", "transformers", "peft", "trl", "bitsandbytes",
                                     "accelerate", "datasets")},
    "seed": SEED, "models": [MODEL_ID, TINYLLAMA_ID],
}
GiB = 2 ** 30
print(f"\n{env['gpu']}  CUDA {env['cuda']}  torch {env['libraries']['torch']}\n")
print("| arm | prompt | peak allocated GiB | peak reserved GiB |")
print("|---|---|---|---|")
for name, r in RESULTS.items():
    if name.startswith("train"):
        print(f"| {name} | train ({r['steps']} steps) | {r['max_memory_allocated_bytes']/GiB:.2f} "
              f"| {r['max_memory_reserved_bytes']/GiB:.2f} |")
    else:
        for label in ("short", "long"):
            x = r[label]
            print(f"| {name} | {label} ({x['prompt_tokens']} tok) | "
                  f"{x['max_memory_allocated_bytes']/GiB:.2f} | {x['max_memory_reserved_bytes']/GiB:.2f} |")
with open("vram_results.json", "w") as f:
    json.dump({"env": env, "results": RESULTS}, f, indent=1)
print("\nwrote vram_results.json")
