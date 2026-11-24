# Week 1 — QLoRA Training Configuration

_Source: `finetune.py` (base model + LoRA + SFT config). Base model: **`google/gemma-3-1b-it`**._

| Hyperparameter | Value | Notes |
|---|---|---|
| **LoRA rank (`r`)** | `8` | |
| **LoRA alpha** | `16` | 2× rank |
| **LoRA dropout** | `0.05` | |
| **Target modules** | `q_proj`, `v_proj` | attention query/value projections only |
| **LoRA bias** | `none` | |
| **Task type** | `CAUSAL_LM` | |
| **Learning rate** | `2e-4` | |
| **Epochs** | `3` | |
| **Per-device batch size** | `1` | |
| **Gradient accumulation** | `4` | → **effective batch size = 4** |
| **Max sequence length** | `512` | |
| **Quantization** | **4-bit NF4** (`bnb_4bit_quant_type="nf4"`), compute dtype `bfloat16` | QLoRA via `bitsandbytes` (Colab/CUDA path) |
| **Precision** | `bf16=True` (Colab/T4), `bf16=False` (local MPS/CPU fallback) | |
| **Optimizer** | ⚠️ **not explicitly set** → framework default (`adamw_torch`) | not overridden in `SFTConfig` |
| **LR scheduler** | ⚠️ **not explicitly set** → framework default (`linear`) | not overridden in `SFTConfig` |
| **Trainer** | `trl.SFTTrainer` / `SFTConfig` | chat-template formatting via tokenizer |
| **Adapter output** | `./serac-gemma-adapter/` | |

> **Honesty note:** optimizer and LR scheduler are **not** specified in `finetune.py`; the values above are the Hugging Face `Trainer` defaults that apply when unset. If the slide needs exact confirmed values, print `trainer.args.optim` and `trainer.args.lr_scheduler_type` during a run.
