# Fine-tune vs RAG, re-run Gemma-vs-Gemma

_Task 5 of the evaluation-hardening plan, run 2026-11-25. Scripts: `leakage_check.py`,
`run_gemma_arms.py`, `score_gemma_arms.py`. This page reports aggregates and question ids
only. The generated answers, per-item CSVs and logs live in the private run directory, because
they quote the proprietary manuals._

## Summary

- **Train/eval leakage is substantial.** Eval ids 1–20 are paraphrases of training rows 0–19,
  written in the same order. 15 of those 20 have an instruction with MiniLM cosine above 0.9 to a
  training instruction (eval 13 scores 0.997). No item is an exact string duplicate.
- **The fine-tune does not significantly beat the base model.** Adapter − base is
  **+0.022 (95% CI −0.009 to +0.052)**: 24 wins, 16 losses. On the 15 items with no
  near-duplicate in the training set (both cosines ≤ 0.8), the adapter scores *below* base
  (−0.013, CI −0.057 to +0.022).
- **The fine-tune does not significantly beat RAG.** Adapter − RAG is
  **+0.056 (CI −0.012 to +0.125)**. The whole advantage comes from the leaked ids 1–20
  (+0.122, CI +0.041 to +0.212). On ids 21–40 RAG is level (−0.010, CI −0.110 to +0.091).
- **The Week 1 "TinyLlama stand-in" label was only half true.** The stored
  `finetuned_outputs.csv` is reproduced **verbatim, 40/40**, by Gemma-3-1b-it plus the adapter
  (fp32, MPS, greedy, with a doubled `<bos>`, exactly as `run_finetuned.py` ran it). Only the
  RAG arm used TinyLlama. Week 1 therefore compared Gemma+adapter against TinyLlama+RAG, which
  confounds the generator with the method.
- **The Week 1 numbers reproduce exactly** with the same scorer: RAG 0.4895, fine-tune 0.5079,
  22/40 wins. The largest difference from `comparison.csv` is 8e-17. Even in Week 1, the delta's
  CI crossed zero (+0.018, CI −0.046 to +0.082).

**Bottom line:** the TinyLlama-era statement "the fine-tune beat RAG" does not survive. Gemma
against Gemma, no arm is separable from another at n=40 with this metric. The adapter's point
estimate leads, but that lead sits almost entirely on items it was effectively trained on.

## Setup

| | |
|---|---|
| Base model | `google/gemma-3-1b-it`, HF cache snapshot `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Adapter | `serac-gemma-adapter`, LoRA r=8, alpha=16, dropout 0.05, `q_proj`/`v_proj`; `adapter_model.safetensors` sha256 `cc6a4e98660237a82d476688eef2fb529d126bd75d030d51f7ffd43a8131201c` |
| Inference | base loaded **non-quantized in fp32**, adapter applied unmerged with `peft.PeftModel` |
| Device | Apple M4 (16 GB unified memory), PyTorch MPS backend |
| Decoding | greedy (`do_sample=False`, `num_beams=1`, sampling params cleared), `max_new_tokens=256` (same as Week 1), stop on `<eos>`/`<end_of_turn>`, no repetition penalty, seed 0 |
| Prompt (arms 1, 2) | Gemma chat template, one user turn = the instruction, no system prompt, single `<bos>`. This matches the training format in `finetune.py` (`apply_chat_template([user, assistant])`). The adapter's saved `chat_template.jinja` is byte-identical to the base tokenizer's. |
| Prompt (arm 3) | the same template; the user turn is the Week 1 grounded prompt (verbatim from `naive_rag_baseline.py`), with the top-4 chunks joined by `\n\n---\n\n` |
| Retriever | Week 1 flat index: `faiss.index` + `chunks.pkl` (pypdf text, 500/50 chunks), `all-MiniLM-L6-v2`, top-4. **No context truncation.** The joined context ran 1,535–1,964 chars and prompts reached at most 668 tokens, far below Gemma 3's 32k window. The 1,200-char cap in `hybrid_intent_router.py` was a TinyLlama workaround and was not used here. |
| Scorer | cosine of `all-MiniLM-L6-v2` embeddings of the answer vs `reference_answer`, the same computation as `score.py` |
| Stats | paired percentile bootstrap over question ids, 10,000 resamples, seed 20261125 |
| Libraries | Python 3.9.6, torch 2.8.0, transformers 4.57.6, peft 0.17.1, sentence-transformers 5.1.2, faiss 1.13.0 |
| Code | generation ran at commit `28270e7` (`run_gemma_arms.py`) |

**Isolation.** The model is loaded once. The base arms run first. The adapter is then wrapped onto
the same model object, so arms 1 and 2 share the same tokenizer, token ids, dtype, device and
decoding: the only difference is the LoRA weights. Arm 3 differs from arm 1 only in the user-turn
text.

**Inference precision differs from training.** `finetune.py` documents a QLoRA path (4-bit NF4,
bitsandbytes, CUDA). The adapter's own `training_args.bin` points elsewhere: logging dir
`…/runs/Jun11_16-52-43_<the author's MacBook>`, device `mps`, `bf16=False`, optimizer
`adamw_torch_fused`. bitsandbytes 4-bit does not run on MPS, so the adapter was most likely
trained by the **local, non-quantized** branch of `finetune.py`: base weights in bf16, LoRA,
no 4-bit. Either way, this evaluation applies the adapter to an fp32, non-quantized base. That
matches what Week 1's `run_finetuned.py` did. It is not 4-bit training-time quantization. The
docs that call the adapter "QLoRA (4-bit NF4)" are probably describing the Colab path that was
generated, not the run that produced this adapter.

**Why fp32.** Gemma 3 overflows in fp16. fp32 on MPS gave coherent output in a 2-prompt sanity
check on every arm, ran at about 15–19 tokens/s, and matches Week 1's fine-tune inference
setting. That last point is what allowed the verbatim reproduction check.

**Double `<bos>` in Week 1.** `run_finetuned.py` rendered the chat template, which already
starts with `<bos>`, and then re-tokenized the string with `add_special_tokens=True`. That adds a
second `<bos>`. Arms 1–4 here use a single `<bos>`. The diagnostic arm `adapter_w1repro` keeps
the doubled token on purpose, and reproduces `finetuned_outputs.csv` exactly. With one `<bos>`
the adapter scores 0.5201; with two it scores 0.5079.

## Leakage check

`leakage_check.py` compares each eval item (instruction + reference answer) against all 152
training rows using exact match, normalized match, MiniLM cosine, token Jaccard, and fault-code
overlap. For calibration: two *different* eval questions have a median nearest-neighbour cosine
of 0.53.

| Eval ids | Nearest train instruction cosine > 0.9 | > 0.8 | Nearest train output cosine > 0.8 | Mean nearest-instruction cosine |
|---|---|---|---|---|
| 1–20 (original set) | 15/20 | 18/20 | 8/20 | 0.903 |
| 21–40 (added at revision) | 1/20 (id 22) | 3/20 | 4/20 | 0.640 |

- **Exact duplicates:** none, for instructions or for references.
- **Ids 1–20 map onto train rows 0–19 in order.** For 14 of the 20, the nearest training
  instruction is exactly the row at the same position. The training rows give the same fault
  and the same remedy, reworded. The revision script stripped the leading fault-code sentence from
  training outputs but not from the references, which partly explains why reference/output
  cosines are lower than instruction cosines.
- **Near-duplicate set** (instruction or reference cosine > 0.9), 16 items: ids
  1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22.
- **Clean set** (both cosines ≤ 0.8), 15 items: ids 19, 21, 25, 26, 27, 28, 31, 32, 33, 34,
  36, 37, 38, 39, 40.
- **Eval-internal duplicate:** id 22 is a paraphrase of id 7 (cosine 0.967).
- **The `.bak` files** show the history. The pre-revision sets were 20 eval and 52 train items.
  The revision kept all of them and added eval ids 21–40 plus 100 training rows. The new eval
  references and the new training outputs share a terse style (median 7 words each, against
  36–50 words for the original items). For 16 of the 20 new eval ids, the nearest training pair
  is one of the 100 added rows. These are topically related one-line answers rather than
  paraphrases.
- **Which training set did the adapter see?** The adapter files are dated 11 June. The script
  that built the 52-row set is dated 10 June. The script that added eval ids 21–40 is dated
  17 June. The adapter was therefore most likely trained on the 52-row set. The paraphrase leak
  on ids 1–20 exists in both versions. A style overlap on ids 21–40 would only matter if the
  adapter saw the 100 added rows, which the dates suggest it did not. This could not be
  confirmed from the artifacts, because no step count was saved.

## Results: three arms plus optional and diagnostic arms (n = 40)

| Arm | Mean cosine | Median | Wins (best of 4 main arms) | Answers hitting the 256-token cap | Mean words | "Not in context" abstentions |
|---|---|---|---|---|---|---|
| 1. base Gemma | 0.4982 | 0.5084 | 8/40 | 40/40 | 173 | 0 |
| 2. base + adapter | **0.5201** | **0.5486** | **19/40** | 30/40 | 158 | 0 |
| 3. RAG + base Gemma | 0.4642 | 0.4921 | 7/40 | 0/40 | 14 | 6 |
| 4. RAG + adapter (optional) | 0.4150 | 0.3977 | 5/40 | 0/40 | 12 | 7 |
| diag: adapter, Week 1 tokenization (double `<bos>`) | 0.5079 | 0.5473 | — | 31/40 | 152 | 0 |

Reference answers average 23 words. The "wins" column counts questions where the arm beat all
three other main arms. The four arms' wins add up to 39 because of one tie at the top.

| Paired delta | n | Mean Δ | 95% CI | W/L/T |
|---|---|---|---|---|
| **adapter − base** | 40 | +0.0219 | [−0.0086, +0.0521] | 24/16/0 |
| **adapter − RAG** | 40 | +0.0558 | [−0.0119, +0.1252] | 24/16/0 |
| **RAG − base** | 40 | −0.0339 | [−0.0977, +0.0272] | 16/24/0 |
| RAG+adapter − RAG | 40 | −0.0492 | [−0.1121, +0.0146] | 8/21/11 |
| RAG+adapter − base | 40 | −0.0831 | [−0.1530, −0.0158] | 14/26/0 |
| RAG+adapter − adapter | 40 | −0.1050 | [−0.1737, −0.0378] | 13/27/0 |

All three primary deltas have CIs that cross zero. The only deltas that clear zero involve the
optional fourth arm: putting the adapter on top of RAG makes things *worse*.

### By leakage subset

| Subset | n | base | adapter | RAG | adapter − base [CI] | adapter − RAG [CI] | RAG − base [CI] |
|---|---|---|---|---|---|---|---|
| near-dup (cos > 0.9) | 16 | 0.537 | 0.577 | 0.447 | +0.040 [−0.015, +0.092] | **+0.130 [+0.042, +0.225]** | −0.090 [−0.187, +0.000] |
| not near-dup | 24 | 0.472 | 0.482 | 0.476 | +0.010 [−0.028, +0.045] | +0.006 [−0.083, +0.102] | +0.004 [−0.079, +0.080] |
| clean (both cos ≤ 0.8) | 15 | 0.474 | 0.462 | 0.479 | −0.013 [−0.057, +0.022] | −0.018 [−0.139, +0.113] | +0.005 [−0.105, +0.105] |
| ids 1–20 | 20 | 0.535 | 0.562 | 0.440 | +0.026 [−0.019, +0.071] | **+0.122 [+0.041, +0.212]** | **−0.095 [−0.179, −0.018]** |
| ids 21–40 | 20 | 0.461 | 0.479 | 0.488 | +0.017 [−0.025, +0.056] | −0.010 [−0.110, +0.091] | +0.027 [−0.063, +0.114] |

The pattern is consistent across subsets. The adapter's lead grows with how closely an item
matches the training data, and it vanishes or reverses on clean items. RAG is behind only on
ids 1–20. Those are the long-reference items, whose references were written in the same style
as the SFT data.

## Week 1 restated Gemma-vs-Gemma

| Claim | TinyLlama era (as published) | Gemma vs Gemma (this run) |
|---|---|---|
| Fine-tune mean | 0.5079 (Gemma+adapter, double `<bos>`) | 0.5201 (adapter); base Gemma 0.4982 |
| RAG mean | 0.4895 (**TinyLlama** generator) | 0.4642 (Gemma generator) |
| Fine-tune − RAG | +0.0185 [−0.046, +0.082], 22/40 wins | +0.0558 [−0.012, +0.125], 24/40 wins |
| Fine-tune − base model | not measured | +0.0219 [−0.009, +0.052], 24/40 wins |

Swapping TinyLlama for Gemma in the RAG arm *lowered* its score: Gemma RAG − TinyLlama RAG is
−0.025 [−0.088, +0.039]. Gemma follows the "use ONLY the context" instruction literally. It gives
14-word answers and declines 6/40 when the retrieved chunks don't contain the answer, whereas
TinyLlama produced longer and more reference-like prose. This is a property of the metric as
much as of the generator.

**Does the Week 1 conclusion survive?** No, not as a finding. The direction (fine-tune ahead of
RAG) holds as a point estimate, but:

1. it is not statistically distinguishable from zero, either then or now;
2. the fine-tune is not distinguishable from the untuned base model;
3. its advantage is concentrated on eval items that paraphrase its training data, and it
   disappears on the 15 clean items.

The defensible statement is that at n=40, on this metric, base Gemma, Gemma+LoRA and Gemma+flat
RAG perform about the same, and the eval set needs to be de-leaked before any of them can be
ranked.

## Why the metric is weak here

- **It conflates retrieval with fluency and length.** Base Gemma writes 173-word generic answers
  and gets truncated on every question. It still scores 0.498, because generic troubleshooting
  prose about "the machine" is close to any troubleshooting reference in MiniLM space. RAG's
  terse, grounded answers, and its honest "not in the context" abstentions, are penalized for
  being short.
- **It rewards style similarity to the references.** The SFT data was synthesized in the same
  voice as the references. Across questions, the adapter − base delta correlates with how much
  shorter the adapter's answer was than base's (r = −0.38). Part of the adapter's gain is
  closer length and register, not more correct content.
- **It doesn't check facts.** A confident wrong remedy that uses the right vocabulary can
  outscore a correct terse one. No fault-code or remedy-correctness check is applied.
- **n=40, with one internal duplicate (ids 7 and 22).** CIs are about ±0.03 for adapter − base
  and about ±0.07 for comparisons involving RAG. Effects below roughly 0.05 cannot be resolved.

## Resources

| Arm | Wall time (generation) | Generated tokens |
|---|---|---|
| base | 555.5 s | 10,240 |
| adapter | 500.6 s | 9,047 |
| RAG | 60.5 s | 743 |
| RAG + adapter | 55.1 s | 696 |
| adapter, Week 1 tokenization | 461.8 s | 8,819 |

Model load took about 3 s from the local cache. Memory, on Apple unified memory:

- **Peak MPS driver-allocated memory: 5.29 GB.** This is the high-water of
  `torch.mps.driver_allocated_memory()`, sampled after each generate call.
- **Peak process RSS: 1.09 GB.** This is `ru_maxrss`. It does not include MPS allocations.

Neither figure is VRAM.

## Reproduce

```
cd <private run dir>        # with evaluation_prompts.jsonl, train.jsonl(.bak), faiss.index,
                            # chunks.pkl, serac-gemma-adapter/, Week 1 CSVs linked in
HF_HUB_OFFLINE=1 python3 <repo>/leakage_check.py --bak
HF_HUB_OFFLINE=1 python3 <repo>/run_gemma_arms.py \
    --arms base,adapter,rag,rag_adapter,adapter_w1repro --device mps --dtype fp32 \
    --max-new-tokens 256 --seed 0
HF_HUB_OFFLINE=1 python3 <repo>/score_gemma_arms.py
```

`run_gemma_arms.py` appends and fsyncs each answer as it is produced, and skips ids already
written. A crashed run can be resumed with the same command.
