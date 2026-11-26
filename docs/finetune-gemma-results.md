# Fine-tune vs RAG, re-run Gemma-vs-Gemma

_Task 5 of the evaluation-hardening plan. v1 run 2026-11-25; de-leaked v2 run 2026-11-26
([v2 section](#v2-de-leaked-eval-set-primary-result)). Scripts: `leakage_check.py`,
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

**Bottom line (v1):** the TinyLlama-era statement "the fine-tune beat RAG" does not survive.
Gemma against Gemma, no arm is separable from another at n=40 with this metric. The adapter's
point estimate leads, but that lead sits almost entirely on items it was effectively trained on.

**Update: v2, the de-leaked eval set, is now the primary result.** v2 has 40 items, and every one
passes a train-overlap gate. On v2:

- **Adapter and flat RAG tie:** adapter − RAG is −0.000, CI [−0.076, +0.080].
- **Adapter vs base is a borderline gain:** adapter − base is +0.025, CI [−0.000, +0.050], with
  29 wins and 11 losses. This is a small, consistent edge whose CI just touches zero.
- **RAG vs base is not separable:** +0.026, CI [−0.050, +0.096].

The v1 lead of the fine-tune over RAG (+0.056) was entirely a leakage effect.

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

## v2: de-leaked eval set (primary result)

_Run 2026-11-26 at commit `7848d72`. Settings are identical to v1: fp32, MPS, greedy,
`max_new_tokens=256`, seed 0, flat top-4 retriever, same prompts and chat template. The eval set
`evaluation_prompts_v2.jsonl`, its key-term file and the review sheet are corpus-derived and live
only in the private run dir. This section names ids, not questions._

### How v2 was built

**The gate** (`leakage_check.py --gate --prompts … --keyterms …`, which exits 1 on any failure).
For every v2 item:

- its instruction and its reference answer each have max MiniLM cosine **< 0.8** against
  **every** training instruction **and** every training output, in both `train.jsonl` and
  `train.jsonl.bak` (204 rows);
- none of its declared key terms (parameter names, numbers, distinctive manual terms; one or two
  per item) occurs in any training output, by case-insensitive substring search;
- no other v2 item has instruction or reference cosine ≥ 0.85 with it.

**Final gate result: 40/40 pass.**

| Gate measure | Max over v2 items |
|---|---|
| Train cosine, instruction | 0.776 |
| Train cosine, reference | 0.791 |
| Intra-set cosine, instruction | 0.838 |
| Intra-set cosine, reference | 0.734 |

**What was kept.** The plan was to keep all 15 "clean" v1 items (19, 21, 25–28, 31–34, 36–40)
unchanged. **Only 10 survive** the stricter gate: 25, 26, 27, 28, 31, 33, 34, 37, 38, 39. They keep
their v1 ids. Five fail the key-term check, because the fact they test is in the SFT data even
though the wording differs:

| Id | Why it fails |
|---|---|
| 19 | its remedy appears in a trained fault-code row |
| 21 | the same fact as id 1 (trained) |
| 32 | the fact appears in a trained HMI row |
| 36 | the named test appears in a trained output |
| 40 | its key phrase is a substring of a trained output |

**Borderline items.** All nine (3, 5, 16, 18, 23, 24, 29, 30, 35) fail the cosine gate; their max
train cosine is 0.80–0.90. None is kept.

**What was replaced.** The 16 near-duplicates, the 9 borderline items and the 5 key-term failures,
30 items in all, were replaced by new questions with ids 101–135, so v1 and v2 answers cannot be
confused. Each new question:

- asks about a fact from the operator manual that `train.jsonl` does not cover;
- is answerable from one manual section;
- has a reference answer written from that section. The builder checks that a supporting excerpt
  is a literal substring of the manual text.

Four of the first 30 drafts (105, 120, 122, 128) and one reserve (132) failed the gate: 120 and
128 on cosine, 105, 122 and 132 on key terms. They were swapped for other reserves.

**Topic mix.** v2 has 20 troubleshooting / "something went wrong" items, 10 procedures, 6
screens/navigation and 4 parameters. v1 was more alarm-heavy (about 27 alarm items). The
troubleshooting manual's alarm list is fully covered by the SFT data, so the new troubleshooting
items come from operator-manual situations instead, such as unexpected screen states,
zeroing and calibration problems, and test or backup failures.

**Reference length.** New references average 19 words. The 10 kept v1 items average 7 words, so
the v2 mean is 16 words, against 23 for v1. The SFT data's newer rows share this terse register,
which favours the adapter under a cosine-to-reference metric. The style bias is still present in
v2; only the fact overlap was removed.

**Review sheet.** `eval_v2_review_sheet.csv` lists each item's question, reference, source
heading, supporting excerpt, key terms, max train cosine, and a blank `ok` column for hand-checking.
It has not been hand-checked yet. Two of the kept v1 items (37 and 39) have no verbatim support in
the manual for their reference wording, and they were kept unchanged as instructed.

**Answer reuse.** For the 10 kept items, v1 answers were reused only after the runner confirmed
that the rendered token ids were byte-identical to the v1 run (`--reuse-from ''`). A fresh
regeneration of one kept item reproduced the v1 answer exactly in both the base and RAG arms.
Only the 30 new items were generated: 120 generations across the four arms.

### v2 results (n = 40, all gate-clean)

| Arm | Mean cosine | Median | Wins (best of 4) | Hit 256-token cap | Mean words | Abstentions |
|---|---|---|---|---|---|---|
| base Gemma | 0.4846 | 0.5105 | 5/40 | 38/40 | 167 | 0 |
| base + adapter | 0.5099 | 0.5389 | 11/40 | 30/40 | 156 | 0 |
| RAG + base Gemma | **0.5102** | **0.5525** | **15/40** | 0/40 | 14 | 6 |
| RAG + adapter | 0.4450 | 0.4675 | 7/40 | 0/40 | 13 | 10 |

The "wins" count questions where the arm beat the other three. Two questions had ties at the top,
so wins sum to 38.

| Paired delta | n | Mean Δ | 95% CI (paired bootstrap, 10k, seed 20261125) | W/L/T |
|---|---|---|---|---|
| **adapter − base** | 40 | +0.0253 | [−0.0002, +0.0504] | 29/11/0 |
| **adapter − RAG** | 40 | −0.0003 | [−0.0760, +0.0800] | 18/22/0 |
| **RAG − base** | 40 | +0.0256 | [−0.0501, +0.0962] | 23/17/0 |
| **RAG+adapter − base** | 40 | −0.0396 | [−0.1247, +0.0411] | 20/20/0 |
| RAG+adapter − RAG | 40 | −0.0652 | [−0.1385, +0.0043] | 9/23/8 |

Split by provenance:

| Subset | n | base | adapter | RAG | adapter − base [CI] | adapter − RAG [CI] | RAG − base [CI] |
|---|---|---|---|---|---|---|---|
| new items (101–135) | 30 | 0.499 | 0.528 | 0.543 | +0.029 [−0.003, +0.060] | −0.015 [−0.102, +0.080] | +0.044 [−0.045, +0.125] |
| kept v1 items | 10 | 0.443 | 0.456 | 0.413 | +0.013 [−0.018, +0.043] | +0.042 [−0.108, +0.210] | −0.030 [−0.177, +0.109] |

Context for the new items:

- **Retrieval coverage.** The flat retriever returned a chunk containing one of the item's key
  terms for 25 of 30 new items. RAG averages 0.555 on those 25 and 0.480 on the other 5.
- **Length does not explain the adapter's edge here.** In v1, the adapter − base delta correlated
  with answer-length difference (r = −0.38). In v2 the correlation is r = +0.08, so the small edge
  is not explained by shorter answers. With this metric it is still impossible to tell domain
  vocabulary and register apart from correct content.

### Seen-in-training subset (secondary)

This subset is the 16 v1 near-duplicates (1, 2, 4, 6–15, 17, 20, 22), scored from the v1 answers.
These are facts the adapter trained on, reworded.

| Arm | Mean | Median | Wins (best of 4) |
|---|---|---|---|
| base | 0.5374 | 0.5286 | 4/16 |
| adapter | **0.5772** | **0.5958** | **9/16** |
| RAG | 0.4473 | 0.4429 | 1/16 |
| RAG + adapter | 0.4160 | 0.3832 | 2/16 |

| Paired delta | n | Mean Δ | 95% CI | W/L/T |
|---|---|---|---|---|
| adapter − base | 16 | +0.0399 | [−0.0148, +0.0920] | 9/7/0 |
| adapter − RAG | 16 | +0.1299 | [+0.0424, +0.2247] | 13/3/0 |
| RAG − base | 16 | −0.0901 | [−0.1867, +0.0004] | 5/11/0 |
| RAG+adapter − base | 16 | −0.1213 | [−0.2186, −0.0247] | 4/12/0 |

On trained facts, the adapter clearly beats flat RAG. Against base, its gain (+0.040) is larger
than on clean items (+0.025), but still not significant at n=16. Recall of trained material is
modest, and the large adapter − RAG gap here comes mostly from RAG doing badly on these
long-reference alarm items: −0.090 vs base.

### What changed vs v1

| Comparison | v1 (40, leaked) | v2 (40, clean) | Seen subset (16) |
|---|---|---|---|
| adapter − base | +0.022 [−0.009, +0.052] | +0.025 [−0.000, +0.050] | +0.040 [−0.015, +0.092] |
| adapter − RAG | +0.056 [−0.012, +0.125] | **−0.000 [−0.076, +0.080]** | +0.130 [+0.042, +0.225] |
| RAG − base | −0.034 [−0.098, +0.027] | +0.026 [−0.050, +0.096] | −0.090 [−0.187, +0.000] |
| RAG+adapter − base | −0.083 [−0.153, −0.016] | −0.040 [−0.125, +0.041] | −0.121 [−0.219, −0.025] |

1. **Fine-tune vs RAG.** The v1 point-estimate lead disappears on clean items. They tie, and RAG
   has the higher median and more wins. The v1 lead was a leakage effect.
2. **Fine-tune vs base.** The adapter's small edge over the untuned model holds on clean items
   (+0.025, 29/40 wins), but the CI's lower bound is −0.0002. Call it borderline, not
   established. It may reflect domain register rather than knowledge.
3. **RAG + adapter** is still the worst arm, though on v2 its gap to base is no longer
   significant.
4. **The Week 1 conclusion stays struck.** No arm is significantly better than another on the
   clean set at n=40 with this metric.

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

v2 run, generating 30 new items per arm with 10 reused:

| Arm | Wall time |
|---|---|
| base | 443.0 s |
| adapter | 501.5 s |
| RAG | 54.7 s |
| RAG + adapter | 52.4 s |

Peak MPS driver-allocated memory was 5.34 GB and peak process RSS was 1.14 GB.

## Reproduce

```
cd <private run dir>        # with evaluation_prompts.jsonl, train.jsonl(.bak), faiss.index,
                            # chunks.pkl, serac-gemma-adapter/, Week 1 CSVs linked in
HF_HUB_OFFLINE=1 python3 <repo>/leakage_check.py --bak
HF_HUB_OFFLINE=1 python3 <repo>/run_gemma_arms.py \
    --arms base,adapter,rag,rag_adapter,adapter_w1repro --device mps --dtype fp32 \
    --max-new-tokens 256 --seed 0
HF_HUB_OFFLINE=1 python3 <repo>/score_gemma_arms.py

# v2 (de-leaked set; evaluation_prompts_v2.jsonl + key terms built privately)
HF_HUB_OFFLINE=1 python3 <repo>/leakage_check.py --gate --prompts evaluation_prompts_v2.jsonl \
    --keyterms evaluation_prompts_v2_keyterms.json            # exit 1 = gate failure
HF_HUB_OFFLINE=1 python3 <repo>/run_gemma_arms.py --arms base,adapter,rag,rag_adapter \
    --prompts evaluation_prompts_v2.jsonl --suffix _v2 --reuse-from '' \
    --device mps --dtype fp32 --max-new-tokens 256 --seed 0
HF_HUB_OFFLINE=1 python3 <repo>/score_gemma_arms.py --arms base,adapter,rag,rag_adapter \
    --prompts evaluation_prompts_v2.jsonl --suffix _v2 --out-prefix gemma_arms_v2 --no-week1
HF_HUB_OFFLINE=1 python3 <repo>/score_gemma_arms.py --arms base,adapter,rag,rag_adapter \
    --ids 1,2,4,6,7,8,9,10,11,12,13,14,15,17,20,22 --out-prefix gemma_arms_seen --no-week1
```

`run_gemma_arms.py` appends and fsyncs each answer as it is produced, and skips ids already
written. A crashed run can be resumed with the same command.
