# What the benchmark numbers measure

This document defines every published score in this repo: the Week 2 benchmark (30 questions,
vanilla RAG vs "Proxy-Pointer + router", the README results table) and the Week 1 benchmark
(40 prompts, fine-tuned SLM vs RAG). It says exactly which strings are compared, how, where the
reference answers come from, how `retrieval_correct` and latency are computed, which published
numbers were re-derived from the stored outputs, and what the numbers do *not* show.

Everything below was checked against the code in this repo and against the private run outputs
(the proprietary corpus, questions, reference answers and generated answers are not in the
repo). No generation was re-run. Only the MiniLM embedding and the retrieval code were re-run.

## Summary

| | Week 2 (README table) | Week 1 |
|---|---|---|
| Questions | 30, in `benchmark_questions.jsonl` (`id`, `question`, `category`, `reference_answer`), 10 each `tabular` / `procedure` / `navigation` | 40, in `evaluation_prompts.jsonl` (`id`, `instruction`, `reference_answer`) |
| Score | cosine(MiniLM(reference answer), MiniLM(generated answer)), mean over questions | same |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, `max_seq_length` 256 word-pieces | same |
| Systems | A: vanilla RAG (pypdf text, 500-char chunks, top-4) + TinyLlama. B: regex router → bare TinyLlama (`fast`) or proxy-pointer top-1 node + TinyLlama (`deep`) | RAG (pypdf chunks, top-4) + TinyLlama vs `gemma-3-1b-it` + LoRA adapter, no retrieval |
| "Retrieval correct" | keyword match against the **generated answer** (not the retrieved context) | not computed |
| Latency | wall-clock seconds around each system's answer call, 1 run, CPU | not recorded |
| Produced by | `run_day10_benchmark.py` | `run_finetuned.py`, `rag_new_questions.py` (+ an uncommitted step for RAG rows 1–20), `score.py` |

So both "cosine" scores measure **semantic similarity between a generated answer and a reference
answer**. That is an end-to-end answer-quality proxy. It is not a retrieval metric.

---

## Week 2 benchmark (`run_day10_benchmark.py`)

### What runs for each question

For each question, in file order (id 1 → 30), the script runs system A and then system B. Both
share one TinyLlama pipeline object (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `torch_dtype=float32`,
`device_map="cpu"`, greedy `do_sample=False`). `hybrid_answer()` sets the router's `_pipeline` to
the benchmark's already-loaded pipeline, so the generator weights really are the same.

| | A: "Vanilla RAG" | B: "Proxy-Pointer + router" (`hybrid_*` columns) |
|---|---|---|
| Source text indexed | `pypdf` raw text of the two PDFs (`naive_rag_baseline.load_pdfs`) | LlamaParse markdown of the same PDFs (`parse_documents.py`) |
| Index unit | `RecursiveCharacterTextSplitter(500, 50)` chunks. 799 vectors, median 466 chars | one *proxy* per heading node. 316 vectors (319 tree nodes minus `ROOT` and 2 `[FILE]` nodes) |
| Index | MiniLM, `normalize_embeddings=True`, FAISS `IndexFlatIP` (exact cosine) | same |
| Routing | none | `classify_query()`: count of matching regexes in `_DEEP_PATTERNS`. 0 → `fast`, ≥1 → `deep` |
| Retrieval | top-4 chunks, joined with `"\n\n---\n\n"` | `fast`: **none**. `deep`: top-1 proxy → `pointer_node_id` → that node's `# heading` + body, plus each **direct** child's `## heading` + body (`context_assembler.query`) |
| Context cap | `context[:1200]` | `deep`: `context[:1200]`. `fast`: no context |
| System prompt | "…Answer the question using ONLY the context below. If the answer is not in the context, say you don't know." | `deep`: "…Answer the operator's question using ONLY the context provided below. If the answer is not in the context, say you don't know." `fast`: "…Answer the operator's question concisely and accurately." (no grounding) |
| User turn | `Context:\n{ctx}\n\nQuestion: {q}` | `deep`: same. `fast`: the bare question |
| `max_new_tokens` | 150 | `deep`: 200. `fast`: 150 |
| Answer string | `out[0]["generated_text"][-1]["content"].strip()` | same |

What a proxy is (`proxy_pointer_indexer._proxy_text`, as the code actually does it): it is the
`heading_title`, plus `"\n"` and the first body line that is non-empty and does not start with
`|` or `#`, cut to 200 chars. If no such line exists (the body is all table rows or empty), the
proxy is the heading alone. On the stored index: 265 proxies are heading + line, 34 are heading
only because the body is table-only, and 17 are heading only because the body is empty. Proxy
length has median 79 chars and max 239 (none over 300). Re-embedding these code-derived proxies
reproduces all 316 stored index vectors (cosine 1.0000). Embedding the heading alone does not
(median cosine 0.72). The module docstring ("embeds every content node's heading_title") is
stale. The README's "heading + first meaningful body line, under 300 chars" is right in effect,
but the real cap is 200 chars on the body line.

A node's `body_content` is only the lines between its heading and the **next heading of any
level** (`document_tree_builder.add_file`). The class comment says "equal-or-higher level",
which is wrong. So the "whole section" is the node's own text plus its direct children's own
text. Grandchildren are not included. For 26 of the 30 benchmark questions the top-1 node has
no children.

### Score: `vanilla_rag_score`, `hybrid_score`

```python
vecs = SentenceTransformer("all-MiniLM-L6-v2").encode([reference_answer, generated_answer],
                                                      normalize_embeddings=True)
score = round(float(np.dot(vecs[0], vecs[1])), 4)
```

- The two strings are the `reference_answer` from `benchmark_questions.jsonl`, verbatim, and the
  stripped generated answer, verbatim. The question and the retrieved context are not involved.
- Normalization: L2-normalized vectors, so the dot product equals the cosine. The text itself is
  not normalized: no lowercasing, no stripping of boilerplate.
- MiniLM truncates input at 256 word-pieces. No Week 2 answer or reference goes over that.
- Each score is rounded to 4 dp per question before the means are taken.
- Published aggregates: `statistics.mean` over the 30 rounded scores, and per category over 10.
  The README's Δ column is hybrid mean − vanilla mean.

### `retrieval_correct_vanilla`, `retrieval_correct_hybrid`

```python
def retrieval_correct_auto(question_id, generated_answer):
    keywords = RETRIEVAL_KEYWORDS.get(question_id, [])
    matched = sum(1 for kw in keywords if kw.lower() in generated_answer.lower())
    return matched >= max(1, len(keywords) // 2)
```

- `RETRIEVAL_KEYWORDS` is a hand-written table in `run_day10_benchmark.py` with 2–7 keywords per
  question. The comment says they were "extracted from each reference answer". No code does
  that extraction, so they were picked by hand. 29 of the 30 lists appear verbatim in their
  reference answer. Q24's `"colour"` does not, because the reference spells it "color". Every
  reference answer passes its own rule (30/30).
- Threshold: at least `len(keywords)//2` matches, with a minimum of 1. In practice that is
  1 of 2, 1 of 3, 2 of 4, 2 of 5 and 3 of 7. Counts: fifteen lists need 2/4, seven need 1/3,
  four need 2/5, three need 1/2 and one needs 3/7.
- The match is a case-insensitive **substring** test with no word boundaries. For example, `text`
  matches "context", `value` matches "values" and `scroll` matches "scrolling".
- **It is applied to the generated answer, not to the retrieved context.** See weakness 1.
- Published figure: `sum(booleans)` out of 30, and that count / 30 as a percentage.

### Latency: `vanilla_rag_latency`, `hybrid_latency`

```python
t0 = time.time(); v_ans = vanilla_answer(text); v_lat = round(time.time() - t0, 2)
t0 = time.time(); h_ans = hybrid_answer(text);  h_lat = round(time.time() - t0, 2)
```

- This is wall-clock seconds for one run per question on CPU in float32. There is no warm-up, no
  repetition, and A always runs right before B.
- Included: lazy model and index loading on first use, question embedding, FAISS search,
  routing, node dereference, prompt building and **generation**. Excluded: the cosine scoring.
- The first call to A loads `faiss.index` + `chunks.pkl`, a MiniLM instance and TinyLlama. The
  first `deep` call to B loads `pp_faiss.index` / `pp_metadata.pkl` / `pp_tree.pkl` and a
  **second** MiniLM instance. Both of these happen on Q1, which routes `deep`.
- Published figure: the mean of the 30 values, shown to 1 dp.

### Published Week 2 numbers and reproduction status

| Published (README) | Re-derived from stored CSV | Status |
|---|---|---|
| Mean cosine 0.6985 / 0.6810 | 0.6985 / 0.6810 | reproduced |
| tabular 0.6625 / 0.6608 (Δ −0.002) | 0.6625 / 0.6608 (−0.0017) | reproduced |
| procedure 0.6169 / 0.6487 (Δ +0.032) | 0.6169 / 0.6487 (+0.0318) | reproduced |
| navigation 0.8160 / 0.7334 (Δ −0.083) | 0.8160 / 0.7334 (−0.0826) | reproduced |
| Retrieval correct 20/30 / 14/30 | 20 / 14 | reproduced |
| Mean latency 22.2 s / 26.7 s | 22.22 / 26.69 | reproduced |
| Q17 +0.42, Q3 +0.30 | +0.4159, +0.2984 | reproduced |
| Per-row cosines in the CSV | re-embedded the stored answer and reference with MiniLM: max \|Δ\| 4.7e-5 (vanilla), 4.9e-5 (hybrid), 0/60 over 1e-3 | reproduced |
| Per-row `retrieval_correct_*` booleans | re-applied `retrieval_correct_auto` to the stored answers: 0/60 mismatches | reproduced |
| `results/week2-proxy-vs-vanilla-results.csv` | identical to the numeric columns of the private full CSV | reproduced |

The latency and the generated answers themselves cannot be re-checked without re-running
generation, which was not done.

---

## Week 1 benchmark (40 prompts)

### Pipeline

| | "Traditional RAG" (`rag_score`) | "Fine-tuned SLM" (`finetuned_score`) |
|---|---|---|
| Retrieval | `pypdf` chunks (500/50), MiniLM, `IndexFlatIP`, top-4, **no context cap** | none |
| Generator | `TinyLlama-1.1B-Chat-v1.0`, `bfloat16`, `device_map="auto"`, greedy, `max_new_tokens=256` | `google/gemma-3-1b-it` + `serac-gemma-adapter` LoRA, float32 on MPS, greedy, `max_new_tokens=256` (`run_finetuned.py`) |
| Prompt | one user turn: grounding instruction + `Context:` + `Question:` (no system turn) | the bare instruction, through the Gemma chat template |
| Produced by | rows 21–40: `rag_new_questions.py`. Rows 1–20: **no committed script matches** (see weakness 11) | `run_finetuned.py` → `finetuned_outputs.csv` |

### Score (`score.py`)

- The two strings are the `reference_answer` from `evaluation_prompts.jsonl` and each system's
  answer. RAG answers are joined to prompts by exact question text. Fine-tuned answers are joined
  by `id`.
- The vectors come from `SentenceTransformer("all-MiniLM-L6-v2").encode([...])` **without**
  normalization. Cosine is then computed explicitly as `a·b / (‖a‖‖b‖)`, which is
  mathematically the same as the Week 2 computation. No rounding.
- The README's 0.5079 / 0.4895 is the plain mean over 40. "22/40 wins" counts rows with
  `finetuned_score > rag_score`. The other 18 went to RAG, with no ties.
- The `score.py` docstring says 20 questions. The data has 40. Ids 21–40 were added later
  (`rag_new_questions.py`).

### Published Week 1 numbers and reproduction status

| Published | Re-derived | Status |
|---|---|---|
| Fine-tune mean 0.5079 | 0.5079 | reproduced |
| RAG mean 0.4895 | 0.4895 | reproduced |
| Δ +0.0185, 22/40 vs 18/40 wins, 0 ties | +0.0185, 22 / 18 / 0 | reproduced |
| Per-row scores in `comparison.csv` | re-ran `score.py` logic on the stored answers: max \|Δ\| 8e-17, 0/80 over 1e-3 | reproduced |
| `comparison.csv` answers vs input CSVs | 40/40 RAG and 40/40 fine-tune answers identical to `rag_baseline_outputs.csv` / `finetuned_outputs.csv`. All 40 references identical to `evaluation_prompts.jsonl` | consistent |

---

## Where the reference answers come from

**Nothing documents it.** The reference answers live only in `benchmark_questions.jsonl`
(Week 2) and `evaluation_prompts.jsonl` (Week 1). No script in the repo creates either file, and
neither `docs/handover-weeks-1-3.md` nor any `docs/week*.md` says who wrote them or how. The
handover lists both files next to `train.jsonl`, which the README describes as
"instruction/output pairs synthesized from the same corpus". There is no record of authorship,
of whether an LLM wrote them, or of any check by someone who knows the machines. Treat them as
unverified author-written (possibly LLM-assisted) answers derived from the manuals.

Properties that can be measured:

- Mean reference length: 264 chars (Week 2) and 147 chars (Week 1).
- Week 1 prompts 1–20 are the same 20 alarm questions hard-coded in `run_rag_baseline.py`.
  None of them appears verbatim in `train.jsonl`. However, **15 of the 20 have a training
  instruction with MiniLM cosine ≥ 0.9** (prompts 21–40: 1 of 20). See weakness 10.
- Week 2 questions: 0 of 30 have a training instruction with cosine ≥ 0.9.

---

## Known weaknesses / confounds

Every number below was measured on the stored run outputs and the stored indices.

1. **`retrieval_correct` is a generation metric with a retrieval label.** `retrieval_correct_auto`
   receives `v_ans` / `h_ans`, the generated answers. The retrieved chunks or node are never
   passed to it. It is true when the generator's text contains enough hand-picked keywords.
   Other weaknesses with this metric:
   - A system that retrieves nothing can score on it. 8 of the 14 hybrid "correct" answers came
     from `fast`-route questions, where nothing was retrieved.
   - For 11 questions (7, 8, 10, 22, 23, 25, 26, 27, 28, 29, 30), the question text alone already
     contains enough keywords to pass. An answer that just repeats the question counts as
     "correct".
   - Substring matching gives false hits, for example `text` inside "context". Switching to
     word-boundary matching would change Q8 on both sides, giving 19/30 vs 13/30.

2. **Cosine(answer, reference) mixes retrieval quality with generator behavior.** It rewards
   wording and length close to the reference whether or not the right passage was retrieved. It
   also cannot tell a correct answer from a fluent wrong one that uses the same vocabulary.
   Generated-answer length correlates with score at r = −0.42 (vanilla) and −0.18 (hybrid).
   TinyLlama makes this worse: the stored answers include restated system prompts and generic
   text unrelated to the manuals. Per-question deltas are large relative to the gaps between
   means: the hybrid − vanilla delta has SD 0.175. A Wilcoxon signed-rank test gives p = 0.66
   over all 30 questions, and p = 0.63 / 0.77 / 0.38 for procedure / tabular / navigation (n = 10
   each). **None of the Week 2 differences, overall or per category, can be told apart from
   noise.** Week 1 is the same: Wilcoxon p = 0.53, sign test on 22/40 p = 0.64.

3. **The "Proxy-Pointer" column is the router, and most questions skip proxy-pointer.** With
   `classify_query`, **18 of the 30 questions route `fast`**: they get no retrieval and no
   context, only bare TinyLlama with a non-grounded prompt. The fast ids are 2, 9, 10, 14, 15,
   17–25 and 27–30. By category, 3 of 10 tabular, 6 of 10 procedure and 9 of 10 navigation
   questions go `fast`. Only 12 questions (1, 3–8, 11–13, 16, 26) use proxy-pointer retrieval,
   5 of them on a single regex match (`borderline`: 3, 6, 7, 8, 26). The patterns miss common
   phrasings: "how do you", "which taskbar icon" and "what information" all fall through to
   `fast`. The published hybrid scores split by route:

   | Subset | n | Vanilla | Hybrid | Δ | Correct V / H |
   |---|---|---|---|---|---|
   | `fast` (no retrieval) | 18 | 0.7183 | 0.6954 | −0.0229 | 11 / 8 |
   | `deep` (proxy-pointer) | 12 | 0.6687 | 0.6594 | −0.0094 | 9 / 6 |
   | procedure · fast | 6 | 0.5453 | 0.6343 | **+0.0890** | |
   | procedure · deep | 4 | 0.7243 | 0.6704 | **−0.0539** | |
   | tabular · fast | 3 | 0.7292 | 0.7025 | −0.0267 | |
   | tabular · deep | 7 | 0.6339 | 0.6429 | +0.0090 | |
   | navigation · fast | 9 | 0.8300 | 0.7338 | −0.0962 | |
   | navigation · deep | 1 | 0.6900 | 0.7302 | +0.0402 | |

   The whole "procedure" gain comes from questions where **no proxy-pointer retrieval
   happened**. On the 4 procedure questions that did use proxy-pointer, it lost to vanilla. Q17,
   the headline +0.42, routed `fast`. Q21 (+0.16) did too, and its stored hybrid answer is
   generic text about desktop operating systems. Of the five largest hybrid wins, only Q3 used
   proxy-pointer. The navigation loss is bare TinyLlama vs RAG on 9 of 10 questions, not "the
   proxy missing a term". There is no fine-tuned model anywhere in this benchmark. The docstring says `fast` means
   "fine-tuned Gemma", but `fast` uses the same TinyLlama pipeline. `_USE_ADAPTER` is defined and
   never read.

4. **"Returns the whole section" is cut to 1200 chars.** Both arms apply `context[:1200]` before
   generation. Running `context_assembler.query` on all 30 questions with the stored `pp_*`
   files gives:
   - On the 12 `deep` questions, where the context is used, **5 of 12 (42%) assembled contexts
     are longer than 1200 chars** and get cut. They are Q3 (12,572 chars, 9.5% kept), Q26 (5,778,
     21%), Q12 (2,140, 56%), Q13 (2,104, 57%) and Q7 (1,417, 85%).
   - Q3 is the headline "table intact" win, and only its first 9.5% reached the model.
   - Two `deep` contexts are almost empty: Q1 (83 chars) and Q11 (167 chars).
   - Across all 30 questions (as if every one routed `deep`), 16 of 30 go over 1200.
   - Vanilla's 4-chunk context goes over 1200 on **30 of 30** questions (median 1,870 chars), so
     vanilla in practice gets about the first 2.5 chunks (65% kept on average).

5. **Different source text.** Vanilla indexes `pypdf` raw text of the PDFs: 0 of 799 chunks
   contain a `|` pipe table or a `#` heading. Proxy-pointer indexes the LlamaParse markdown
   (premium mode, with a table-preserving parsing instruction). So the comparison mixes PDF
   extraction quality with retrieval method. Some of any proxy-pointer gain could come from
   cleaner text rather than from the proxy/pointer design, and the stored vanilla answers do
   reproduce page-header noise from the raw PDF text.

6. **Generation settings differ between and within arms.** `max_new_tokens` is 150 for vanilla,
   200 for `deep` and 150 for `fast`. System prompts differ, and `fast` has no grounding
   instruction. Week 1 differs again: no context cap, 256 tokens, no system turn, bf16. "Same
   generator" holds only in the narrow sense of the same TinyLlama weights.

7. **The proxy definition is documented three different ways.** The indexer docstring says
   heading only. The README says "heading + first meaningful body line, under 300 chars". The
   code does heading + first body line not starting with `|`/`#`, cut to 200 chars, falling back
   to the heading alone (51 of 316 nodes). The stored index matches the code (see the Week 2
   pipeline section above).

8. **Latency includes one-time model load and is dominated by generation length.**
   - Q1 vanilla took 57.09 s against a vanilla median of 21.44 s. That Q1 includes loading the
     vanilla index, MiniLM and TinyLlama, about 35.7 s of one-time load, which **adds ≈ 1.2 s to
     the vanilla mean**. Without Q1, vanilla is 21.02 s and hybrid is 27.20 s, so the published
     4.5 s gap becomes 6.2 s.
   - Hybrid Q1 (11.76 s) includes loading the `pp_*` files and a second MiniLM. Q1 is short
     because its retrieved context was 83 chars.
   - The hybrid's extra time comes from the **`fast` route, not the proxy-pointer route**.
     Without Q1: `fast` averages 30.79 s for hybrid vs 19.83 s for vanilla on the same
     questions. `deep` averages 21.33 s vs 22.97 s, so proxy-pointer was *faster* than vanilla.
     Unguided `fast` answers are about twice as long (385 vs 196 chars).
   - Latency correlates with answer length at r = 0.70 (vanilla) and 0.74 (hybrid).
   - These are single uncontrolled runs on a shared CPU. They should not be read as retrieval
     cost. Retrieval itself (one MiniLM query embedding plus an exact search over fewer than 800 vectors)
     takes tens of milliseconds.

9. **The fine-tune arm in Week 1 did not use TinyLlama.** The README ("every score above was
   generated with TinyLlama") and `docs/week1-comparison-results.md` both say otherwise.
   `run_finetuned.py` has no TinyLlama path: it loads `google/gemma-3-1b-it` plus the LoRA
   adapter, and a Gemma adapter cannot attach to TinyLlama. The stored outputs fit this. 31 of 40
   fine-tune answers use Markdown bold, against 0 of 40 TinyLlama RAG answers, and they are about
   twice as long (968 vs 491 chars). So Week 1 compares two different generators *and* retrieval
   vs no retrieval at once. It cannot isolate either factor.

10. **Week 1 scoring has two further biases.**
    - (a) MiniLM truncates at 256 word-pieces. **27 of 40 fine-tune answers are longer than
      that**, so only their opening is scored. No RAG answer or reference is over the limit.
    - (b) Train/eval near-duplicates: 15 of eval prompts 1–20 have a training instruction with
      cosine ≥ 0.9. The fine-tune's lead is +0.027 on prompts 1–20 and +0.010 on prompts 21–40.
      Part of its small advantage may therefore come from questions close to its training data.

11. **The Week 1 RAG rows 1–20 have no matching committed script.** `run_rag_baseline.py` asks
    the same 20 questions, but it writes columns `question_id, question, retrieved_context,
    rag_answer`, with each chunk cut to 200 chars. The stored file has `question,
    retrieved_chunks, generated_answer`, with full contexts averaging about 1,824 chars, which is
    what `rag_new_questions.py` writes for rows 21–40. So the exact settings for rows 1–20 are not
    recorded in code.

12. **No repetition and no variance reported.** Each number comes from one greedy run. With
    n = 10 per category and the per-question spread above, the category deltas (+0.032, −0.002,
    −0.083) are within noise.

13. **Small inaccuracies in the docs.**
    - `verify_linkage()` prints table/step/non-empty checks for 3 random nodes. It does not
      assert anything, contrary to the README and the handover.
    - The auto-generated report conclusion (`results/week2-proxy-vs-vanilla-table.md`, from
      `run_day10_benchmark.write_report`) says correctness "improved from 67% (Vanilla) to 47%".
      That is a drop.
    - The same conclusion says the hybrid wins on table-heavy questions, but tabular Δ is −0.0017.
    - It attributes the latency gap to the deep route's "two-step embed-then-fetch". The deep
      route was faster (weakness 8).

### What the published numbers support

- On these 30 questions, scored this way, vanilla RAG and the router are statistically
  indistinguishable on the answer-similarity score.
- On the 12 questions that actually used proxy-pointer retrieval, it scored −0.009 against
  vanilla, which is also noise.
- The per-category story in the README (proxy-pointer wins on procedures, loses on navigation
  because of the proxy) is not supported. The procedure gain and the navigation loss both come
  from `fast`-route questions that used no proxy-pointer retrieval.

---

## Re-running the verification

Against a private copy of the data, with the run outputs and indices in the working directory:

1. Week 2 aggregates: load `proxy_rag_vs_vanilla_rag_results.csv` and take `statistics.mean` of
   `vanilla_rag_score` / `hybrid_score` (overall and per `category`), the sums of
   `retrieval_correct_*` and the means of `*_latency`.
2. Week 2 per-row scores: `from run_day10_benchmark import cosine_score,
   retrieval_correct_auto`. For each row, compare `cosine_score(reference_answer, answer)` with
   the stored score (tolerance 1e-3), and `retrieval_correct_auto(id, answer)` with the stored
   boolean.
3. Route split: `from hybrid_intent_router import classify_query` (pure regex) on each question.
4. Context lengths: `context_assembler.query(question)` with `pp_faiss.index`,
   `pp_metadata.pkl` and `pp_tree.pkl` in cwd. For vanilla,
   `naive_rag_baseline.retrieve(..., k=4)` joined with `"\n\n---\n\n"`. Compare `len()` with 1200.
5. Week 1: the `score.py` logic on `evaluation_prompts.jsonl`, `rag_baseline_outputs.csv` and
   `finetuned_outputs.csv`. Compare with `comparison.csv`, then take the means and count
   `finetuned_score > rag_score`.

None of these steps needs an LLM. Set `HF_HUB_OFFLINE=1` once MiniLM is cached.
