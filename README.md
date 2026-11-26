# Proxy-Pointer RAG

Structure-aware retrieval for technical documentation, benchmarked head-to-head against
fixed-size-chunk RAG — and a QLoRA fine-tune of the same corpus for comparison.

Fixed-size chunking shreds structured content. A 500-character window cuts a pipe table in
half and splits a numbered procedure across two chunks, so the model receives fragments where
it needed an intact section. This repo indexes a short **proxy** for every heading node in the
document tree, and on a hit **dereferences a pointer** back to that node's full body — so
retrieval returns a complete section, tables included, instead of a 500-character window.

## Results

30 categorized questions over two industrial equipment manuals. `all-MiniLM-L6-v2` embeddings,
FAISS `IndexFlatIP`, same TinyLlama-1.1B weights and same question set on both sides (prompts,
context cap and `max_new_tokens` differ). The right-hand column is the **router**: 18/30
questions routed `fast` and got no retrieval at all. What every number measures, and its known
confounds: [`docs/metrics.md`](docs/metrics.md).

| Metric | Vanilla RAG | Proxy-Pointer + router |
|---|---|---|
| Mean cosine | **0.6985** | 0.6810 |
| Keyword match in generated answer¹ | **20/30 (67%)** | 14/30 (47%) |
| Mean latency | **22.2 s** | 26.7 s |

¹ Despite its old name, this checks the *generated answer*, not the retrieved context — it is not
a retrieval metric.

Per category (n = 10 each):

| Category | Vanilla | Proxy-Pointer | Δ |
|---|---|---|---|
| procedure | 0.6169 | **0.6487** | **+0.032** |
| tabular | 0.6625 | 0.6608 | −0.002 |
| navigation | **0.8160** | 0.7334 | −0.083 |

**No difference here is statistically significant** (Wilcoxon p = 0.66 overall; 0.63 / 0.77 /
0.38 per category). An earlier version of this README read the procedure row as a
proxy-pointer win. It isn't one: the procedure gain comes from `fast`-route questions that used
no proxy-pointer retrieval (Q17, the +0.42 headline, included). On the 12 questions that actually
used proxy-pointer retrieval it scored 0.6594 vs vanilla's 0.6687. Q3 (+0.30) did use it, but only
the first 1,200 of 12,572 assembled characters reached the generator.

So this benchmark does **not** establish that proxy-pointer retrieval is better for procedures
and tables. It cannot: the score is answer-similarity, which mixes retrieval with generation,
and the hybrid column mixes retrieval with routing. Direct retrieval metrics are the fix — see
[`docs/metrics.md`](docs/metrics.md) and the evaluation plan.

Full numbers: [`results/week2-proxy-vs-vanilla-results.csv`](results/week2-proxy-vs-vanilla-results.csv)
· analysis: [`docs/week2-benchmark-report.md`](docs/week2-benchmark-report.md)

## How it works

![Proxy-pointer architecture](docs/diagrams/week2_proxy_pointer_architecture.png)

1. **`parse_documents.py`** — PDFs to heading-structured markdown.
2. **`document_tree_builder.py`** — builds a `DocumentTree` of `DocumentNode`s, each carrying
   `heading_title`, `body_content`, `parent_node_id`, `start_line`/`end_line`.
3. **`proxy_pointer_indexer.py`** — embeds a proxy per content node (heading + first body line not
   starting with `|` or `#`, that line cut to 200 chars) and stores the pointer: `pointer_node_id`, `doc_id`, line span.
   `ROOT` and `[FILE]` container nodes are excluded from the index; table-only nodes fall back
   to their heading. `verify_linkage()` round-trips random proxies back to their nodes and
   prints whether each pointer still resolves to non-empty body content.
4. **`context_assembler.py`** — dereferences the top-1 hit to the node's full body plus its
   direct children. The benchmark truncates this context to 1,200 chars before generation.
5. **`hybrid_intent_router.py`** — classifies a query fast vs deep on a regex trigger table
   (`table`, `parameters`, `procedure`, `steps`, `how do i`, `list`, …). Zero triggers →
   fast/certain; exactly one → deep/borderline; two or more → deep/certain, biasing ambiguous
   structured questions toward the retriever.

The baseline it is measured against is `naive_rag_baseline.py`: `RecursiveCharacterTextSplitter`
at `chunk_size=500 / overlap=50`, top-k 4, stuffed into a grounded prompt.

## The fine-tuning arm

`finetune.py` LoRA-tunes `google/gemma-3-1b-it` on instruction/output pairs synthesized from
the same corpus. Its CUDA path is QLoRA (4-bit NF4, bf16 compute); LoRA `r=8` / `alpha=16` on `q_proj` and `v_proj`,
dropout 0.05, 3 epochs at `lr=2e-4`, max length 512. It is MPS-aware: it attempts local
training on Apple Silicon and, on the expected MPS kernel failure, auto-generates a Colab
notebook targeting a free T4, because the 4-bit stack is CUDA-bound.

Re-run Gemma-vs-Gemma ([`run_gemma_arms.py`](run_gemma_arms.py),
[`docs/finetune-gemma-results.md`](docs/finetune-gemma-results.md)), over the same 40 evaluation
questions:

| Arm (all `gemma-3-1b-it`, greedy, 256 new tokens) | Mean cosine | Median |
|---|---|---|
| base | 0.498 | 0.508 |
| base + adapter | **0.520** | 0.549 |
| flat RAG (top-4) + base | 0.464 | 0.492 |

**None of these gaps is significant at n = 40.** Adapter − base is +0.022, 95% CI [−0.009, +0.052];
adapter − RAG is +0.056 [−0.012, +0.125]. The adapter's lead is concentrated on evaluation
questions that paraphrase its training data (ids 1–20). On the 15 items with no near-duplicate in
the training set, it scores slightly *below* the base model. The original Week 1 figures (0.5079 vs
0.4895, 22/40 wins) compared Gemma + adapter against a **TinyLlama**-generated RAG arm, and their CI
also crossed zero. Read this as "no measurable difference yet; the eval set needs de-leaking",
not as a win for fine-tuning.

The shipped adapter's `training_args.bin` records an MPS device with bf16 off. 4-bit
bitsandbytes doesn't run on MPS, so the adapter was most likely trained with `finetune.py`'s
non-quantized LoRA path, not 4-bit QLoRA.

## Honest caveats

- **Week 2 still uses a stand-in generator.** The Week 2 proxy-pointer vs vanilla benchmark was
  generated with `TinyLlama-1.1B-Chat`, so those scores measure retrieval and routing mixed with a
  weak generator, not answer quality. The Week 1 fine-tuning comparison has been re-run
  Gemma-vs-Gemma (above).
- **The fine-tune eval set overlaps the SFT data.** Eval ids 1–20 are paraphrases of training rows
  0–19 (15/20 have instruction cosine > 0.9), which inflates any fine-tuned arm. See the leakage
  section of [`docs/finetune-gemma-results.md`](docs/finetune-gemma-results.md).
- **No training curve was captured.** The LoRA adapter exists as proof training ran, but no
  loss trace or notebook output was saved. See `docs/week1-training-evidence.md`.
- **Week 1 recorded no latency.** Only Week 2 has per-question timings. Not estimated.
- **The corpus is not in this repo.** The source manuals are a third party's proprietary
  documentation, so the PDFs, the markdown conversions, the SFT dataset, the FAISS indices and
  the per-question generated answers are all excluded. The pipeline is corpus-agnostic — point
  `parse_documents.py` at your own PDFs and the rest runs.

## Running it on your own corpus

```bash
pip install sentence-transformers faiss-cpu pypdf langchain-text-splitters transformers \
            peft trl bitsandbytes

python parse_documents.py          # your PDFs -> heading-structured markdown
python proxy_pointer_indexer.py    # build proxy index + pointer metadata + tree
python run_day10_benchmark.py      # vanilla vs proxy-pointer, per category
python score.py                    # cosine scoring of answers vs references
```

`run_rag_baseline.py` runs the flat-chunk baseline alone; `finetune.py` trains the adapter.
