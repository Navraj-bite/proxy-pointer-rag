# Proxy-Pointer RAG

Structure-aware retrieval for technical documentation, benchmarked head-to-head against
fixed-size-chunk RAG — and a QLoRA fine-tune of the same corpus for comparison.

Fixed-size chunking shreds structured content. A 500-character window cuts a pipe table in
half and splits a numbered procedure across two chunks, so the model receives fragments where
it needed an intact section. This repo indexes a short **proxy** for every heading node in the
document tree, and on a hit **dereferences a pointer** back to that node's full body — so
retrieval returns a complete section, tables included, instead of a 500-character window.

## Results

### Retrieval (primary metric)

30 categorized questions over two industrial equipment manuals (private corpus — these numbers
cannot yet be regenerated from a clean clone; a public-corpus reproduction is in progress).
`all-MiniLM-L6-v2`, FAISS `IndexFlatIP`, no router, no generator. Each system returns a ranked
list of sections, scored against gold section labels, which are **agent-drafted and not yet
human-verified**. Method: [`docs/retrieval-metrics.md`](docs/retrieval-metrics.md).

| Metric (section-level, strict) | Flat 500/50 chunks (same markdown) | Proxy-pointer |
|---|---|---|
| Recall@1 | **0.60** | 0.50 |
| Recall@3 | **0.83** | 0.63 |
| Recall@5 | **0.87** | 0.70 |
| MRR | **0.72** | 0.57 |
| nDCG@5 | **0.72** | 0.57 |

Proxy − flat, paired bootstrap 95% CI (n = 30): MRR −0.15 [−0.28, −0.03], Recall@5 −0.17
[−0.30, −0.03]. **Measured directly, proxy-pointer retrieval is worse than flat chunking on this
corpus.** The gap is mostly tables (tabular Recall@5 0.50 vs 0.90, n = 10). The proxy is the heading
plus the first body line, so it can't see the rows of a large table. On procedures the two tie
under the strict rule. Counting what the assembler actually hands the generator (node + direct
children), proxy-pointer reaches procedure Recall@3 = 1.0 vs 0.8. One exploratory run with a
heading-only proxy closed most of the MRR gap (0.70); the proxy-size sweep is planned.

### Answer similarity (secondary, legacy)

The original benchmark, kept for continuity. TinyLlama generator; both systems' context cut to
1,200 chars before generation.

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

### Latency (retriever only)

Measured stage by stage on an Apple M4 (CPU, MiniLM query embedding + FAISS `IndexFlatIP`),
models preloaded, 3 repeats per question. Details: [`docs/latency-memory.md`](docs/latency-memory.md).

| Retrieval, end to end (p50 / p95) | Flat 500/50, top-4 | Proxy-pointer, top-1 + children |
|---|---|---|
| Public corpus (252 queries) | 3.6 / 6.1 ms | 3.5 / 5.7 ms |
| Private corpus (30 queries) | 3.7 / 5.0 ms | 3.7 / 5.3 ms |

Query embedding is 99% of that; search, pointer dereference and context assembly together take
< 0.05 ms. TinyLlama generation on the same machine takes 11–13 s per answer, set mainly by
answer length. **The 22.2 s / 26.7 s figures above measure the generator and the router's fast
route (which writes longer answers), not retrieval.** GPU memory (VRAM) on a Colab T4 is not yet
measured: [`colab/measure_vram.py`](colab/measure_vram.py) is ready to run.

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

`finetune.py` QLoRA-tunes `google/gemma-3-1b-it` on instruction/output pairs synthesized from
the same corpus — 4-bit NF4, bf16 compute, LoRA `r=8` / `alpha=16` on `q_proj` and `v_proj`,
dropout 0.05, 3 epochs at `lr=2e-4`, max length 512. It is MPS-aware: it attempts local
training on Apple Silicon and, on the expected MPS kernel failure, auto-generates a Colab
notebook targeting a free T4, because the 4-bit stack is CUDA-bound.

Over 40 evaluation questions the fine-tune scored a mean cosine of 0.5079 against RAG's 0.4895
(22/40 head-to-head wins) — not significant (sign test p = 0.64). That arm used Gemma-3-1b +
adapter with no retrieval, against TinyLlama + RAG, so it isn't a like-for-like comparison. 27/40
fine-tune answers exceed MiniLM's 256-token input limit (only their opening is scored), and 15 of
the first 20 evaluation prompts have near-duplicates in the training set.

## Honest caveats

- **The generator is a stand-in.** `gemma-3-1b-it` is gated on Hugging Face and the evaluation
  environment had no HF auth, so every RAG/router answer was generated with `TinyLlama-1.1B-Chat`; the Week 1
  fine-tune arm used Gemma-3-1b + LoRA.
  The retrieval and routing logic is what's being measured; absolute answer quality is not.
  Swap `_GEN_MODEL` / `_USE_ADAPTER` in `hybrid_intent_router.py` once you have auth.
- **No training curve was captured.** The LoRA adapter exists as proof training ran, but no
  loss trace or notebook output was saved. See `docs/week1-training-evidence.md`.
- **Week 1 recorded no latency.** Only Week 2 has per-question timings. Not estimated.
- **The corpus is not in this repo.** The source manuals are a third party's proprietary
  documentation, so the PDFs, the markdown conversions, the SFT dataset, the FAISS indices and
  the per-question generated answers are all excluded. The pipeline is corpus-agnostic — point
  `parse_documents.py` at your own PDFs and the rest runs.

## Reproduce on a public corpus

The private-corpus numbers above can't be regenerated from a clone. To check the pipeline
independently, run it on a pinned public corpus: a subset of the
[PX4 User Guide](https://github.com/PX4/PX4-user_guide) (CC BY 4.0, tag `v1.17.0`, commit
`d6f12ad`). It has calibration procedures, parameter tables, status tables and
troubleshooting, about 0.5 MB of markdown. The fetch script checks every file against a
committed sha256 manifest. A deterministic, stdlib-only converter then turns it into
heading-structured markdown. No API key is needed.

```bash
pip install sentence-transformers faiss-cpu langchain-text-splitters pypdf
./scripts/fetch_corpus.sh                       # download, verify, convert -> corpus/
export PPRAG_CORPUS=public
python3 check_corpus_tree.py --questions questions/public_seed.jsonl
python3 proxy_pointer_indexer.py
python3 naive_rag_baseline.py --build-only      # flat baseline over the same markdown
python3 run_day10_benchmark.py --retrieval-only # or drop the flag to generate answers
```

`PPRAG_CORPUS` defaults to `private` (the original setup). Public indices and outputs go to
`artifacts/public/`. On macOS, set `OMP_NUM_THREADS=1` for generation runs (faiss/torch
import-order segfault). Licence, conversion rules, tree statistics and the question/gold format
are in [`docs/corpus.md`](docs/corpus.md).

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
