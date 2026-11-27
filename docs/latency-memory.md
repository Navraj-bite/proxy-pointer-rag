# Latency, index cost and memory

The published latencies (vanilla RAG 22.2 s, proxy-pointer + router 26.7 s mean) are
wall-clock times around a whole answer call. They include loading the models on the first
question and TinyLlama generation, so they mostly measure the generator. This page measures
the retriever on its own, stage by stage, then generation separately with identical settings
for both systems, plus index build cost, on-disk size and peak memory.

**Headline.** Retrieval overhead is about **3.5 ms p50 / 6 ms p95 per query** on the M4's CPU,
for proxy-pointer and flat-chunk alike, on both corpora. That is independent of whichever
generator is bolted on. Embedding the query is 99% of it. FAISS search, the pointer
dereference and context assembly together take under 0.05 ms. Generating one answer with
TinyLlama on the same machine takes about 11 to 13 s (p50), roughly 3,000 times the
retrieval cost, and its time is set by how many tokens it generates.

Script: [`benchmark_latency.py`](../benchmark_latency.py). Public-corpus result files (aggregates
plus per-query timings, no corpus text): [`results/latency/`](../results/latency/).
Public-corpus numbers can be regenerated from a clean clone. **Private-corpus numbers can't:**
they come from the proprietary manuals, and only aggregates are reported here.

## Method

**Systems.** Both use the production code paths and the same on-disk indices the benchmarks use.

| | proxy-pointer | flat-chunk |
|---|---|---|
| Index | one `all-MiniLM-L6-v2` vector per heading node (heading + first body line), `IndexFlatIP` | one vector per `RecursiveCharacterTextSplitter(500, 50)` chunk, `IndexFlatIP` |
| embed | `SentenceTransformer.encode([q], normalize_embeddings=True)` | same call |
| search | FAISS top-1 | FAISS top-4 |
| dereference | `metadata[i]["pointer_node_id"]` → `tree.nodes[...]` | (none) |
| assemble | node `# heading` + body + each direct child's `## heading` + body (`context_assembler.query`) | look up the 4 chunks and join them with `"\n\n---\n\n"` (`naive_rag_baseline.retrieve` + `run_day10_benchmark`) |

The staged code is checked against the production functions on every run. For 25 questions,
the assembled strings are compared byte for byte with `context_assembler.query()` and
`naive_rag_baseline.retrieve()`, and the run aborts on any difference. Each stage is timed with
`time.perf_counter`. "Retrieval total" runs from the start of embed to the end of assemble.

**Protocol.**

- Models and indices are loaded once, before timing, and the load is timed separately.
- 20 untimed warm-up queries per system come first.
- Then every question runs 3 times per system, with question order reshuffled each repeat
  (seed 0) and the order of the two systems alternated.
- One MiniLM instance is shared by both systems. The production benchmark loads two, which
  costs memory but not per-query time.
- Python import time is excluded.

**Questions.**

- Public: 252 query strings. They are the union of `questions/public_seed.jsonl` (12) and a
  snapshot of the 252-question public set that is still being verified (task 4,
  `public_v1.jsonl`, sha256 `a0219aa4…c4ec`, which contains the 12 seed questions). Only the
  query strings are used, so verification status does not matter for timing.
  That gives 756 timed queries per system per configuration.
- Private: the 30 benchmark questions, so 90 timed queries per system.

**Configurations.** Three embedding setups:

- CPU with torch's default 4 threads.
- CPU with `OMP_NUM_THREADS=1` (torch then uses 1 thread). This is the setting the generation runs need.
- MPS. On a Mac, `SentenceTransformer(...)` with no `device` argument picks MPS, so this is what
  `context_assembler.py` and `naive_rag_baseline.py` actually use.

`torch` is imported before `faiss` (see the segfault note in [`corpus.md`](corpus.md)).

**Environment** (recorded in every result file):

| | |
|---|---|
| Machine | Apple M4 (4 performance + 6 efficiency cores), 16 GB unified memory, macOS 26.6.2, no CUDA |
| Threads | torch 4 intra-op / 10 inter-op threads by default, faiss OpenMP 10; with `OMP_NUM_THREADS=1`, torch 1 thread |
| Software | Python 3.9.6, torch 2.8.0, transformers 4.57.6, sentence-transformers 5.1.2, faiss-cpu 1.13.0, numpy 2.0.2, tokenizers 0.22.2, langchain-text-splitters 0.3.11, pypdf 6.13.1 |
| Models | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1`, loaded from the local cache (`HF_HUB_OFFLINE=1`) |
| Seed | 0 |
| Date | 2026-11-26 |
| Code | `benchmark_latency.py` as committed on branch `task-7-latency-memory`. The runs used the working tree on top of `9ac223e`, before the commit, so the files say `git_dirty: true`. The only changes since are the `dtype=` keyword and when `loadavg_start` is read. |
| Public corpus | PX4 User Guide, commit `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` ([`corpus.md`](corpus.md)) |

**The machine was not fully idle.** Other agents were doing light file reading and grep
during the runs. The 1-minute load average at the end of each retrieval and build run was
about 2.3 to 3.2. The p95 columns are the ones most affected.

**Timing windows and contention.** All retrieval and build timings ran between 05:48 and
05:59. That was before another agent's Gemma generation job (`run_gemma_arms.py`, MPS) started
at 06:03:21, so none of them overlapped it. After that job finished (06:21), the public CPU
(4-thread) retrieval run was repeated once as a check. Its retrieval total was 3.16 / 4.35 ms
(p50 / p95) for proxy-pointer and 3.17 / 4.35 ms for flat, against 3.55 / 5.75 and 3.57 / 6.13
in the tables below. So the published p50s are at most about 0.4 ms high, and the conclusion
does not change. The generation run is a separate case; its conditions are described in that
section.

## Retrieval latency, stage by stage

All values are milliseconds per query.

### Public corpus (PX4, reproducible): 252 questions × 3 repeats = 756 queries per system

| Stage | proxy-pointer p50 | p95 | flat-chunk p50 | p95 |
|---|---|---|---|---|
| **CPU, 4 threads** | | | | |
| embed query | 3.522 | 5.687 | 3.522 | 6.068 |
| FAISS search | 0.023 | 0.038 | 0.042 | 0.063 |
| dereference pointer | 0.002 | 0.004 | n/a | n/a |
| assemble context | 0.003 | 0.008 | 0.003 | 0.005 |
| **retrieval total** | **3.55** | **5.75** | **3.57** | **6.13** |
| **CPU, 1 thread (`OMP_NUM_THREADS=1`)** | | | | |
| embed query | 3.244 | 4.180 | 3.243 | 4.197 |
| FAISS search | 0.021 | 0.025 | 0.039 | 0.045 |
| dereference pointer | 0.002 | 0.002 | n/a | n/a |
| assemble context | 0.002 | 0.006 | 0.003 | 0.004 |
| **retrieval total** | **3.27** | **4.21** | **3.28** | **4.24** |
| **MPS (sentence-transformers' default on a Mac)** | | | | |
| embed query | 6.596 | 8.233 | 6.617 | 7.817 |
| FAISS search | 0.028 | 0.049 | 0.052 | 0.081 |
| dereference pointer | 0.002 | 0.003 | n/a | n/a |
| assemble context | 0.002 | 0.008 | 0.003 | 0.005 |
| **retrieval total** | **6.63** | **8.28** | **6.68** | **7.88** |

Means (CPU, 4 threads): proxy-pointer 3.79 ms, flat 3.83 ms. Embedding is 99.1% of
proxy-pointer retrieval time and 98.7% of flat.

### Private corpus (two equipment manuals, not reproducible): 30 questions × 3 repeats = 90 queries per system

| Stage | proxy-pointer p50 | p95 | flat-chunk p50 | p95 |
|---|---|---|---|---|
| **CPU, 4 threads** | | | | |
| embed query | 3.628 | 5.219 | 3.612 | 4.921 |
| FAISS search | 0.024 | 0.032 | 0.037 | 0.081 |
| dereference pointer | 0.003 | 0.004 | n/a | n/a |
| assemble context | 0.003 | 0.006 | 0.004 | 0.006 |
| **retrieval total** | **3.66** | **5.25** | **3.66** | **4.97** |
| **CPU, 1 thread** | | | | |
| **retrieval total** | **2.99** | **3.29** | **3.00** | **3.24** |
| **MPS** | | | | |
| **retrieval total** | **7.19** | **10.79** | **7.35** | **10.96** |

The full private stage breakdown is in the private run directory. As on the public corpus,
search + dereference + assemble add up to under 0.05 ms in every configuration.

### What this says

- **Retrieval overhead is about 3 to 4 ms p50 (under 6.2 ms p95) on CPU, and 7 ms on MPS, for
  both systems.** The two systems differ by less than the run-to-run noise. Proxy-pointer's
  extra step, the pointer dereference, costs about 2 µs. Its search is about half the cost of
  flat's (0.02 vs 0.04 ms), because it has fewer vectors (654 vs 1,423 public, 316 vs 799
  private) and asks for k=1 instead of k=4.
- **The query embedding dominates.** It is the same MiniLM call in both systems, so it doesn't
  separate them. Anything that changes retrieval latency in practice is about the embedder:
  device, threads, batch. Exact search over a few thousand vectors is negligible.
- **MPS is slower than CPU for a single short query** (6.6 vs 3.5 ms). Kernel launch and
  transfer overhead outweigh the compute for a 6-layer model on ~20 tokens. A single CPU thread
  is slightly faster than four, for the same reason. MPS wins for bulk encoding (index builds, below).
- **Loading, once per process** (CPU, warm disk cache): MiniLM 45 to 115 ms (170 to 320 ms on
  MPS), the proxy index + metadata + tree 1 to 4 ms, the flat index + chunks under 2 ms. A
  cold first load of MiniLM on MPS took 1.6 s.
  The published Q1 latency included about 35.7 s of one-time load, mostly TinyLlama
  ([`metrics.md`](metrics.md), weakness 8).

### Context length (what each system hands the generator)

| Assembled context, chars | proxy-pointer p50 | p95 | max | > 1,200 chars | flat top-4 p50 | p95 | max | > 1,200 chars |
|---|---|---|---|---|---|---|---|---|
| Public (252 q) | 964 | 4,719 | 9,404 | 111 / 252 (44%) | 1,568 | 1,827 | 1,960 | 234 / 252 (93%) |
| Private (30 q) | 1,493 | 8,587 | 12,572 | 16 / 30 (53%) | 1,870 | 1,925 | 1,949 | 30 / 30 (100%) |

The flat context is bounded: 4 × 500 chars plus separators. The proxy-pointer context is a
whole section plus its children, so it has a long tail. The published benchmark cut **both**
to 1,200 chars, which trims almost every flat context a little and the long proxy-pointer
contexts a lot. The private numbers match [`metrics.md`](metrics.md): 16 of 30 over the cap,
with a maximum of 12,572 chars (Q3).

## Generation (TinyLlama), measured separately

**Setup.** TinyLlama-1.1B-Chat, fp32, CPU, `OMP_NUM_THREADS=1` (torch 1 thread; needed on
this Mac, see [`corpus.md`](corpus.md)), greedy, `max_new_tokens=150`.

- Both systems get **the same** system prompt (`vanilla_answer`'s grounded prompt) and the same
  user template, `Context:\n{ctx}\n\nQuestion: {q}`. Only the context differs. This removes the
  published run's asymmetries: deep route 200 new tokens against 150, and different prompts.
- Sample: 10 public questions, drawn with seed 0 from the 252, 1 run each. The model was loaded
  once (6.4 s) and one untimed warm-up generation came first.
- Arms: each system with the context capped at 1,200 chars (as published) and uncapped, plus the
  router's `fast` route on the same questions (no retrieval, its ungrounded prompt).
- "Uncapped" still has to fit TinyLlama's 2,048-token window (prompt + 150). No sampled prompt
  needed cutting; the longest was 1,008 tokens.
- Time to first token (TTFT, roughly the prefill) comes from a stopping-criteria callback on the
  first generated token. Decode rate = (new tokens − 1) / (total − TTFT).

**Run conditions.**

- A first attempt overlapped with another agent's Gemma generation on MPS. Its timings
  roughly tripled (one 150-token answer went from ~13 s to 35 s). It was stopped and discarded.
- The run below started after that job ended, and a logger sampled the top processes every 15 s.
  No other ML job ran during it, but macOS background services (Spotlight, media analysis,
  Software Update) spiked during the last question (B-053, 06:30:48 to 06:31:48). Its two
  slowest generations (27.7 s and 35.6 s for 150 tokens) are contended.
- Medians are reported because they are robust to that one question. Means are shown with and
  without B-053 in the result file.
- Total generation compute used for this task was about 18 minutes: smoke test 0.6 min,
  discarded contended run ~7 min, final run 10.1 min.

| Arm (public, n = 10) | context chars (mean) | prompt tokens mean (max) | new tokens (mean) | TTFT p50 | total p50 | decode tok/s p50 |
|---|---|---|---|---|---|---|
| proxy-pointer, capped 1,200 | 813 | 328 (449) | 110 | 0.92 s | 11.3 s | 12.7 |
| flat, capped 1,200 | 1,191 | 422 (456) | 106 | 1.03 s | 12.6 s | 12.3 |
| proxy-pointer, uncapped | 1,369 | 464 (1,008) | 118 | 1.17 s | 12.7 s | 10.8 |
| flat, uncapped | 1,510 | 507 (603) | 120 | 1.41 s | 13.1 s | 12.7 |
| router `fast` route (no retrieval) | 0 | 93 (112) | 136 | 0.24 s | 11.3 s | 13.5 |

- **Generation costs about 11 to 13 s per answer, which is about 3,000 times the 3.5 ms of
  retrieval.** Decode runs at about 12.6 tokens/s (p50 over all 50 generations, excluding B-053),
  so a full 150-token answer costs about 12 s of decoding.
- **Answer length sets the time.** Prefill is cheap at these prompt sizes. Total time
  correlates with new tokens at r = 0.85 and with prompt tokens at r = 0.22. A least-squares fit
  gives about 96 ms per generated token against 6.6 ms per prompt token. TTFT grows at about
  3.5 ms per prompt token (r = 0.90): about 1 s for a 330-token prompt, 4.2 s for 1,008 tokens.
- **Capped vs uncapped.**
  - In this sample, the 1,200-char cap cut 5 of 10 proxy-pointer contexts and 9 of 10 flat
    ones. Across all 252 public questions it cuts 44% of proxy-pointer contexts and 93% of flat.
  - Removing the cap raised proxy-pointer's mean prompt from 328 to 464 tokens and its TTFT p50
    from 0.92 to 1.17 s. The total p50 went from 11.3 to 12.7 s.
  - The worst case was A-055: its 3,946-char section became a 1,008-token prompt. TTFT went
    from 1.5 to 4.2 s and the total from 18.6 to 23.7 s.
  - Flat changed less (12.6 → 13.1 s), because its context is bounded at about 2,000 chars.
  - The bigger effect of uncapping is often on answer length, not prefill. D-025's proxy-pointer
    answer went from 50 to 114 tokens (4.6 → 10.3 s) once it saw the whole section.
- **The long tail doesn't fit the generator.** Proxy-pointer contexts reach 9,404 chars on the
  public corpus and 12,572 on the private one. That is over 2,000 tokens, more than TinyLlama's
  whole window. An uncapped deployment has to cut them anyway, at about 1,900 prompt tokens with
  150 new. By the linear fit above, prefill alone would then take about 6.5 s. That is an
  extrapolation and is likely an underestimate, because attention cost grows faster than
  linearly. A generator with a longer window would fit them, at a prefill cost that grows with
  section size. Flat-chunk contexts never get near the window.
- **The `fast` route saves prefill but spends it on longer answers.** Its prompts are 93 tokens
  and its TTFT is 0.24 s. Its answers are longer, though: 136 new tokens against 106 to 120, 520
  against 453 chars, and 5 of 10 hit the 150-token limit. So its median total (11.3 s) is no
  lower than the grounded arms'.

## Index build time and on-disk size

Each build mirrors `proxy_pointer_indexer.py` and `naive_rag_baseline.build_index` and writes
to the corpus' configured paths. The embedder load is timed separately. Each build ran 3
times; the table shows the median, and the first repeat (cold cache) was the slowest. Times
are in ms.

| | Public proxy-pointer | Public flat | Private proxy-pointer | Private flat (pypdf, as published) |
|---|---|---|---|---|
| Vectors (dim 384) | **654** | **1,423** | **316** | **799** |
| Tree build / load source | 4 | 0.3 (read markdown) | 4 | 4,058 (pypdf over 2 PDFs, 13 MB) |
| Proxy texts / chunking | 1 | 4 | 1 | 6 |
| Embedding, CPU 4 threads | 701 | 5,443 | 341 | 4,868 |
| Embedding, MPS | 386 | 2,626 | 180 | 2,120 |
| FAISS add | < 0.1 | 0.1 | < 0.1 | < 0.1 |
| Write to disk | 2 | 2 | 1 | 1 |
| **Total, CPU** | **709** | **5,449** | **347** | **8,933** |
| **Total, MPS** | **393** | **2,630** | **189** | **6,234** |
| Embedder load (once) | 67 ms CPU, 1.6 s MPS (cold) | | 53 ms CPU, 118 ms MPS | |

| On-disk size, bytes | Public proxy-pointer | Public flat | Private proxy-pointer | Private flat |
|---|---|---|---|---|
| FAISS index | 1,004,589 | 2,185,773 | 485,421 | 1,227,309 |
| Metadata pickle | 52,792 | (none) | 24,660 | (none) |
| Tree pickle (full text of every node) | 553,439 | (none) | 477,576 | (none) |
| Chunks pickle | (none) | 528,059 | (none) | 374,568 |
| **Total** | **1,610,820 (1.54 MiB)** | **2,713,832 (2.59 MiB)** | **987,657 (0.94 MiB)** | **1,601,877 (1.53 MiB)** |

- Proxy-pointer builds 7.7× (public) and 26× (private) faster and is 40% smaller on disk.
  It embeds fewer, shorter texts: one heading + first line per node, against overlapping
  500-char chunks. The private flat build is dominated by pypdf (4.1 s), not by embedding.
- An `IndexFlatIP` costs `ntotal × 384 × 4` bytes. Vector count sets the index size, and the
  tree pickle (which holds the full corpus text) is the proxy-pointer system's other main cost.
- The rebuilt private indices match the legacy ones used in the published benchmark.
  `pp_metadata.pkl`, `pp_tree.pkl` and `chunks.pkl` are byte-identical, and the vectors agree
  to within 6e-7, which is float noise.

## Memory

Apple Silicon has **unified memory**: CPU and GPU share the same 16 GB. None of the numbers
below is VRAM. They are labelled **"unified-memory high-water mark / peak RSS"**.

- **Peak RSS** is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, which is bytes on macOS, for one
  process per run.
- **MPS driver high-water** is the maximum of `torch.mps.driver_allocated_memory()` sampled after
  load and after every query or batch. torch only exposes the current value. That figure
  includes the Metal allocator's cached heap, so it is much larger than the live tensors
  (`current_allocated` high-water is 87 MiB, about MiniLM's fp32 weights).

| Process (unified-memory high-water mark / peak RSS) | Public | Private |
|---|---|---|
| Python + numpy + torch imported | 179 MiB | 179 MiB |
| Retrieval, CPU: MiniLM + both indices loaded | 439 MiB | 417 MiB |
| Retrieval, CPU: peak over the timed run | **526 MiB** | **485 MiB** |
| Retrieval, MPS: peak RSS / MPS driver high-water | 726 MiB / 1,067 MiB | 561 MiB / 1,067 MiB |
| Index build, CPU (both indices, 3 repeats) | 1,051 MiB | 935 MiB |
| Index build, MPS: peak RSS / MPS driver high-water | 653 MiB / 1,300 MiB | 685 MiB / 1,229 MiB |
| Retrieval stack loaded (generation run, CPU, 1 thread) | 486 MiB | not run |
| + TinyLlama-1.1B fp32 loaded | 4,008 MiB | not run |
| Generation, peak over the run (MiniLM + both indices + TinyLlama fp32, CPU) | **4,715 MiB** | not run |

**VRAM (CUDA): not yet measured.** There is no CUDA GPU here.
[`colab/measure_vram.py`](../colab/measure_vram.py) is a runnable Colab T4 script and has not
been run yet. It reports `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` for
these arms:

- the QLoRA training arm (finetune.py's config: 4-bit NF4, LoRA r=8/α=16 on q_proj/v_proj,
  batch 1 × grad-accum 4, max length 512), once on the training pairs and once with every
  example at the full 512 tokens;
- base Gemma-3-1b-it;
- Gemma + LoRA adapter;
- TinyLlama.

Each inference arm runs with a short prompt and with an ~1,800-token context.

| Arm (Colab T4) | Peak VRAM |
|---|---|
| QLoRA training (finetune.py config) | not yet measured |
| Gemma-3-1b-it inference | not yet measured |
| Gemma-3-1b-it + adapter inference | not yet measured |
| TinyLlama-1.1B-Chat inference | not yet measured |

## Why 22.2 s vs 26.7 s

The published numbers are mean wall-clock per answer over the 30 private questions, one run
each, on CPU. Each one includes loading on first use and **generation**. Against those:

- **Retrieval is about 0.02% of an answer.** It is roughly 4 ms out of 22,000 to 27,000 ms. The
  4.5 s gap is about 1,200 times the entire retrieval cost of either system, so it can't come
  from retrieval. The README's old explanation, a "two-step embed-then-fetch" on the deep
  route, is ruled out: the fetch (dereference + assemble) costs about 5 µs.
- **The gap comes from the router's fast route.** [`metrics.md`](metrics.md) (weakness 3 and 8)
  shows that 18 of the 30 hybrid questions routed `fast`: no retrieval at all, bare TinyLlama
  with an ungrounded prompt. Those answers were about twice as long (385 vs 196 chars). Without
  Q1, `fast` averaged 30.79 s for the hybrid against 19.83 s for vanilla on the same questions.
  On the 12 `deep` questions, where proxy-pointer retrieval was actually used, the hybrid was
  faster (21.33 s vs 22.97 s). The controlled run above shows the same mechanism: with no
  context the model writes longer answers (136 vs about 110 new tokens), and that costs more
  than the prefill it saves.
- **Generation time follows the number of generated tokens.** Latency correlated with answer
  length at r = 0.70 / 0.74 in the published run. The deep route also allowed 200 new tokens
  against vanilla's 150. At the measured ~12.6 tokens/s (this machine, 1 thread), 4.5 s is
  about 55 extra generated tokens per answer, roughly 220 chars. That is the size of the
  answer-length difference between the routes.
- **One-time load.** Vanilla's Q1 included about 35.7 s of model and index loading, which adds
  1.2 s to its mean. Retrieval's own load (MiniLM + indices) is about 0.1 s of that. The rest
  is TinyLlama.

So the published latencies compare generators and routing, not retrievers. Measured directly,
the two retrievers cost the same: a few milliseconds, almost all of it spent embedding the query.

## Reproduce (public corpus)

```bash
export HF_HUB_OFFLINE=1 PPRAG_CORPUS=public
./scripts/fetch_corpus.sh
python3 benchmark_latency.py build     --corpus public --repeats 3 --device cpu --out results/latency/public_build_cpu.json
python3 benchmark_latency.py build     --corpus public --repeats 3 --device mps --out results/latency/public_build_mps.json
python3 benchmark_latency.py retrieval --corpus public --questions questions/public_seed.jsonl [public_v1.jsonl] \
        --repeats 3 --warmup 20 --device cpu --out results/latency/public_retrieval_cpu.json
OMP_NUM_THREADS=1 python3 benchmark_latency.py retrieval ... --out results/latency/public_retrieval_cpu_omp1.json
python3 benchmark_latency.py retrieval ... --device mps --out results/latency/public_retrieval_mps.json
OMP_NUM_THREADS=1 python3 benchmark_latency.py generate --corpus public \
        --questions questions/public_seed.jsonl [public_v1.jsonl] --gen-n 10 --caps 1200 none --fast-arm \
        --out results/latency/public_generate_cpu.json
```

The build writes the indices that `retrieval` and `generate` read, so build first. For the
private corpus, run from a directory holding the private inputs, with `--corpus private` and
`--out` pointing outside the repo; the script refuses to write private results inside it.
