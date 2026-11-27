# Week 2 — Proxy-Pointer RAG vs. Vanilla RAG: Benchmark

_Source: `proxy_rag_vs_vanilla_rag_results.csv` — **30 categorized questions** on the Serac manuals. Computed directly from the CSV (matches `proxy_rag_vs_vanilla_rag.md`)._

## Headline metrics

| Metric | Vanilla RAG | Hybrid Router + Proxy-Pointer |
|---|---|---|
| Mean cosine score | **0.6985** | 0.6810 |
| Retrieval correctness (keyword match) | **20/30 = 67%** | 14/30 = 47% |
| Mean latency (s) | **22.2** | 26.7 |

Δ score (hybrid − vanilla) = **-0.0175**. _Note: 'recall' as such is not recorded; retrieval **correctness** (keyword match vs. reference) is the closest metric present._

## Category breakdown (mean cosine)

| Category | Vanilla | Hybrid | Δ |
|---|---|---|---|
| navigation | 0.8160 | 0.7334 | -0.0826 |
| procedure | 0.6169 | 0.6487 | +0.0318 |
| tabular | 0.6625 | 0.6608 | -0.0017 |

## Before/After cards — 3 biggest Proxy-Pointer WINS

### 17 (procedure) — Δ = **+0.416**
**Question:** How do you use the Filter function when selecting a production file?

- **Vanilla (score 0.225):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Proxy-Pointer (score 0.641):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Improvement:** +0.416 cosine

### 3 (tabular) — Δ = **+0.298**
**Question:** Which alarms in the DataLiner table have FDA logging enabled?

- **Vanilla (score 0.290):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Proxy-Pointer (score 0.588):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Improvement:** +0.298 cosine

### 21 (navigation) — Δ = **+0.165**
**Question:** What applications are available in the taskbar?

- **Vanilla (score 0.535):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Proxy-Pointer (score 0.699):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Improvement:** +0.165 cosine

## Before/After cards — 3 biggest Vanilla WINS (where hybrid lost)

### 24 (navigation) — Δ = **-0.401**
**Question:** What information is shown in the bottom-right information field on the screen?

- **Vanilla (score 0.839):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Proxy-Pointer (score 0.438):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

### 25 (navigation) — Δ = **-0.395**
**Question:** Which taskbar icon do I use to access maintenance functions?

- **Vanilla (score 0.876):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Proxy-Pointer (score 0.481):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

### 11 (procedure) — Δ = **-0.296**
**Question:** What are the steps to start production mode on the Serac filling machine?

- **Vanilla (score 0.621):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_
- **Proxy-Pointer (score 0.325):** _[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

## Conclusion (from proxy_rag_vs_vanilla_rag.md)

- Vanilla wins the **aggregate** cosine mean, but Proxy-Pointer wins on **structured content** (tables, procedures) by returning the exact section node with intact pipe tables instead of 500-char fragments.
- Latency trade-off: hybrid is slightly slower (two-step embed-then-dereference).
- Generator = TinyLlama stand-in for gated Gemma-3-1b; scores reflect retrieval quality more than generation.

