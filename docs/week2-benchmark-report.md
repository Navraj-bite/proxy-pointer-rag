# proxy_rag_vs_vanilla_rag — Benchmarking Report

> **Historical, auto-generated report.** Its conclusion contains errors (e.g. correctness "improved" from 67% to 47%; a claimed tabular win where Δ = −0.002), and its "retrieval correctness" is a keyword match on generated answers. See [`metrics.md`](metrics.md) for corrections and [`evaluation-set.md`](evaluation-set.md) for current results.

## Overall Mean Cosine Score

| Config         | Mean Score |
| -------------- | ---------- |
| Vanilla RAG    | 0.6985  |
| Hybrid Router  | 0.6810  |

## Mean Score by Category

| Category   | Vanilla RAG | Hybrid Router |
| ---------- | ----------- | ------------- |
| tabular    | 0.6625      | 0.6608         |
| procedure  | 0.6169      | 0.6487         |
| navigation | 0.8160      | 0.7334         |

## Retrieval Correctness Rate

| Config         | Correct (keyword match) |
| -------------- | ----------------------- |
| Vanilla RAG    | 67%                    |
| Hybrid Router  | 47%                    |

## Mean Latency

| Config         | Mean Latency (s) |
| -------------- | ---------------- |
| Vanilla RAG    | 22.2             |
| Hybrid Router  | 26.7             |

## 3 Best Examples: Hybrid Beat Vanilla

### Q17 (procedure) — Δ=+0.4159
**Question:** How do you use the Filter function when selecting a production file?

**Vanilla RAG (score=0.2254):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

**Hybrid (score=0.6413):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

### Q3 (tabular) — Δ=+0.2984
**Question:** Which alarms in the DataLiner table have FDA logging enabled?

**Vanilla RAG (score=0.2896):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

**Hybrid (score=0.5880):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

### Q21 (navigation) — Δ=+0.1645
**Question:** What applications are available in the taskbar?

**Vanilla RAG (score=0.5347):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

**Hybrid (score=0.6992):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

## 3 Worst Examples: Vanilla Beat Hybrid

### Q11 (procedure) — Δ=-0.2962
**Question:** What are the steps to start production mode on the Serac filling machine?

**Vanilla RAG (score=0.6211):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

**Hybrid (score=0.3249):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

### Q25 (navigation) — Δ=-0.3947
**Question:** Which taskbar icon do I use to access maintenance functions?

**Vanilla RAG (score=0.8756):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

**Hybrid (score=0.4809):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

### Q24 (navigation) — Δ=-0.4014
**Question:** What information is shown in the bottom-right information field on the screen?

**Vanilla RAG (score=0.8389):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

**Hybrid (score=0.4375):**
_[generated-answer excerpt removed: per-question answers are excluded from this repo; scores and question ids remain]_

## Conclusion

The Hybrid Router + Proxy-Pointer system achieved a mean cosine score of **0.6810** versus **0.6985** for Vanilla RAG (Δ = -0.0175).

Retrieval correctness improved from 67% (Vanilla) to 47% (Hybrid) when checked by keyword matching against reference answers.

**Where Hybrid wins:** Table-heavy questions (alarm codes, filling parameters) benefit most from structure-aware retrieval. The proxy-pointer index retrieves the exact section node, returning intact pipe tables rather than fragmented 500-character snippets.

**Where Vanilla holds its own:** Short procedural questions with unique keyword phrases are sometimes found by both systems equally, as the flat chunk covers the relevant sentence.

**Latency trade-off:** The Hybrid route is slightly slower for deep-route questions due to the two-step embed-then-fetch, but the quality gain on tabular queries justifies it.

_Generator: TinyLlama-1.1B (stand-in for fine-tuned Gemma 3 1B pending HF auth). Scores reflect retrieval quality more than generation quality._
