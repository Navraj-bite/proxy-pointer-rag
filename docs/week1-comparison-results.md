# Week 1 — SLM Fine-Tuning vs. Traditional RAG: Comparison Results

_Source: `comparison.csv` — 40 evaluation questions on the Serac manuals. Scores are cosine similarity of each system's answer vs. the reference answer (generator = TinyLlama-1.1B stand-in for gated Gemma-3-1b; scores reflect retrieval/pipeline quality)._

## Average scores

| System | Mean cosine score | Wins (head-to-head) |
|---|---|---|
| Traditional RAG | **0.4895** | 18/40 |
| Fine-tuned SLM (QLoRA) | **0.5079** | 22/40 |
| _ties_ | — | 0 |

Δ (fine-tuned − RAG) mean = **+0.0185**. Higher scorer overall: **Fine-tuned SLM**.

## ⚠️ Latency

`comparison.csv` contains **no latency column** — per-item latency was not recorded for the Week 1 RAG-vs-fine-tune benchmark. **MISSING — do not fabricate.** (Latency data does exist for Week 2, in `proxy_rag_vs_vanilla_rag_results.csv`.)

---

_Per-question answers are omitted: generated answers quote the source manuals, which are not redistributable. The scoring code (`score.py`) and the benchmark harness are included; point them at your own corpus._
