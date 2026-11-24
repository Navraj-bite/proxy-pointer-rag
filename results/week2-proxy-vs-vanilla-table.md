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

- **Vanilla (score 0.225):** The context of the question is a support assistant for Serac filling machines. The instruction 117D000265/0EN Page 23/359 is a French instruction for the English translation of the original French version 117D000265EN. T…
- **Proxy-Pointer (score 0.641):** When selecting a production file using the Filter function in Serac filling machines, follow these steps: 1. Open the Serac software. 2. Click on the "Filters" tab. 3. Click on the "Filter" button. 4. In the "Filter" dia…
- **Improvement:** +0.416 cosine

### 3 (tabular) — Δ = **+0.298**
**Question:** Which alarms in the DataLiner table have FDA logging enabled?

- **Vanilla (score 0.290):** The FDA tick box is mainly used for errors in the DataLiner table. The user may sort the Life messages using the FDA criteria by clicking on the FDA tick box. The Life messages of the DataLiner table may be composed of s…
- **Proxy-Pointer (score 0.588):** The context provided in the question is the DataLiner table in the Operator Intervention window. The answer is that FDA logging is enabled for the alarms in the table.
- **Improvement:** +0.298 cosine

### 21 (navigation) — Δ = **+0.165**
**Question:** What applications are available in the taskbar?

- **Vanilla (score 0.535):** The context of the question is the English translation of the original French version of the instruction for Serac filling machines. The question asks about the available applications in the taskbar, which are "HMI" (ret…
- **Proxy-Pointer (score 0.699):** The taskbar is a small area on the bottom of your computer screen that displays various applications and settings. Here are some commonly available applications in the taskbar: 1. Windows Explorer: This is the default fi…
- **Improvement:** +0.165 cosine

## Before/After cards — 3 biggest Vanilla WINS (where hybrid lost)

### 24 (navigation) — Δ = **-0.401**
**Question:** What information is shown in the bottom-right information field on the screen?

- **Vanilla (score 0.839):** The information shown in the bottom-right information field on the screen is the machine status, the current access level (or name of the user connected), and the machine speed.
- **Proxy-Pointer (score 0.438):** The bottom-right information field on the screen shows the date and time of the last operation performed on the Serac filling machine.

### 25 (navigation) — Δ = **-0.395**
**Question:** Which taskbar icon do I use to access maintenance functions?

- **Vanilla (score 0.876):** The taskbar icon used to access maintenance functions is "Maintenance" (located at the bottom of the screen).
- **Proxy-Pointer (score 0.481):** To access maintenance functions on Serac filling machines, you should use the "Maintenance" taskbar icon. This icon is located on the taskbar of the machine and can be accessed by clicking on it. The Maintenance icon wil…

### 11 (procedure) — Δ = **-0.296**
**Question:** What are the steps to start production mode on the Serac filling machine?

- **Vanilla (score 0.621):** The steps to start production mode on the Serac filling machine are: 1. Stop the machine by pressing the "Jog" button on the remote control box. 2. Press the "Normal/Maintenance" key switch on the control panel. 3. Close…
- **Proxy-Pointer (score 0.325):** Answer: 1. Turn on the machine's power switch. 2. Wait for the machine to come to life and the LED indicator to turn on. 3. Press the "Start" button on the control panel. 4. The machine will start to fill the container w…

## Conclusion (from proxy_rag_vs_vanilla_rag.md)

- Vanilla wins the **aggregate** cosine mean, but Proxy-Pointer wins on **structured content** (tables, procedures) by returning the exact section node with intact pipe tables instead of 500-char fragments.
- Latency trade-off: hybrid is slightly slower (two-step embed-then-dereference).
- Generator = TinyLlama stand-in for gated Gemma-3-1b; scores reflect retrieval quality more than generation.

