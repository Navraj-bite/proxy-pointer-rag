# AI Engineering Internship — Weeks 1–3 Technical Summary

**Program:** 4-week applied AI engineering internship
**Domain (Weeks 1–2):** Operator support for **Serac filling machines** (industrial packaging equipment), grounded in two source manuals — `machine_overview_troubleshooting.pdf` and `operator_manual.pdf`.
**Domain (Week 3+):** Desktop and e-commerce automation (Amazon India), building toward the Week 4 browser agent.

This document is a technical handover: enough detail to reproduce the work, the concrete artifacts each week produced, and the honest friction points. A closing **Cross-Week Synthesis** shows how the three tracks converge into Week 4's browser-use agent.

---

## Week 1 — Small Language Model Fine-Tuning vs. Traditional RAG

### Objective
Answer a practical build question for a domain-support assistant: **when you have a narrow corpus (two equipment manuals) and a small compute budget, should you fine-tune a small language model (SLM) or stand up a retrieval-augmented generation (RAG) pipeline?** The deliverable is a like-for-like comparison and a recommendation, not just a working demo.

### Architecture / Approach
Two systems were built against the same corpus and evaluated on the same question set.

**1. Baseline RAG pipeline** — `naive_rag_baseline.py` (plus `rag.py`, `run_rag_baseline.py`, `parse_documents.py`):
- Load both PDFs → extract text with `pypdf`.
- Chunk with `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
- Embed with `all-MiniLM-L6-v2` (`sentence-transformers`), L2-normalized.
- Index in FAISS `IndexFlatIP` (inner product on normalized vectors = cosine). Persisted as `faiss.index` + `chunks.pkl`.
- Retrieve `TOP_K=4` chunks, stuff into a grounded prompt, generate with `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

**2. Fine-tuned SLM** — `finetune.py`:
- Base model **`google/gemma-3-1b-it`**, QLoRA (4-bit NF4 quantization, bf16 compute) via `bitsandbytes` + `peft` + `trl`'s `SFTTrainer`.
- SFT dataset `train.jsonl` (instruction/output pairs synthesized from the manuals), formatted through the tokenizer's chat template.

### Key Implementation Details
| Setting | Value |
|---|---|
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.05 |
| Quantization | 4-bit NF4, bf16 compute dtype |
| Epochs / batch / grad-accum | 3 / 1 / 4 |
| Learning rate / max length | 2e-4 / 512 |

`finetune.py` is MPS-aware: it attempts local training on Apple Silicon, and on the (expected) MPS kernel failure it **auto-generates `finetune_colab.ipynb`** targeting a free Colab **T4 GPU**, since the 4-bit training stack is CUDA-bound. The trained adapter lands in `serac-gemma-adapter/`.

### Results / Deliverables
- Index + chunk store: `faiss.index`, `chunks.pkl`.
- SFT + eval data: `train.jsonl`, `evaluation_prompts.jsonl`, `benchmark_questions.jsonl`.
- Generated notebook: `finetune_colab.ipynb`; LoRA adapter: `serac-gemma-adapter/`.
- Head-to-head outputs: `rag_baseline_outputs.csv`, `finetuned_outputs.csv`, merged into `comparison.csv`; scored via `score.py` / `run_finetuned.py`.

| | Traditional RAG | Fine-tuned SLM (QLoRA) |
|---|---|---|
| Knowledge source | External index, retrieved at query time | Baked into weights |
| Updating the corpus | Re-embed & re-index (cheap) | Re-train the adapter |
| Grounding / citations | Strong — answers trace to chunks | Weaker — no provenance |
| Latency | Higher (retrieve + generate) | Lower (single forward pass) |
| Best at | Fact lookup, tables, "where is X" | Fluent phrasing of learned procedures |

**Recommendation:** For a factual, frequently-updated manual corpus, retrieval is the stronger backbone (provenance + cheap updates); fine-tuning is complementary — a fast, low-latency "reflex" answerer for common, stable questions. This directly motivated Week 2's **hybrid router**.

### Challenges & Learnings
- **Gemma is a gated Hugging Face model.** Without HF auth in the local environment, downstream evaluation used **`TinyLlama-1.1B` as a stand-in generator**. All pipeline, prompt, and scoring logic is identical; only base weights differ — so scores reflect *retrieval/pipeline* quality more than final generation quality.
- **MPS ≠ CUDA for 4-bit training.** Don't fight it locally; the Colab-fallback path is the reproducible route.
- Cosine-on-normalized-vectors with `IndexFlatIP` is the simplest correct FAISS setup for small corpora — no training, exact search.

---

## Week 2 — Structure-Aware Proxy-Pointer RAG

### Objective
Naive fixed-size chunking shreds structured content: a 500-character window splits pipe tables and multi-step procedures mid-way, so the model receives fragments instead of intact sections. Week 2 builds a **structure-aware retriever** that respects the document's heading hierarchy and returns *whole sections*, then routes queries to the cheapest system that can answer them.

### Architecture / Approach
1. **Markdown conversion** — `parse_documents.py` converts the PDFs into heading-structured markdown (`operator_manual.md`, `machine_overview_troubleshooting.md`).
2. **Document tree** — `document_tree_builder.py` builds a `DocumentTree` of `DocumentNode`s. Each node carries `heading_title`, `body_content`, `parent_node_id`, and `start_line`/`end_line`. A synthetic `ROOT` and per-file `[FILE] …` container nodes anchor the hierarchy.
3. **Proxy-Pointer index** — `proxy_pointer_indexer.py` embeds a short **proxy** for every content node (its `heading_title` + the first meaningful body line, kept `<300` chars) rather than the full body. It stores:
   - `pp_faiss.index` — proxy embeddings (`all-MiniLM-L6-v2`, `IndexFlatIP`),
   - `pp_metadata.pkl` — the **pointer**: `pointer_node_id`, `heading_title`, `doc_id`, `start_line`, `end_line`,
   - `pp_tree.pkl` — the full tree for dereferencing.

   At query time the retriever matches against light proxies, then **dereferences the pointer** back to the node's full `body_content` (`context_assembler.py`) — returning an intact section (with complete tables) instead of a 500-char fragment.
4. **Intelligent query router** — `hybrid_intent_router.py` classifies each question **fast** vs **deep** via a regex trigger table (`table`, `parameters`, `procedure`, `steps`, `how do i`, `list`, …) with a `certain`/`borderline` confidence signal. **Fast** → the fine-tuned SLM answers directly; **deep** → proxy-pointer RAG assembles context, then the SLM answers grounded.

### Key Implementation Details
- **Proxy/pointer separation** is the core idea: index a cheap, discriminative summary; fetch heavy content only on a hit. `[FILE]` and `ROOT` nodes are excluded from the index (no searchable content, they'd pollute results). Table-only nodes fall back to their heading as the proxy.
- **Linkage verification** is built into the indexer: `verify_linkage()` round-trips random proxy entries back to their full nodes, asserting each pointer still resolves to non-empty body content (with table/step detection).
- Router thresholds: **0 triggers →** fast/certain; **exactly 1 →** deep/borderline; **≥2 →** deep/certain — biasing ambiguous structured questions toward the retriever.

### Results / Deliverables
Benchmarked over 25+ categorized questions (`run_day10_benchmark.py`), reported in `proxy_rag_vs_vanilla_rag.md` + `proxy_rag_vs_vanilla_rag_results.csv`:

| Metric | Vanilla RAG | Hybrid Router + Proxy-Pointer |
|---|---|---|
| Mean cosine score | **0.6985** | 0.6810 |
| — tabular | 0.6625 | 0.6608 |
| — procedure | 0.6169 | **0.6487** |
| — navigation | **0.8160** | 0.7334 |
| Retrieval correctness (keyword) | **67%** | 47% |
| Mean latency | **22.2 s** | 26.7 s |

**Reading the numbers honestly:** the hybrid system did **not** beat vanilla on the aggregate cosine mean, and was slower (the deep route pays a two-step embed-then-dereference cost). Where it clearly wins is **structured content** — e.g. Q17 (procedure) improved by **+0.42**, Q3 (alarm table) by **+0.30**, because the proxy-pointer index returns the exact section node with its pipe table intact rather than a fragmented snippet. Where it loses, short navigation questions with a unique keyword are already trivially found by flat chunking.

### Challenges & Learnings
- Aggregate metrics can hide the win: structure-aware retrieval helps precisely on the queries flat RAG mangles (tables, procedures), so **evaluate by category**, not just a global mean.
- As in Week 1, generation quality is capped by the TinyLlama stand-in — the router and retrieval logic is production-ready; only the model weights are provisional.
- Building the tree from *converted* markdown means conversion fidelity (`parse_documents.py`) is load-bearing: bad heading detection upstream degrades every node downstream.

---

## Week 3 — Desktop Automation & E-Commerce Trajectory Capture (OpenAdapt)

### Objective
Move from text pipelines to **acting on a live UI**. Capture a real human demonstration (shopping on Amazon India), scrub it of private data, verify actions are correctly grounded to on-screen elements, then **abstract the recording into a reusable, parameterized skill** — and stress-test whether that skill generalizes. This is the raw material for a real agent.

### Architecture / Approach
The pipeline: **record → validate → summarize events → ground/verify → abstract → replay**.

- **Install & foundation** — `automation_setup.md` documents the setup; a dedicated venv `.venv-openadapt` runs **OpenAdapt meta 1.2.5 / openadapt-capture 0.5.3** with `pyautogui`. `foundation_automation.py` and `workflow_runner.py` (mapped in `workflow_map.md`) exercise the capture/replay loop on local Notepad workflows first.
- **Privacy scrubbing** — two layers. OpenAdapt's built-in `openadapt/scrub.py` handles PII scrubbing in the capture pipeline; `sanitize_screenshots.py` (custom) produces **side-by-side raw-vs-sanitized** comparison images, Gaussian-blurring and red-labelling operator-specific fields (dates, sign-off blocks, operator names) before any screenshot is shared. Output in `sanitized/`.
- **VLM-based action grounding** — `build_trajectories.py` establishes VLM ground-truth anchors for each action; `coordinate_verifier.py` cross-checks recorded click coordinates against the actual on-screen target (OpenCV fallback), producing `vlm_verification_results.json` and `day13_verification_output.png`.
- **Amazon India recording** — `amazon_recording_setup.py` / `amazon_validate_capture.py` capture a real add-to-cart session into `amazon_add_to_cart/recording.db`. `validate_recording.py` + `event_log_summary.md` distill it.
- **Trajectory abstraction / generalization** — `coordinate_abstractor.py` lifts raw pixels into named intents (`ui_coordinates.py` constants) and a parameterized skill (`agent_skills/amazon_add_to_cart.py`, `abstracted_trajectory.json`). `replay_test.py` replays it and logs `replay_results.json`.

### Key Implementation Details
- **Event denoising:** the raw capture held **875 events**; after dropping 568 mouse-moves and 239 scrolls, only **18 meaningful events** remained (5 logical clicks: search box → first result → add-to-cart, plus navigation/typing). Denoising is what makes abstraction tractable.
- **Intent-level abstraction beats pixel replay** where it was applied:
  - Navigation replaced with `webbrowser.open()` instead of replaying an address-bar click + keystrokes — the single most reliable abstraction.
  - The search term was **parameterized** (`search_term` arg), working across 5 distinct queries despite the recording's typo (`"mechaincal keyboard"`).
- **Generalization stress test:** the recording was made on **amazon.in** but replayed against **amazon.ca** on purpose, to test whether the abstractions survive a site change.

### Results / Deliverables
Reported in `desktop_automation_performance.md` and `coordinate_abstraction_report.md`:

| Metric | Result |
|---|---|
| Harness "success" (calls fired) | **5/5 (100%)** |
| Task success — blind replay | **0/5 (0%)** |
| Task success — after coordinate correction | ✅ item added to cart (one product layout) |

**The key finding:** the harness reports success when a click *fires*, not when it *lands*. The navigation/search half generalized cleanly across domains and queries; the **Add-to-Cart click did not**, because the buy-box moves vertically per product (title length, badges, coupons). The recorded pixel `(1353,396)` missed on 100% of runs; correcting to `(1238,747)` from the product screenshot completed the flow — but only for that one layout.

Artifacts: `event_log_summary.md`, `coordinate_abstraction_report.md`, `desktop_automation_performance.md`, `abstracted_trajectory.json`, `agent_skills/amazon_add_to_cart.py`, `replay_results.json`, plus scrubbed comparison images and proof screenshots (`day13_verification_output.png`, `proof_of_execution.png`).

### Challenges & Learnings
- **A coordinate-only skill cannot self-verify.** It literally can't tell "clicked Add to Cart" from "clicked empty space." Post-action verification (did the cart count change?) is mandatory.
- **A bad abstraction heuristic silently mislabels the goal.** `coordinate_abstractor.py`'s "last click = the goal" rule tagged the wrong click as Add-to-Cart; ground truth showed the *second-to-last* click was the real target. Heuristics need verification against the visual state.
- **Reported "latency" was mostly hardcoded sleeps** (4s page-load, 3s post-submit, etc.), not real page responsiveness — fixed-wait design is both wasteful and fragile.
- **The load-bearing conclusion:** for a webpage that ships a queryable DOM, **desktop pixel automation is the wrong tool**. Variable elements should be located by text/role at runtime (VLM/OCR or the DOM), not by a recorded pixel. This lesson is exactly what Week 4 acts on.

---

## Cross-Week Synthesis → Week 4 Browser Agent

Week 4 builds an **autonomous, LLM-driven Amazon add-to-cart agent** (`run_amazon_trajectory.py` on **browser-use 0.13.1**). Each earlier week feeds it a distinct capability:

**Week 3 supplies the task and the hard lesson.** The recorded demonstration is distilled into `amazon_trajectory_plan.json` — natural-language steps with the recorded pixels **demoted to advisory hints**. Week 4's driver flattens that plan into a single NL prompt (`build_cart_task`) and instructs the model to resolve each step onto a live element via a custom `find_element_by_description` tool. This is the direct response to Week 3's finding that pixels don't generalize: **replay the intent, not the coordinates.** The result is verified end-to-end — an `ASUS TUF K3 Mechanical Keyboard` added to cart (quantity read deterministically from Amazon's own cart page via CDP), with a **hard stop before checkout** and a CAPTCHA-halt guard.

**Week 2 supplies the routing pattern and grounded knowledge.** The hybrid **fast/deep router** — cheap reflex path vs. retrieval-backed deep path — is the same architecture an agent needs to decide when to act directly versus consult a knowledge source. The proxy-pointer retriever is the template for grounding an agent in structured domain docs (manuals, policy pages) without stuffing whole documents into context.

**Week 1 supplies the model-selection discipline.** The RAG-vs-fine-tune comparison establishes *when a small, cheap model is enough* — the same trade-off that dominated Week 4's real fight: finding a free-tier LLM (Groq `llama-4-scout` won over Gemini's 20-requests/day cap) that could sustain ~14k-token agent steps. The QLoRA-adapted SLM is the natural candidate for the router's low-latency **fast route** or an on-device policy model.

**The through-line:** Week 1 asked *how to answer cheaply*, Week 2 asked *how to retrieve the right structured context*, Week 3 asked *how to capture and generalize a real UI task* — and Week 4 fuses all three into an agent that **perceives a live page, decides with a rate-limit-appropriate model, and acts on intent rather than recorded pixels.**

---

*Handover note: throughout Weeks 1–2 the generator is TinyLlama-1.1B standing in for the fine-tuned Gemma-3-1b (gated behind HF auth); swap `_GEN_MODEL`/`_USE_ADAPTER` in `hybrid_intent_router.py` and re-run once auth is available. Week 3's skill is a working prototype on the recorded path and a proof-of-concept for general products — one robust-targeting (visual/DOM) layer short of a general agent, which Week 4 provides.*
