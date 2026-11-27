# Router evaluation: how much of the hybrid is routing? (Task 6)

The published "Proxy-Pointer" column was not proxy-pointer retrieval. It was
`hybrid_intent_router`: a regex trigger table sends every question with **zero** triggers to
`fast` (a bare LLM, **no retrieval**) and everything else to `deep` (proxy-pointer retrieval +
LLM). On the private benchmark 18 of 30 questions went `fast` ([`metrics.md`](metrics.md),
weakness 3), so routing and retrieval were confounded. This page measures them separately:

1. **Router accuracy** on its own, against hand labels and against a category oracle.
2. **A routing × retrieval 2×2 at retrieval level**, on both corpora.
3. **The same 2×2 at answer level** on the private corpus, with every generation setting held
   equal across cells (section 3).

Public-corpus numbers reproduce from a clean clone:

```bash
./scripts/fetch_corpus.sh
HF_HUB_OFFLINE=1 python3 router_eval.py retrieval --md corpus/px4_*.md \
    --questions questions/public_v1.jsonl --labels questions/router_labels_public.jsonl \
    --out-dir artifacts/public/task6 --bootstrap 10000 --seed 0
```

Private-corpus numbers are labelled **(private)**. They come from the proprietary manuals and
cannot be regenerated from this repo. Only aggregates are reported.

## Labelling criterion

Each question was hand-labelled `fast` or `deep` by asking what the router's own docstring says
it is for: should this go to a domain-tuned model with no lookup, or to retrieval?

- **deep**: the answer needs structured content from the documentation. That means a table
  (one or more rows or columns), ordered steps, content spread over several sections, or
  specific identifiers and values a model can't be expected to know: exact parameter names,
  enum values, defaults, thresholds, file paths, message names.
- **fast**: a simple question that a domain-tuned model could plausibly answer without lookup.
  That covers conceptual or "why" questions, yes/no capability questions, general drone or
  RC/battery knowledge, well-known UI locations, and questions whose answer is implied by the
  wording (for example, a parameter whose name spells out its meaning).

This was per-question judgement, not a keyword rule. The labeller saw the question and its
reference answer, not the router's output. Borderline calls, recorded in the reasons, mostly
fall into three groups:

- short, very common QGC flows (flash firmware, pick an airframe) were labelled `fast`;
- arbitrary PX4 code tables (LED colours, tune meanings) were labelled `deep`, even when the
  answer is one row;
- general piloting technique (hand-launch, manual landing) was labelled `fast`.

Labels:

- public: [`questions/router_labels_public.jsonl`](../questions/router_labels_public.jsonl),
  with `id`, `category`, `label` and a one-line `reason`;
- private: kept next to the private data, not committed.

**Caveat:** one labeller made one pass. There is no second annotator, so no inter-annotator
agreement is reported. Treat the oracle row as "a reasonable router", not ground truth.

| Corpus | n | labelled deep | labelled fast | tabular | procedure | navigation | multihop |
|---|---|---|---|---|---|---|---|
| public | 252 | 205 | 47 | 49 / 9 | 55 / 9 | 50 / 27 | 51 / 2 |
| private | 30 | 19 | 11 | 10 / 0 | 6 / 4 | 3 / 7 | n/a |

(Category cells are deep / fast.) Even navigation questions are mostly `deep` on the public
set, because most of them ask for an exact parameter name or value.

## 1. Router accuracy

`classify_query` (0 triggers → fast, ≥1 → deep) compared against the hand labels and against a
category oracle (tabular / procedure / multihop → deep, navigation → fast).

| Corpus | Reference | n | Accuracy [95% CI] | Cohen's κ | fast P / R | deep P / R |
|---|---|---|---|---|---|---|
| public | hand labels | 252 | **0.623** [0.563, 0.683] | 0.23 | 0.30 / 0.79 | 0.92 / 0.59 |
| public | category oracle | 252 | 0.623 [0.563, 0.683] | 0.24 | 0.43 / 0.68 | 0.81 / 0.60 |
| private | hand labels | 30 | **0.700** [0.533, 0.867] | 0.43 | 0.56 / 0.91 | 0.92 / 0.58 |
| private | category oracle | 30 | 0.667 [0.500, 0.833] | 0.37 | 0.50 / 0.90 | 0.92 / 0.55 |

The CIs come from a bootstrap over questions (10k resamples, seed 0). P = precision and R = recall
for that route. As a sanity check on the category oracle, it agrees with the hand labels on
0.722 (public) and 0.767 (private).

Confusion matrices (rows = hand label, columns = router):

| public | router fast | router deep | | private | router fast | router deep |
|---|---|---|---|---|---|---|
| **label fast** | 37 | 10 | | **label fast** | 10 | 1 |
| **label deep** | 85 | 120 | | **label deep** | 8 | 11 |

Per category (router accuracy vs hand labels / vs category oracle; label and router splits are
fast / deep):

| Corpus | Category | n | labels | router | acc vs labels | acc vs category oracle |
|---|---|---|---|---|---|---|
| public | tabular | 58 | 9 / 49 | 25 / 33 | 0.621 | 0.569 |
| public | procedure | 64 | 9 / 55 | 21 / 43 | 0.719 | 0.672 |
| public | navigation | 77 | 27 / 50 | 52 / 25 | 0.571 | 0.675 |
| public | multihop | 53 | 2 / 51 | 24 / 29 | 0.585 | 0.547 |
| private | tabular | 10 | 0 / 10 | 3 / 7 | 0.700 | 0.700 |
| private | procedure | 10 | 4 / 6 | 6 / 4 | 0.800 | 0.400 |
| private | navigation | 10 | 7 / 3 | 9 / 1 | 0.600 | 0.900 |

**The router's error is almost entirely one-directional: it sends deep questions down the
no-retrieval path.** When it says `deep` it is right 92% of the time on both corpora. But it
catches only 58–59% of the questions that need retrieval: 85 of 205 public and 8 of 19 private
deep questions go `fast`. Of the public questions routed `fast`, 70% (85/122) needed lookup.
Even 22 of 51 deep multihop questions (which need two or more sections) get no retrieval.

### Misroute patterns

**Missed deep (label deep, router fast): 85 public, 8 private.** These questions need a lookup
but contain none of the 14 trigger words. Recurring phrasings on the public set:

- **Parameter/value questions that don't say "parameter".** "What value do I give
  COM_RC_IN_MODE…", "Which EKF2_MAG_TYPE option…", "What PWM threshold…", "How long does the
  pitch have to stay past the limit…", "How many satellites…". About 60 of the 85 hinge on an
  exact identifier, value, default or threshold. `how long` / `how many` / `how fast` /
  `by default` / `what value` / `which option` are not triggers.
- **Procedures not phrased as "steps" or "how do I".** All 15 missed procedure questions use
  wording like "Walk me through…" (3), "What's the sequence / process / workflow…" (6),
  "In what order…", "How should I…" or "What do I need to do…". Only the exact strings
  `how do i`, `steps` and `procedure` fire.
- **Table-row lookups without the word "table"** (11): the LED-status questions (S-009,
  D-001…D-004), tune meanings (D-039), the filter-latency table (C-001) and the ROS 2
  offboard flag tables (C-004…C-006). Each is one or two rows of a table.
- **Multihop questions with neutral wording** (22 of 51). Comparisons ("…versus…", "how do X
  and Y differ") and two-part questions have no trigger.
- Private (aggregate only): 3 single-row alarm-table lookups, 2 UI questions whose answer is a
  list of several items, 2 navigation questions needing a list or a specific value, 1
  multi-step procedure. The phrasings "how do you", "which/what … icon" and "what information"
  all fall through.

**Spurious deep (label fast, router deep): 10 public, 1 private.** These are rare, and cheap at
retrieval level, since retrieving for an easy question costs nothing in recall. By trigger
(public): `parameters?` 4 (questions that *mention* a parameter but ask what it means, e.g.
"What does the parameter TC_A1_TREF represent?"), `how do i` 4 (simple, well-known actions:
put a board in DFU mode, flash firmware from QGC), and `table`, `where do i`, `all`,
`settings?`, `what are the`, `list` 1 each. Private: 1, on `where do i`.

How each trigger fires on hand-deep vs hand-fast public questions: `parameters?` 51/4,
`how do i` 39/4, `settings?` 22/1, `steps?` 10/0, `what are the` 6/1, `all` 2/1,
`where do i` 2/1, `list` 2/1, `procedure` 2/0, `configure` 2/0, `table` 1/1. `which screen`
and `multi-step` never fire. The triggers are precise but cover too little.

## 2. Routing × retrieval 2×2 at retrieval level

Rows are routing policies:

- **no router**: always retrieve;
- **regex router**: `classify_query`;
- **oracle router**: the hand labels;
- **category oracle**: as above.

Columns are the two retrievers: flat 500/50 chunks over the same markdown, and proxy-pointer
(heading + first-line proxy, `title_firstline`). Both use MiniLM with an exact FAISS index, as in
[`retrieval-metrics.md`](retrieval-metrics.md).

A `fast`-routed question retrieves nothing, so **its retrieval metrics are 0**. The main tables
report the cells that way. A separate table restricts each row to its deep-routed questions.

**At retrieval level a router can only hurt.** Skipping retrieval turns every hit on a skipped
question into a miss, and it can't turn a miss into a hit. The router−no-router deltas below are
therefore ≤ 0 by construction. Any value a router has must show up in answer quality or
latency (Section 3; latency is Task 7). The CIs are still worth reading for their size.

Strict rule (the retrieved section itself is gold). Paired bootstrap, 10,000 resamples, seed 0.
✱ = the 95% CI excludes zero.

### Public corpus (n = 252, reproducible)

| Routing | retrieves for | Flat R@1 | Flat R@5 | Flat MRR | Proxy R@1 | Proxy R@5 | Proxy MRR | Flat / Proxy AllParts@5 (multihop, n = 53) |
|---|---|---|---|---|---|---|---|---|
| no router | 252 | 0.480 | 0.817 | 0.617 | 0.397 | 0.643 | 0.507 | 0.396 / 0.321 |
| regex router | 130 | 0.234 | 0.409 | 0.303 | 0.190 | 0.325 | 0.250 | 0.283 / 0.151 |
| oracle router | 205 | 0.385 | 0.667 | 0.497 | 0.329 | 0.536 | 0.422 | 0.396 / 0.321 |
| category oracle | 175 | 0.341 | 0.579 | 0.435 | 0.278 | 0.456 | 0.356 | 0.396 / 0.321 |

Proxy − flat within each row:

| Routing | Δ R@1 | Δ R@5 | Δ MRR | Δ AllParts@5 (multihop) |
|---|---|---|---|---|
| no router | -0.083 [-0.155, -0.012] ✱ | -0.175 [-0.238, -0.111] ✱ | -0.110 [-0.165, -0.055] ✱ | -0.075 [-0.226, +0.075] |
| regex router | -0.044 [-0.095, +0.008] | -0.083 [-0.131, -0.036] ✱ | -0.053 [-0.095, -0.013] ✱ | -0.132 [-0.245, -0.038] ✱ |
| oracle router | -0.056 [-0.119, +0.008] | -0.131 [-0.187, -0.075] ✱ | -0.075 [-0.124, -0.025] ✱ | -0.075 [-0.226, +0.075] |

Routing − no router within each column:

| Routing | Retriever | Δ R@1 | Δ R@5 | Δ MRR | Δ AllParts@5 (multihop) |
|---|---|---|---|---|---|
| regex router | flat | -0.246 [-0.302, -0.194] ✱ | -0.409 [-0.472, -0.349] ✱ | -0.314 [-0.368, -0.263] ✱ | -0.113 [-0.208, -0.038] ✱ |
| regex router | proxy | -0.206 [-0.258, -0.159] ✱ | -0.317 [-0.377, -0.262] ✱ | -0.257 [-0.309, -0.208] ✱ | -0.170 [-0.283, -0.075] ✱ |
| oracle router | flat | -0.095 [-0.135, -0.060] ✱ | -0.151 [-0.194, -0.107] ✱ | -0.121 [-0.160, -0.084] ✱ | 0 (no multihop hit skipped) |
| oracle router | proxy | -0.067 [-0.099, -0.040] ✱ | -0.107 [-0.147, -0.071] ✱ | -0.086 [-0.119, -0.055] ✱ | 0 (no multihop hit skipped) |

Restricted to the questions each policy retrieves for:

| Routing | n | Flat R@5 | Flat MRR | Proxy R@5 | Proxy MRR | Δ MRR proxy − flat |
|---|---|---|---|---|---|---|
| no router | 252 | 0.817 | 0.617 | 0.643 | 0.507 | -0.110 [-0.165, -0.055] ✱ |
| regex router | 130 | 0.792 | 0.588 | 0.631 | 0.485 | -0.103 [-0.181, -0.023] ✱ |
| oracle router | 205 | 0.820 | 0.611 | 0.659 | 0.518 | -0.092 [-0.152, -0.033] ✱ |

### Private corpus (n = 30, agent-drafted gold, not reproducible)

| Routing | retrieves for | Flat R@1 | Flat R@5 | Flat MRR | Proxy R@1 | Proxy R@5 | Proxy MRR |
|---|---|---|---|---|---|---|---|
| no router | 30 | 0.600 | 0.867 | 0.716 | 0.500 | 0.700 | 0.569 |
| regex router | 12 | 0.167 | 0.333 | 0.227 | 0.167 | 0.300 | 0.213 |
| oracle router | 19 | 0.367 | 0.533 | 0.433 | 0.267 | 0.400 | 0.313 |
| category oracle | 20 | 0.400 | 0.567 | 0.460 | 0.300 | 0.433 | 0.346 |

| Contrast | Δ R@1 | Δ R@5 | Δ MRR |
|---|---|---|---|
| proxy − flat, no router | -0.100 [-0.233, +0.033] | -0.167 [-0.300, -0.033] ✱ | -0.147 [-0.275, -0.034] ✱ |
| proxy − flat, regex router | +0.000 [-0.100, +0.100] | -0.033 [-0.100, +0.000] | -0.014 [-0.067, +0.047] |
| proxy − flat, oracle router | -0.100 [-0.233, +0.033] | -0.133 [-0.267, -0.033] ✱ | -0.119 [-0.245, -0.007] ✱ |
| regex router − none, flat | -0.433 [-0.600, -0.267] ✱ | -0.533 [-0.700, -0.367] ✱ | -0.489 [-0.656, -0.322] ✱ |
| regex router − none, proxy | -0.333 [-0.500, -0.167] ✱ | -0.400 [-0.567, -0.233] ✱ | -0.356 [-0.522, -0.189] ✱ |
| oracle router − none, flat | -0.233 [-0.400, -0.100] ✱ | -0.333 [-0.500, -0.167] ✱ | -0.283 [-0.433, -0.133] ✱ |
| oracle router − none, proxy | -0.233 [-0.400, -0.100] ✱ | -0.300 [-0.467, -0.133] ✱ | -0.256 [-0.411, -0.111] ✱ |

Restricted to deep-routed questions:

| Routing | n | Flat MRR | Proxy MRR | Δ MRR proxy − flat |
|---|---|---|---|---|
| no router | 30 | 0.716 | 0.569 | -0.147 [-0.275, -0.034] ✱ |
| regex router | 12 | 0.568 | 0.533 | -0.035 [-0.163, +0.122] |
| oracle router | 19 | 0.683 | 0.494 | -0.189 [-0.373, -0.014] ✱ |

### What the retrieval 2×2 shows

- **Routing is the larger effect, by a wide margin.** On the public set, the regex router
  costs flat −0.31 MRR and proxy −0.26 MRR. Switching retriever (proxy − flat, no router) costs
  −0.11. Private: −0.49 / −0.36 for routing vs −0.15 for the retriever. The router discards
  retrieval on 48% (public) and 60% (private) of questions, and most of those (70% and 44%)
  needed it.
- **Even a perfect router loses recall.** With the hand labels as the router, MRR drops by
  0.09–0.12 (public) and 0.26–0.28 (private), because some questions that a model "could
  plausibly answer" are still ones the retriever gets right. Whether skipping them is worth it
  depends on whether the bare model actually answers them (Section 3).
- **The router hides the retriever gap on the private set.** In the published configuration the
  proxy − flat gap shrinks from −0.147 to −0.014, for two reasons. Zeroing out 18 questions
  removes their deltas. And the regex router happened to send the questions where proxy-pointer
  fails worst (single-row lookups in the large alarm table, which the proxy can't see) down the
  no-retrieval path. On the 12 questions the router retrieves for, proxy − flat is −0.035
  [−0.163, +0.122]. On the 19 a sensible router would retrieve for, it is −0.189 [−0.373,
  −0.014] ✱. On the public set there is no such luck: the router's deep subset shows the same
  gap as the full set (−0.103 vs −0.110 MRR).
- **The retriever ordering doesn't change with routing.** Flat ≥ proxy in every row under the
  strict rule on both corpora. Under the context rule (what the generator receives: proxy node +
  children), the two tie in every row on both corpora. For example, public no-router MRR is
  −0.017 [−0.073, +0.040], and on oracle-routed questions it is +0.013 [−0.048, +0.073]. The
  per-row context-rule tables are in `artifacts/public/task6/router_eval_tables.md`.

## 3. Answer-level 2×2 (private)

**(private, n = 30.)** All four cells use one setup: TinyLlama-1.1B-Chat, one prompt template,
the same 1,200-char context cap, `max_new_tokens=200`, greedy decoding. Flat context = top-4
flat_md chunks; proxy context = top-1 node + direct children. Fast-routed questions get the bare
LLM answer, reused across the router cells (79 generations total; `router_eval.py contexts |
generate | answers`). Score = cosine(MiniLM(answer), MiniLM(reference)), the published metric,
with all its weaknesses ([`metrics.md`](metrics.md)). Paired bootstrap, 10k resamples, seed 0;
✱ = CI excludes zero.

| Routing | Flat 500/50 | Proxy-pointer | Proxy − flat [95% CI] |
|---|---|---|---|
| Always retrieve | **0.701** | 0.652 | −0.049 [−0.096, −0.002] ✱ |
| Regex router | 0.675 | 0.668 | −0.007 [−0.039, +0.023] |
| Oracle router (hand labels) | 0.701 | 0.672 | −0.029 [−0.073, +0.014] |

| Router − always retrieve | Flat | Proxy-pointer |
|---|---|---|
| Regex router | −0.026 [−0.064, +0.009] | +0.015 [−0.014, +0.047] |
| Oracle router | −0.000 [−0.023, +0.021] | +0.020 [−0.001, +0.045] |

**Does the fast route help?** We compare answers on exactly the questions each router skips:

| Skipped questions | n | Bare LLM | Flat | Proxy-pointer | Bare − flat | Bare − proxy |
|---|---|---|---|---|---|---|
| Regex router's fast set | 18 | 0.716 | 0.760 | 0.691 | −0.044 [−0.106, +0.017] | +0.026 [−0.027, +0.077] |
| Oracle's fast set | 11 | 0.753 | 0.754 | 0.699 | −0.001 [−0.064, +0.059] | +0.054 [−0.004, +0.114] |

What this shows:

- **With no router, proxy-pointer answers are worse than flat answers** (−0.049, CI excludes 0),
  consistent with the retrieval-level result.
- **Adding the regex router closes that gap to −0.007.** It does this by replacing proxy-pointer
  retrieval with *no* retrieval on 18 questions. On those questions the bare LLM scores slightly
  above proxy-pointer context, and below flat context. So the router helps proxy-pointer only
  because proxy-pointer's context is, on those questions, no better than nothing; with flat
  retrieval the router costs −0.026. This is the mechanism behind the published "hybrid ≈
  vanilla" result.
- **No answer-level routing effect is significant.** Under this metric and generator, skipping
  retrieval neither clearly helps nor clearly hurts answer quality. Its only demonstrated
  benefit would be latency, and [`latency-memory.md`](latency-memory.md) shows retrieval costs
  ~3.5 ms against ~12 s of generation, so even that is negligible.
- **Answer to the plan's question.** The published hybrid's behaviour is mostly routing: the
  router decides whether there is any context at all for 60% of questions. And the routing
  decision happened to hide proxy-pointer's weakness. Proxy-pointer retrieval itself is the
  worse retriever in every cell, at both retrieval and answer level.

## Limitations

- One labeller, one pass. There is no second annotator for the router labels. The category
  oracle is a crude, labeller-independent alternative, and the router's accuracy against it
  is the same on public (0.623) and similar on private (0.667).
- The public questions were written by LLM agents who saw the gold section. Their phrasing
  ("Walk me through…", "What's the sequence…") may differ from how real operators ask. That
  could make the regex router look better or worse than in deployment.
- Retrieval-level numbers are necessarily lopsided against any router. They show what routing
  costs, not what it buys.
- The private set is 30 questions (10 per category) with agent-drafted gold. Per-category
  router accuracies rest on 10 questions each.
