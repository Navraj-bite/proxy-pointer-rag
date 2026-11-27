# Retrieval metrics

`retrieval_metrics.py` scores retrieval itself instead of using answer cosine as a stand-in.
Each retriever has to return a ranked list of **section nodes**, the same `DocumentNode`s that
`document_tree_builder.py` produces. The script compares that list with hand-labelled gold
sections and reports Recall@K, MRR and nDCG@K. The same metric code runs for every system.

> **Gold-label status: agent-drafted, not yet human-verified.** An AI agent drafted the
> private-corpus gold labels by reading the questions, the reference answers and the document
> tree. Nobody has checked them by hand yet. Every number below depends on those labels. A
> verification sheet with one row per question has been generated next to the private data
> (see [Gold labels](#gold-labels)). When it has been filled in, rerun the script and replace
> the tables.

## Systems compared

These comparisons involve no router and no generator. They measure retrieval only.

| System | What is indexed | How hits become sections |
|---|---|---|
| `proxy_pointer` | One embedding per heading node. The proxy text is identical to the one `proxy_pointer_indexer.py` builds: the heading plus the first non-table body line. The rebuilt index matches `pp_faiss.index` vector for vector (min cosine 0.9999999, same node order). | Nodes are ranked by proxy score. Every node is a section. |
| `flat_md` (primary baseline) | `RecursiveCharacterTextSplitter(500, 50)` run over the **same markdown files the tree is built from**. | Each chunk maps to the section(s) it overlaps (rule below). |
| `legacy_pypdf` (secondary) | The original `faiss.index` + `chunks.pkl`, built from pypdf text, exactly as the Week-2 benchmark used them. | Each chunk is fuzzy-mapped to markdown sections (rule below). The mapping is approximate. |

`flat_md` is the primary baseline because it removes a confound. The old benchmark compared
proxy-pointer retrieval over LlamaParse markdown against chunk retrieval over pypdf text. The
two arms therefore saw different source text, not just different retrieval strategies. One
example: pypdf extracts none of the rows of the main alarm/message table, because in the PDF it is a screenshot.

## Relevance rules

Let G be the set of gold node ids for a question.

- **strict (primary):** a retrieved section *s* is relevant iff *s* ∈ G.
- **context (secondary):** *s* is relevant iff the context the system would actually pass to
  the generator for *s* contains a gold section.
  - proxy-pointer: `context_assembler.query` returns the hit node plus its **direct
    children**. So ctx(*s*) = {*s*} ∪ children(*s*), and a hit on a gold node's parent counts.
  - flat / legacy: the context is the chunk itself, and the chunk is already mapped to the
    section(s) it overlaps. So ctx(*s*) = {*s*}, and the context rule is identical to strict.
- **context_parent (sensitivity only):** applies the proxy context function to every system,
  so that parent credit is symmetric. It is reported in the CSVs. It changes no conclusion.

## Metrics

All metrics are computed per question and then averaged.

- **Recall@K** is a *hit rate*: 1 if at least one relevant section is in the top K, otherwise 0.
  Gold lists hold **alternative** sections that each contain the answer, for example the same
  table in both manuals. "Any gold in top K" is therefore the right notion. It is not the
  fraction of gold sections found.
- **MRR** is 1/rank of the first relevant section within the top 10 sections, and 0 if there is
  none.
- **nDCG@5** uses binary relevance. IDCG assumes min(R, 5) relevant sections at the top. R is
  the number of distinct indexable sections that are relevant under the rule: |G| for strict,
  and gold plus gold parents for proxy under the context rule. This means nDCG is **not
  comparable across rules**. Compare systems within one rule only.
- **Paired bootstrap:** the per-question delta (proxy − baseline) is resampled over questions,
  10,000 times with seed 0, and the 95% CI is taken from the percentiles. This is done for MRR
  and Recall@5.

## Chunk → section mapping

**flat_md (exact).** A chunk's character span `[start, start+len)` in its file is converted to a
line range. Heading nodes split each file into non-overlapping line spans
(`start_line..end_line`): every line belongs to the most recent heading, which is the deepest
node that contains it. The chunk maps to **every** content node whose span it overlaps,
ordered by overlapping characters (largest first, ties in document order). Lines before a
file's first heading belong to the synthetic `[FILE]` node and map to nothing.

The system's ranked section list is built by concatenating the mapped sections chunk by chunk,
in score order, and keeping only the first occurrence of each section. A chunk that straddles
a boundary therefore fills several consecutive ranks. K always counts sections, not chunks. On
the private corpus, 188 of 1,195 chunks straddle a boundary and 0 are unmapped.

**legacy_pypdf (fuzzy).** pypdf text and LlamaParse markdown differ, with broken words,
reordered tables and image descriptions, so legacy chunks are matched by word 4-gram shingles.

1. Shingles that occur in more than 5 sections are dropped. These are page headers and footers.
2. A greedy cover repeatedly picks the section that explains the most not-yet-explained
   shingles. A section is picked only if it adds at least 10%.
3. The chunk is mapped only if the picked sections together explain at least 50% of its
   shingles.
4. Sections tied within 0.05 of the best single-section share are also kept. This covers
   duplicated content.

Measured quality on the private corpus:

- **Coverage:** 763 of 799 legacy chunks mapped (95.5%).
- **Precision check on markdown chunks** (where the true mapping is known): section precision
  0.94, recall 0.92.
- **Independent check:** 354 of 383 mapped legacy chunks that carry a `Page N/359` footer land
  in a section spanning page N±1 (92.4%).
- **Effect on rankings:** 4 of 30 questions have an unmapped chunk in their legacy top 10.

Treat the legacy numbers as approximate.

## Gold labels

Gold labels are stored by **(doc file, heading path from the file root)**. They are not stored
by node id: ids like `n0042` depend on build order. The heading paths are resolved to node ids
at run time. If a path matches zero nodes or more than one node, the script raises
`GoldResolutionError`. The run fails loudly rather than silently scoring against the wrong
section.

Format (JSONL, one record per question id; every question must have a record):

```json
{"id": 2,
 "gold": [{"doc": "widget_manual.md", "heading_path": ["2. OPERATION", "2.1. Start-up procedure"]}],
 "label_source": "agent-drafted",
 "verified": false,
 "notes": "free text: why these sections, near-misses deliberately excluded"}
```

- `doc` is the basename of one of the `--md` files.
- `heading_path` lists heading titles exactly as they appear in the markdown, without the
  leading `#`s. Comparison collapses whitespace. Note that the path follows the tree *as
  parsed*: if a converter mis-levels a heading, the path follows the parsed nesting.
- When sibling headings repeat, a path element can be written as
  `{"title": "...", "occurrence": n}` (1-based) to pick the n-th one.
- `label_source` is `agent-drafted` or `human`. `verified` becomes `true` only after a human has
  checked the label.
- Labelling rule: gold contains the sections that hold most of the reference answer's facts.
  Sections with only fragments, or with related but different content, are left out and named
  in `notes`.

A runnable example with fake headings lives in
[`examples/retrieval_metrics/`](../examples/retrieval_metrics/). Run it from that directory:

```bash
python3 ../../retrieval_metrics.py --md widget_manual.md glossary.md \
    --questions questions.example.jsonl --gold gold_labels.example.jsonl --out-dir out
```

The private-corpus gold file and its verification sheet (`gold_labels.jsonl`,
`gold_verification_sheet.csv`) are derived from the corpus and are git-ignored. The sheet has
one row per question with: the question, the reference answer, the gold heading paths, the
first 300 characters of each gold section, the agent's notes, and blank `verified` and
`verifier_notes` columns.

## Running it

```bash
cd /dir/with/corpus
python3 /path/to/retrieval_metrics.py \
    --md machine_overview_troubleshooting.md operator_manual.md \
    --questions benchmark_questions.jsonl --gold gold_labels.jsonl \
    --k 1 3 5 --out-dir retrieval_results \
    --verification-sheet gold_verification_sheet.csv \
    --legacy-chunks chunks.pkl --legacy-index faiss.index      # optional
```

Outputs, all git-ignored:

- `retrieval_per_question.csv`: one row per question × system × rule, including the top-5
  section labels.
- `retrieval_summary.csv`: overall and per-category means with n.
- `retrieval_summary.json`: config, aggregates, bootstrap CIs, legacy mapping coverage, and the
  environment (device, model id, seed, library versions, date, git commit).

`--check-gold-only` resolves the labels and writes the sheet without loading any model.

### Reuse API

```python
from retrieval_metrics import SectionRetriever, build_proxy_index, PROXY_FUNCS

r = SectionRetriever(["a.md", "b.md"], device="cpu")
r.rank_sections_proxy(q, k=5)                          # shipped proxy (heading + first line)
r.rank_sections_proxy(q, k=5, proxy_mode="title")      # heading only
r.rank_sections_proxy(q, k=5, proxy_mode=lambda node: node.heading_title + node.body_content[:300])
r.rank_sections_flat(q, k=5, chunk_size=500, overlap=50)
build_proxy_index(tree, proxy_fn, embedder)            # -> (faiss index, node_ids, proxy_texts)
```

`init(md_files)` plus the module-level `rank_sections_proxy(question, k, proxy_mode=...)` and
`rank_sections_flat(question, k, chunk_size, overlap)` do the same through a default retriever.
For scoring, use `resolve_gold`, `score_ranking` and `paired_bootstrap`. No existing module was
modified.

## Results: private corpus (not reproducible from a clean clone)

> These numbers come from the proprietary manuals, which are not in this repo. They are
> aggregates only. **Gold labels: agent-drafted, not human-verified.** n = 30 questions
> (10 per category), `all-MiniLM-L6-v2` on CPU, FAISS `IndexFlatIP`, seed 0.
> Run on 2026-11-25 with `retrieval_metrics.py` at commit `0946a39` (the only uncommitted
> file was this doc). The aggregate outputs are identical to the pre-commit run.

### Strict relevance (primary)

| System | Category | n | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| proxy-pointer | overall | 30 | 0.500 | 0.633 | 0.700 | 0.569 | 0.574 |
| proxy-pointer | tabular | 10 | 0.100 | 0.300 | 0.500 | 0.239 | 0.280 |
| proxy-pointer | procedure | 10 | 0.800 | 0.800 | 0.800 | 0.800 | 0.785 |
| proxy-pointer | navigation | 10 | 0.600 | 0.800 | 0.800 | 0.667 | 0.656 |
| flat_md | overall | 30 | **0.600** | **0.833** | **0.867** | **0.716** | **0.716** |
| flat_md | tabular | 10 | 0.400 | 0.800 | 0.900 | 0.581 | 0.618 |
| flat_md | procedure | 10 | 0.800 | 0.800 | 0.800 | 0.800 | 0.761 |
| flat_md | navigation | 10 | 0.600 | 0.900 | 0.900 | 0.767 | 0.768 |
| legacy_pypdf ≈ | overall | 30 | 0.467 | 0.767 | 0.833 | 0.609 | 0.619 |
| legacy_pypdf ≈ | tabular | 10 | 0.100 | 0.600 | 0.700 | 0.320 | 0.371 |
| legacy_pypdf ≈ | procedure | 10 | 0.700 | 0.700 | 0.800 | 0.725 | 0.704 |
| legacy_pypdf ≈ | navigation | 10 | 0.600 | 1.000 | 1.000 | 0.783 | 0.781 |

### Context-contains-gold relevance (secondary)

Only the proxy-pointer rows differ from the strict table, because flat and legacy contexts are
the chunk itself.

| System | Category | n | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|---|
| proxy-pointer | overall | 30 | 0.500 | 0.733 | 0.800 | 0.618 |
| proxy-pointer | tabular | 10 | 0.100 | 0.400 | 0.600 | 0.262 |
| proxy-pointer | procedure | 10 | 0.800 | **1.000** | **1.000** | **0.883** |
| proxy-pointer | navigation | 10 | 0.600 | 0.800 | 0.800 | 0.710 |
| flat_md | overall | 30 | 0.600 | 0.833 | 0.867 | 0.716 |

### Paired bootstrap, proxy-pointer − flat_md (95% CI, n = 30)

| Rule | Metric | Δ | 95% CI | Includes 0? |
|---|---|---|---|---|
| strict | MRR | −0.147 | [−0.275, −0.034] | no |
| strict | Recall@5 | −0.167 | [−0.300, −0.033] | no |
| context | MRR | −0.098 | [−0.233, +0.026] | yes |
| context | Recall@5 | −0.067 | [−0.233, +0.100] | yes |

Against legacy_pypdf (approximate mapping): strict MRR −0.041 [−0.159, +0.084], strict R@5
−0.133 [−0.267, −0.033], context MRR +0.009 [−0.109, +0.134], context R@5 −0.033 [−0.200, +0.100].

Per category (n = 10, so the intervals are wide): the tabular strict delta excludes 0 (R@5 −0.40
[−0.70, −0.10]). Procedure strict is an exact tie on every question. Procedure under the context
rule favours proxy (R@5 +0.20 [0.00, +0.50]; the lower bound touches 0).

### What this says

- **Under the primary rule, proxy-pointer retrieval is worse than flat chunking over the same
  markdown,** and the 95% CIs for MRR and R@5 exclude zero. Most of the gap comes from
  **tabular** questions. Six of the ten are answered by one ~4 kB alarm/message table, and the
  proxy (heading + first line) cannot see table rows. Proxy-pointer ranks that section first
  only for Q3. For the three single-alarm lookups (Q2, Q9, Q10) it misses the section entirely
  within the top 10, because the per-alarm troubleshooting nodes outrank it. It also misses a
  parameter table that sits deep in a long management section (Q4). This contradicts the README's claim that proxy-pointer is
  "the right tool for tables".
- **Procedures:** a tie under strict. Under the context rule proxy-pointer reaches R@3 = 1.0, ahead
  of flat (0.8), because hitting the parent chapter heading delivers the three sub-sections holding the steps
  as its children. This is the only place the structure-aware design visibly helps retrieval.
- **Navigation:** flat is slightly better (strict MRR Δ −0.10 [−0.20, −0.02]). Proxy misses the bottom-right information-field
  questions (Q23, Q24) in its top 5. The answer sits mid-body in the taskbar section.
- **The legacy pypdf baseline is roughly level with proxy-pointer on MRR.** Its text holds no
  rows of that alarm table: the table is a screenshot in the PDF. Its tabular "hits" (Q1, Q3, Q7)
  land on the right section through surrounding prose, so for this system a section hit does
  not mean the answer reached the generator. Once the source text is equalised, flat chunking
  moves ahead. The old answer-cosine comparison mixed two effects: source-text quality and
  retrieval strategy.

### Sensitivity: heading-only proxy

With `--proxy-mode title`, the first body line is removed from the proxy. The
proxy-pointer strict scores rise to R@1 0.667, R@5 0.733, MRR 0.700. Against flat_md: strict
MRR −0.016 [−0.173, +0.127], context MRR +0.045 [−0.131, +0.217], strict R@5 still −0.133
[−0.267, −0.033]. On this corpus the first-line augmentation **hurts** ranking. This is a single
exploratory run, not a tuned result.

### Diagnostic: how much of the gold section reaches the generator

This compares what the two benchmarked pipelines actually pass to the generator:

- proxy-pointer: the top-1 node plus its children.
- flat: the top-4 chunks, joined with `---`.

The table shows the share of the best gold section's characters present in that context. "Capped"
applies the 1,200-character truncation that `hybrid_intent_router._deep_answer` and
`run_day10_benchmark.vanilla_answer` both apply.

| System | overall | tabular | procedure | navigation |
|---|---|---|---|---|
| proxy-pointer, uncapped / capped | 0.500 / 0.341 | 0.100 / 0.029 | 0.800 / 0.541 | 0.600 / 0.454 |
| flat_md top-4, uncapped / capped | 0.282 / 0.257 | 0.168 / 0.149 | 0.368 / 0.351 | 0.308 / 0.270 |

Section-level hit metrics flatter flat chunking in one respect. A flat "hit" means only that some
≤500-character window of the gold section was retrieved, not that the window holds the
answer. The proxy pipeline returns the whole section. On the share of the gold section
delivered, proxy-pointer is ahead for procedure and navigation, but the 1,200-character cap
removes about a third of that advantage.

Caveat: covering a *section* is not the same as covering the *answer*. A short answer can be
fully inside one well-placed chunk. Treat this as a diagnostic, not a headline metric.

## Limitations

- The gold labels are agent-drafted and unverified (see above). Q29 is a flagged judgment call:
  a near-identical paragraph under a sibling monitoring screen is not counted as gold.
- n = 30, 10 per category. Per-category deltas mostly have CIs that include zero.
- Recall@K is any-hit. Q11, a multi-section procedure, counts as found when any one of its three
  sections is retrieved.
- These metrics evaluate sections, not answer spans (see the diagnostic above).
- The heading structure produced by LlamaParse is noisy (stray level-1 headings created from
  callouts and page logos). Gold paths follow the parsed tree. If the parser changes, labels fail loudly
  rather than drift.
