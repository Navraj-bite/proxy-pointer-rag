# Public corpus: PX4 User Guide (v1.17.0 subset)

Every number in the main README comes from a proprietary equipment manual that cannot be
committed. This page describes a public corpus that anyone can fetch, verify and convert
from a clean clone, so the pipeline and its benchmarks can be re-run independently.

## What it is and why it was chosen

The **PX4 User Guide** is the operator and configuration manual for the PX4 open-source
autopilot (drones and other uncrewed vehicles). The subset used here covers vehicle setup,
sensor calibration, safety/failsafe configuration, flying, flight modes and advanced
configuration. It reads like an equipment manual: numbered calibration procedures
(compass, ESC, radio), parameter tables with defaults, LED/tune status tables, preflight
error explanations and troubleshooting sections. That is the regime this retriever is built
for, and it is close to the private corpus (an operator manual plus a troubleshooting manual).

Why this one over the other candidates:

| Candidate | Why not chosen |
| --- | --- |
| Kubernetes docs (CC BY 4.0) | Mostly procedures, very few tables (11 pipe-table lines across all 71 `tasks/administer-cluster` pages), heavy Hugo shortcodes, some content pulled from other files |
| PostgreSQL manual | DocBook HTML; parameters are `<dl>` lists rather than headings, so sections get very long; the conversion is more complex |
| FAA handbooks / IRS publications | PDF only, so no free, deterministic conversion to headings |
| **PX4 User Guide** | Native markdown, CC BY 4.0, pinned to a release tag/commit, about 0.5 MB after conversion, many tables **and** procedures |

## Exact version

| | |
| --- | --- |
| Upstream repo | https://github.com/PX4/PX4-Autopilot (the docs live in `docs/en/`) |
| Release tag | `v1.17.0` (annotated tag object `a5eb12d2ab591251faa009f76b2685b8cc64405d`) |
| **Pinned commit** | **`d6f12ad1c4f70ad3230afd7d86e971421e02fef4`** (2026-04-24) |
| Files | 98 markdown pages + `docs/LICENSE`, listed with sha256 in [`scripts/px4_docs_v1.17.0.sha256`](../scripts/px4_docs_v1.17.0.sha256) |
| Download | `https://raw.githubusercontent.com/PX4/PX4-Autopilot/<commit>/<path>`, one file at a time. The URL contains the commit SHA, so the content cannot change |

When the manifest was made, each downloaded file was also checked against the git blob id
from `git ls-tree d6f12ad1…`. All 99 matched, so the sha256 manifest describes exactly
that commit.

## Licence

- **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
  `docs/LICENSE` in the upstream repo starts with "CC BY 4.0" and links to
  https://creativecommons.org/licenses/by/4.0/legalcode. (The PX4 *code* is BSD-3-Clause;
  only the docs are used here.)
- **Redistribution is allowed**, including of adapted material, for any purpose
  (Section 2(a)(1)).
- **Attribution is required** (Section 3(a)): name the creator (PX4 Dev Team / Dronecode),
  keep the licence notice and link, link to the source, and say what was changed.
- This repo does **not** commit the corpus text. `scripts/fetch_corpus.sh` downloads it, and
  the converter writes `corpus/ATTRIBUTION.md` (source, commit, licence, list of changes)
  and copies the upstream licence to `corpus/LICENSE-PX4-DOCS` next to the converted files.
  `questions/public_seed.jsonl` is committed. It holds short facts and paraphrases taken
  from the guide, attributed here: *PX4 User Guide, © PX4 Dev Team, CC BY 4.0*.

## Subset

These 7 section directories of `docs/en/`, in this order:

`getting_started`, `config`, `config_mc`, `flying`, `flight_modes`, `flight_modes_mc`,
`advanced_config`

Excluded:

- `advanced_config/parameter_reference.md`: an auto-generated 1.6 MB dump of all parameters.
  It is a reference table, not manual prose, and it would make up 75% of the corpus.
- Pages that are only a `<Redirect …/>` stub (17 of them, mostly in `flight_modes/`).
  They are fetched and verified, then skipped because they have no body text.
- `config/_autotune.md` is a partial. It is not a page of its own and is inlined into
  `autotune_mc.md` / `autotune_fw.md` (see below).

What is kept: 80 pages.

## Conversion (`prepare_public_corpus.py`)

The converter is stdlib-only and deterministic: two runs give byte-identical output. It uses
no network and no paid API. It only reads the downloaded files and runs as `python3 -I`.

**Layout:** there is one output file per section directory, `corpus/px4_<section>.md`, so
each section is one `[FILE]` node in the `DocumentTree`. Each page keeps its own `#` title as
a level-1 heading, and the page's `##`…`#####` headings keep their relative depth. Pages are
ordered `index.md` first, then alphabetically. So the tree is
ROOT → section file → page (`#`) → page sections (`##`, `###`, …).

**Cleaning:**

- YAML front matter is removed. Its `title` would be used for a page with no `#`; no page
  in this subset needed that.
- `<!--@include: _partial.md-->` is inlined. Vue `<div v-if="$frontmatter.frame === '…'">`
  / `v-else-if` / `v-else` blocks are evaluated against the including page's front matter.
  So `autotune_mc.md` renders the multicopter text and `autotune_fw.md` the fixed-wing text,
  as on the website. Any other Vue condition makes the converter fail rather than guess.
- These are removed: HTML comments, `<script>`, `<iframe>`, `<audio>`, `<lite-youtube>`,
  `<Redirect>`, images and `<a id>` anchors. A line holding only version badges is dropped;
  a badge inside a line becomes `(PX4 v1.14)`.
- Links and reference links are reduced to their text, and link definitions are dropped.
  Code spans are protected throughout: nothing inside backticks is changed.
- `:::tip` / `::: info` / `:::warning` containers become a bold label (`**Tip:**`,
  `**Note:**`, `**Warning:**`) joined to the first line of their paragraph. The closing
  `:::` is dropped.
- `{#custom-id}` heading anchors and emphasis markers in headings are removed.
- Pipe tables are normalised to `| a | b |` rows with `| --- |` delimiters and trimmed
  cells. Upstream pads cells with hundreds of spaces, and some tables have no outer pipes.
- Numbered and bulleted lists are left exactly as written. PX4 numbers every step `1.`,
  which still matches `NUMBERED_STEP_RE`.
- Skipped heading levels inside a page are closed up: an `###` directly under `#` becomes
  `##`. This happened 29 times. It does not change any parent/child link, because the tree
  builder already attaches to the nearest shallower heading.
- Inside fenced code blocks, a line starting with `#` gets one leading space (7 lines,
  all shell comments). Without it, `document_tree_builder.py` would read them as headings.
  Nothing else inside code is touched.

## Tree statistics

Produced by `PPRAG_CORPUS=public python3 check_corpus_tree.py`:

| Stat | Value |
| --- | --- |
| Markdown | 513,931 bytes, 8,928 lines, 7 files, 80 pages (private corpus: ~460 KB) |
| Nodes | 662 = ROOT + 7 `[FILE]` + **654 content nodes** (all indexed as proxies) |
| Integrity errors (`validate_tree`) | 0 |
| Depth histogram | h1 80 · h2 303 · h3 216 · h4 50 · h5 5 |
| Nodes per file | getting_started 76 · config 185 · config_mc 64 · flying 61 · flight_modes 27 · flight_modes_mc 76 · advanced_config 165 |
| Empty body | 21: 19 are container headings whose content is in children, and 2 are leaves whose only upstream content was an audio clip or a video |
| Table-bearing nodes | 53 |
| Numbered-step nodes | 78 |
| Body length median / p90 / max | 580 / 1,679 / 4,345 chars |
| Nodes with body > 8,000 chars | **0** |
| Assembled context (node + direct children, as `context_assembler.query` returns it) median / p90 / max | 834 / 3,599 / 13,961 chars; 9 nodes over 8,000 (all page-level `#` nodes with many children) |
| Very short headings (< 3 chars) | 0 |
| Duplicate heading titles | 47 titles shared by 153 nodes (`Parameters` ×15, `Further Information` ×14, `See Also` ×13, `Overview` ×11, …) |
| Duplicate (doc, heading_path) | 0, so every gold label is unambiguous |
| Skipped heading levels | 0 (after closing up) |

Note for retrieval analysis: the proxy is the heading plus the first body line, without the
parent path. So repeated headings such as `Parameters` (15 different flight-mode pages)
are hard cases for the proxy index. The autotune MC/FW pages are also near-duplicates of
each other, and flat chunks from them come back twice. Both are real features of
documentation sets, and the corpus keeps them on purpose.

Spot checks by eye against the upstream source (`--show 6`, plus the seed-question sections):
the LED status table, the Position-mode parameter table, the autotune troubleshooting
entry (Vue-conditional multicopter text chosen correctly), the ESC calibration steps,
the tunes list and the RTL types list all match the source text, apart from the
documented markup removal.

## Seed questions

[`questions/public_seed.jsonl`](../questions/public_seed.jsonl) has 12 questions: 4
tabular, 4 procedure, 4 navigation. They use the private set's schema
(`id`, `question`, `category`, `reference_answer`) plus:

- `gold`: a list of `{"doc", "heading_path"}`. `doc` is the tree's `doc_id`, i.e. the path
  as listed in `corpus_config.py` (for example `corpus/px4_config.md`). `heading_path` is
  the list of heading titles from the page `#` down to the section that holds the answer.
  Resolve it with `document_tree_builder.find_by_heading_path(tree, doc, heading_path)`.
- `keywords`: answer keywords. The benchmark's keyword-match "retrieval correct" check uses
  them instead of the private `RETRIEVAL_KEYWORDS` table. `check_corpus_tree.py` checks
  that each keyword appears in the gold section.

`PPRAG_CORPUS=public python3 check_corpus_tree.py --questions questions/public_seed.jsonl`
confirms that every gold label resolves to exactly one node and contains its keywords
(problems: 0).

## Configuration interface

`corpus_config.py` is selected by the environment variable **`PPRAG_CORPUS`**:

| | `private` (default, unchanged behaviour) | `public` |
| --- | --- | --- |
| `input_files` (tree + markdown baseline) | `machine_overview_troubleshooting.md`, `operator_manual.md` | `corpus/px4_<section>.md` ×7 |
| `baseline_source` | `pdf` (pypdf over `resources/*.pdf`) | `markdown` (the same files as the tree) |
| `pp_index_file` / `pp_meta_file` / `pp_tree_file` | `pp_faiss.index` / `pp_metadata.pkl` / `pp_tree.pkl` | `artifacts/public/…` same names |
| `flat_index_file` / `flat_chunks_file` | `faiss.index` / `chunks.pkl` | `artifacts/public/…` |
| `questions_file` | `benchmark_questions.jsonl` | `questions/public_seed.jsonl` |
| `results_csv` / `report_md` / `retrieval_csv` | `proxy_rag_vs_vanilla_rag_*.{csv,md}` | `artifacts/public/…` |
| `assistant_subject` (system prompts) | `Serac filling machines` | `the PX4 Autopilot flight stack (drones and other uncrewed vehicles)` |

Use `from corpus_config import get_config; cfg = get_config()`. Calling
`get_config("public")` ignores the env var. With the default, every module's
paths and prompts are byte-identical to what they were before. `corpus/` and `artifacts/`
are gitignored.

Benchmark flags added to `run_day10_benchmark.py`: `--retrieval-only` (no generation; writes
`retrieval_csv` with the proxy-pointer hit, the flat-chunk line spans and gold hits),
`--limit N` and `--questions PATH`. `naive_rag_baseline.py` gained `--source pdf|markdown`
and `--rebuild`.

## Reproduce from a clean clone

```bash
git clone <this repo> && cd proxy-pointer-rag
pip install sentence-transformers faiss-cpu langchain-text-splitters pypdf numpy
# (transformers + torch only for the generation step)

./scripts/fetch_corpus.sh                       # download 99 files, verify sha256, convert -> corpus/

export PPRAG_CORPUS=public
python3 check_corpus_tree.py --questions questions/public_seed.jsonl   # tree stats + gold check
python3 proxy_pointer_indexer.py                # -> artifacts/public/pp_faiss.index (654 vectors)
python3 naive_rag_baseline.py --build-only      # -> artifacts/public/faiss.index (1423 chunks)
python3 run_day10_benchmark.py --retrieval-only # retrieval for both configs, no LLM
OMP_NUM_THREADS=1 python3 run_day10_benchmark.py --limit 2   # generation smoke test (TinyLlama, CPU)
```

`OMP_NUM_THREADS=1` is needed in this environment (faiss-cpu 1.13 with torch 2.8 on macOS
arm64). When faiss is imported before torch, loading the TinyLlama pipeline segfaults
(exit 139) unless OpenMP is limited to one thread. The crash reproduces with just
`import faiss, torch` followed by `transformers.pipeline(...)`. It comes from the
environment, not from the corpus code, and it would affect the private run the same way.

`fetch_corpus.sh` is idempotent. If a verified copy is already in
`corpus/raw/px4-<commit>/`, it skips the download and only re-converts. `--force`
re-downloads. Downloads go into a fresh `mktemp -d` directory and are moved into place only
after every checksum matches and no extra file is present.

### Recorded run

Run on 2026-11-25 from a fresh `git clone` of branch `task-3-public-corpus` into a temp dir.
Environment: macOS 26.6.2 arm64 (M4, 16 GB); Python 3.9.6; sentence-transformers 5.1.2;
faiss-cpu 1.13.0; langchain-text-splitters 0.3.11; pypdf 6.13.1; numpy 2.0.2; torch 2.8.0;
transformers 4.57.6; curl 8.7.1. Models were loaded from the local HF cache
(`HF_HUB_OFFLINE=1`).

- `fetch_corpus.sh`: 99 files verified; 80 pages kept, 17 redirect stubs skipped; 513,931 bytes.
- `check_corpus_tree.py --questions …`: the statistics above; gold problems 0.
- Proxy index 654 vectors; flat index 1,423 chunks (500 / 50, markdown source).
- `--retrieval-only` on the 12 seed questions. Gold section in the assembled context:
  proxy-pointer 7/12, flat top-4 9/12. This is a smoke check of the path, not a
  benchmark result: 12 questions are far too few, and the metric suite is a separate task.
- `OMP_NUM_THREADS=1 … --limit 2` generation smoke test (TinyLlama-1.1B-Chat, fp32, CPU)
  completed (exit 0) and wrote `artifacts/public/proxy_rag_vs_vanilla_rag_{results.csv,.md}`.
  Q1: vanilla 0.8339 vs hybrid 0.6734. Q2: vanilla 0.8283 vs hybrid 0.8543. Mean latency
  was 17.5 s vs 18.4 s, and peak RSS about 4.6 GB. Two questions only prove the path runs;
  these are not results. Two caveats for later tasks: the hybrid path caps the context
  at 1,200 chars before generation, which can cut off the very table the pointer returned;
  and `write_report()`'s "Conclusion" prose is hard-coded text about the private corpus.
