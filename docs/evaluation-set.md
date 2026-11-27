# Public evaluation set and results (Task 4)

The reproducible benchmark: **252 questions** over the PX4 User Guide subset (see
[`corpus.md`](corpus.md)). Everything below regenerates from a clean clone:

```bash
./scripts/fetch_corpus.sh
python3 scripts/build_public_v1.py          # optional: rebuilds questions/public_v1.jsonl (byte-identical)
python3 retrieval_metrics.py --md corpus/px4_*.md --questions questions/public_v1.jsonl \
        --out-dir artifacts/public/task4/v1 --bootstrap 10000 --seed 0
```

## How the questions were made

1. **Drafted** by four LLM agents, one per corpus slice (`questions/drafts/group_A..D.jsonl`),
   plus the 12 seed questions. Each question carries a reference answer, gold section(s) by
   heading path, keywords, and a verbatim evidence snippet from the gold section. Writers were
   told to paraphrase rather than copy heading text, so the set doesn't favour a
   heading-based proxy.
2. **Independently verified** by four *different* LLM agents
   (`questions/verification/verify_*.jsonl`). Each checked every row for answer support,
   missing alternative gold, gold precision, category, ambiguity and heading copying. Result:
   212 ok, 40 fixed, 0 dropped. The dominant error was **missing alternative gold**: the same
   fact restated in a page intro, a sibling page or another manual. Left unfixed, that would
   have penalised a retriever for finding a valid copy. 10 "multihop" questions turned out to be
   answerable from one section and were recategorised.
3. **Applied deterministically** by `scripts/build_public_v1.py`.
4. **Human spot-check: pending.** `questions/human_verification_sample.csv` is a stratified
   sample (15 per category, seed 20261126). Until it is filled in, treat the labels as
   LLM-verified, not gold-standard.

| Category | n | Definition |
|---|---|---|
| tabular | 58 | answer lives in a markdown table |
| procedure | 64 | answer is an ordered sequence of steps |
| navigation | 77 | short lookup: a parameter, a location, a term |
| multihop | 53 | needs ≥ 2 different sections (50 need 2, 3 need 3) |

### Gold format and scoring

`gold` is the union of every acceptable section. For multihop questions, `gold_parts` lists the
**required parts**, each with its alternative locations. Metrics:

- **R@k, MRR, nDCG@5**: a hit on *any* gold section counts. For multihop this measures
  "found at least one needed piece".
- **AllParts@k**: every required part has at least one alternative in the top k. For
  non-multihop questions it equals R@k. Before this metric existed, retrieving one hop of a
  two-hop question earned full credit.
- Rules: **strict** = the retrieved section itself is gold; **context** = the context each
  system actually hands the generator contains gold (proxy-pointer: node + direct children;
  flat: the chunk's section(s)).

Paired bootstrap over questions, 10,000 resamples, seed 0. ✱ = the 95% CI excludes zero.

## Results

### Strict rule (did the retriever rank the right section?)

| Category | n | Flat R@5 | Proxy R@5 | Δ R@5 [95% CI] | Flat MRR | Proxy MRR | Δ MRR [95% CI] |
|---|---|---|---|---|---|---|---|
| overall | 252 | 0.818 | 0.643 | -0.175 [-0.238, -0.111] ✱ | 0.617 | 0.507 | -0.110 [-0.165, -0.055] ✱ |
| tabular | 58 | 0.759 | 0.534 | -0.224 [-0.362, -0.086] ✱ | 0.601 | 0.402 | -0.199 [-0.311, -0.090] ✱ |
| procedure | 64 | 0.844 | 0.719 | -0.125 [-0.219, -0.031] ✱ | 0.639 | 0.547 | -0.092 [-0.199, +0.017] |
| navigation | 77 | 0.779 | 0.610 | -0.169 [-0.299, -0.039] ✱ | 0.596 | 0.497 | -0.100 [-0.207, +0.005] |
| multihop | 53 | 0.906 | 0.717 | -0.189 [-0.321, -0.057] ✱ | 0.640 | 0.590 | -0.050 [-0.155, +0.057] |

### Context rule (did the right section reach the generator?)

| Category | n | Flat R@5 | Proxy R@5 | Δ R@5 [95% CI] | Flat MRR | Proxy MRR | Δ MRR [95% CI] |
|---|---|---|---|---|---|---|---|
| overall | 252 | 0.818 | 0.754 | -0.063 [-0.123, -0.004] ✱ | 0.617 | 0.600 | -0.017 [-0.073, +0.037] |
| tabular | 58 | 0.759 | 0.724 | -0.034 [-0.155, +0.086] | 0.601 | 0.539 | -0.063 [-0.181, +0.052] |
| procedure | 64 | 0.844 | 0.812 | -0.031 [-0.125, +0.047] | 0.639 | 0.652 | +0.013 [-0.095, +0.126] |
| navigation | 77 | 0.779 | 0.714 | -0.065 [-0.195, +0.065] | 0.596 | 0.552 | -0.044 [-0.149, +0.059] |
| multihop | 53 | 0.906 | 0.774 | -0.132 [-0.264, +0.000] | 0.640 | 0.675 | +0.035 [-0.079, +0.151] |

### Multihop: all required parts in the top 5

| Rule | Flat AllParts@5 | Proxy AllParts@5 | Δ [95% CI] |
|---|---|---|---|
| strict | 0.396 | 0.321 | -0.075 [-0.226, +0.075] |
| context | 0.396 | 0.415 | +0.019 [-0.132, +0.170] |

## What this says

- **Proxy-pointer ranks the right section worse than flat chunking, in every category.** Strict
  R@5 is lower in all four categories with CIs excluding zero. Overall: -0.175 [-0.238, -0.111] ✱ R@5,
  -0.110 [-0.165, -0.055] ✱ MRR. This reproduces the private-corpus finding (Task 2) on an
  independent corpus at 8× the sample size.
- **Most of the gap closes once you count what the pointer delivers.** Under the context rule
  (node + children), no per-category MRR difference is significant, and procedure and multihop MRR
  are nominally ahead for proxy-pointer. The pointer-dereference idea works as designed; the
  weakness is the *ranking* signal (heading + first body line).
- **Multihop, where whole-section retrieval was expected to win biggest, is a tie.** AllParts@5
  under the context rule: +0.019 [-0.132, +0.170].
- **Tables are still the worst case** under the strict rule (-0.224 [-0.362, -0.086] ✱ R@5). The
  proxy can't see table rows. That makes the proxy-size sweep (Task 8) the obvious next
  experiment.

## Limitations

- Labels are LLM-drafted and LLM-verified; the human spot-check is pending.
- Writers saw the gold section while writing, so questions may share more vocabulary with the
  gold text than real user queries would. This favours both systems roughly equally (both embed
  body text: flat directly, proxy via the first line), but it inflates absolute recall.
- One embedding model (`all-MiniLM-L6-v2`), one chunk setting (500/50), one proxy construction.
  Task 8 sweeps these.
- Retrieval only: no generator is involved in these numbers.

## Environment

`sentence-transformers/all-MiniLM-L6-v2` on cpu, seed 0; Python 3.9.6,
sentence-transformers 5.1.2, faiss-cpu 1.13.0,
torch 2.8.0; macOS-26.6.2-arm64-arm-64bit; run 2026-11-26 at commit
`cce25bb` (clean tree); corpus PX4 User Guide v1.17.0 @ `d6f12ad`.
