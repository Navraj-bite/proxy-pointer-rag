#!/usr/bin/env python3
"""Score the Gemma arms (run_gemma_arms.py) with the Week 1 metric and report.

Metric: cosine similarity between all-MiniLM-L6-v2 embeddings of the generated
answer and the reference_answer -- identical to score.py (raw embeddings,
numpy cosine, encoded per item as [reference, answer, ...]).

Reports per arm: mean, median, wins; per arm pair: mean paired delta with a
paired bootstrap 95% CI (10k resamples of question ids, fixed seed), W/L/T.
Repeats the pair stats on leakage subsets (from leakage_per_item.csv, if
present) and on eval ids 1-20 vs 21-40. Also re-scores the stored Week 1
outputs (rag_baseline_outputs.csv, finetuned_outputs.csv) to check that the
published 0.4895 / 0.5079 reproduce, and checks whether the adapter_w1repro
arm reproduces finetuned_outputs.csv verbatim.

Writes (cwd = private run dir; ids + numbers only, no text):
  gemma_arms_scores.csv, gemma_arms_summary.json
Prints a markdown report to stdout.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")

EMBED_MODEL = "all-MiniLM-L6-v2"
PROMPTS_FILE = "evaluation_prompts.jsonl"
BOOT_N = 10_000
BOOT_SEED = 20261125
ARM_ORDER = ["base", "adapter", "rag", "rag_adapter", "adapter_w1repro"]
ABSTAIN_RE = re.compile(r"don.t know|context (?:does ?n.t|does not|do(?:es)? not)|not (?:in|provided in) the context", re.I)
PAIRS = [("adapter", "base"), ("adapter", "rag"), ("rag", "base"),
         ("rag_adapter", "rag"), ("rag_adapter", "base"), ("rag_adapter", "adapter")]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def boot_ci(delta, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means <= 0).mean())


def pair_stats(df, a, b):
    d = (df[a] - df[b]).to_numpy()
    lo, hi, p_le0 = boot_ci(d)
    return {"a": a, "b": b, "n": int(len(d)), "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)), "ci95": [lo, hi],
            "boot_frac_delta_le_0": p_le0,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def fmt_pair(s):
    return (f"| {s['a']} − {s['b']} | {s['n']} | {s['mean_delta']:+.4f} | "
            f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
            f"{s['wins']}/{s['losses']}/{s['ties']} |")


def week1_rescore(prompts, embedder):
    """Replicates score.py exactly on the stored Week 1 CSVs."""
    if not (os.path.exists("rag_baseline_outputs.csv") and os.path.exists("finetuned_outputs.csv")):
        return None
    rag_df = pd.read_csv("rag_baseline_outputs.csv")
    ft_df = pd.read_csv("finetuned_outputs.csv").set_index("id")
    rag_by_q = {str(q).strip(): a for q, a in zip(rag_df["question"], rag_df["generated_answer"])}
    rows = []
    for p in prompts:
        ref = p["reference_answer"]
        ra = str(rag_by_q[p["instruction"].strip()])
        fa = str(ft_df.loc[p["id"], "generated_answer"])
        rv, av, fv = embedder.encode([ref, ra, fa])
        rows.append({"id": p["id"], "w1_rag": cosine(rv, av), "w1_ft": cosine(rv, fv)})
    df = pd.DataFrame(rows)
    out = {"w1_rag_mean": float(df.w1_rag.mean()), "w1_ft_mean": float(df.w1_ft.mean()),
           "w1_ft_wins": int((df.w1_ft > df.w1_rag).sum()),
           "w1_rag_wins": int((df.w1_rag > df.w1_ft).sum()),
           "w1_ft_minus_rag": pair_stats(df.rename(columns={"w1_ft": "ft", "w1_rag": "rag"}),
                                         "ft", "rag")}
    if os.path.exists("comparison.csv"):
        c = pd.read_csv("comparison.csv").set_index("id").loc[df.id]
        out["max_abs_diff_vs_comparison_csv"] = float(max(
            np.abs(c.rag_score.to_numpy() - df.w1_rag.to_numpy()).max(),
            np.abs(c.finetuned_score.to_numpy() - df.w1_ft.to_numpy()).max()))
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    prompts = load_jsonl(PROMPTS_FILE)
    by_id = {p["id"]: p for p in prompts}

    answers, meta = {}, {}
    for arm in [a for a in args.arms.split(",") if a]:
        path = f"gemma_arm_{arm}.jsonl"
        if not os.path.exists(path):
            continue
        recs = {r["id"]: r for r in load_jsonl(path)}
        if set(recs) != set(by_id):
            print(f"[skip] {arm}: {len(recs)}/{len(by_id)} answers")
            continue
        answers[arm] = {i: r["answer"] for i, r in recs.items()}
        meta[arm] = recs
    arms = [a for a in ARM_ORDER if a in answers]

    rows = []
    for p in prompts:
        i = p["id"]
        texts = [p["reference_answer"]] + [answers[a][i] for a in arms]
        vecs = embedder.encode(texts)
        row = {"id": i}
        for a, v in zip(arms, vecs[1:]):
            row[a] = cosine(vecs[0], v)
            row[f"{a}_new_tokens"] = meta[a][i]["new_tokens"]
            row[f"{a}_words"] = len(answers[a][i].split())
        row["ref_words"] = len(p["reference_answer"].split())
        rows.append(row)
    df = pd.DataFrame(rows)

    summary = {"boot": {"n": BOOT_N, "seed": BOOT_SEED, "type": "paired percentile"},
               "arms": {}, "pairs": [], "subsets": {}}
    main_arms = [a for a in arms if a != "adapter_w1repro"]
    for a in arms:
        others = [b for b in main_arms if b != a]
        best = df[a].to_numpy()[:, None] > df[others].to_numpy() if others else None
        summary["arms"][a] = {
            "mean": float(df[a].mean()), "median": float(df[a].median()),
            "std": float(df[a].std()),
            "wins_vs_all_other_main_arms": int(best.all(axis=1).sum()) if others else None,
            "hit_max_new_tokens": int(sum(meta[a][i]["hit_max_new_tokens"] for i in by_id)),
            "mean_words": float(df[f"{a}_words"].mean()),
            "abstentions": int(sum(bool(ABSTAIN_RE.search(answers[a][i])) for i in by_id)),
            "wall_seconds_gen_sum": float(sum(meta[a][i]["gen_seconds"] for i in by_id)),
        }
    for a, b in PAIRS:
        if a in df and b in df:
            summary["pairs"].append(pair_stats(df, a, b))

    # Leakage / provenance subsets
    subsets = {"ids_1_20": df.id <= 20, "ids_21_40": df.id > 20}
    if os.path.exists("leakage_per_item.csv"):
        lk = pd.read_csv("leakage_per_item.csv")
        lk = lk[lk.set == "current"].set_index("eval_id").loc[df.id]
        near = ((lk.max_cos_instruction > 0.9) | (lk.max_cos_reference > 0.9)).to_numpy()
        clean = ((lk.max_cos_instruction <= 0.8) & (lk.max_cos_reference <= 0.8)).to_numpy()
        subsets["near_dup_cos_gt_0.9"] = pd.Series(near, index=df.index)
        subsets["not_near_dup"] = pd.Series(~near, index=df.index)
        subsets["clean_cos_le_0.8"] = pd.Series(clean, index=df.index)
    for name, mask in subsets.items():
        sub = df[mask.to_numpy()]
        summary["subsets"][name] = {
            "ids": [int(x) for x in sub.id],
            "means": {a: float(sub[a].mean()) for a in main_arms},
            "pairs": [pair_stats(sub, a, b) for a, b in PAIRS[:3] if a in sub and b in sub],
        }

    w1 = week1_rescore(prompts, embedder)
    if w1:
        w1df, summary["week1_rescore"] = w1
        df = df.merge(w1df, on="id")
        if "rag" in df:
            summary["week1_rescore"]["gemma_rag_minus_tinyllama_rag"] = pair_stats(df, "rag", "w1_rag")
        if "adapter_w1repro" in answers:
            ft = pd.read_csv("finetuned_outputs.csv").set_index("id")
            same = sum(str(ft.loc[i, "generated_answer"]).strip() == answers["adapter_w1repro"][i]
                       for i in by_id)
            summary["week1_rescore"]["adapter_w1repro_verbatim_matches_finetuned_csv"] = int(same)

    df.to_csv("gemma_arms_scores.csv", index=False)
    with open("gemma_arms_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- markdown report
    print("| Arm | Mean cosine | Median | Best-of-arms wins | Hit max tokens | Mean words | Abstentions |")
    print("|---|---|---|---|---|---|---|")
    for a in arms:
        s = summary["arms"][a]
        w = "—" if a == "adapter_w1repro" else f"{s['wins_vs_all_other_main_arms']}/{len(df)}"
        print(f"| {a} | {s['mean']:.4f} | {s['median']:.4f} | {w} | "
              f"{s['hit_max_new_tokens']}/{len(df)} | {s['mean_words']:.0f} | {s['abstentions']} |")
    print(f"\n(reference answers: mean {df.ref_words.mean():.0f} words)\n")
    print("| Delta | n | Mean Δ | 95% CI (paired bootstrap) | W/L/T |")
    print("|---|---|---|---|---|")
    for s in summary["pairs"]:
        print(fmt_pair(s))
    for name, s in summary["subsets"].items():
        print(f"\n**{name}** (n={len(s['ids'])}) means: " +
              ", ".join(f"{a} {m:.4f}" for a, m in s["means"].items()))
        for p in s["pairs"]:
            print(fmt_pair(p))
    if w1:
        print("\nWeek 1 re-score:", json.dumps(summary["week1_rescore"], indent=1))


if __name__ == "__main__":
    main()
