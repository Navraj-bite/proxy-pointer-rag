#!/usr/bin/env python3
"""Train/eval leakage check for the Week 1 fine-tune comparison.

Compares every evaluation prompt (instruction + reference_answer) against every
SFT training example (instruction + output) and reports, per eval id, the
nearest training example by:
  - exact / whitespace-and-case-normalized string match
  - MiniLM (all-MiniLM-L6-v2) cosine similarity
  - token Jaccard overlap (lowercased word tokens)
  - fault-code overlap (defNNNN codes mentioned in both)

Writes leakage_report.json and leakage_per_item.csv to the cwd. Both contain
only ids, train-row indices and numbers -- no text -- but keep them in the
private run dir anyway. Optionally also checks the *.bak (pre-revision) sets.

Run from the private run dir:
    python3 /path/to/repo/leakage_check.py [--bak]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

EMBED_MODEL = "all-MiniLM-L6-v2"
COS_THRESHOLD = 0.9
JACCARD_THRESHOLD = 0.5
TOKEN_RE = re.compile(r"[a-z0-9]+")
CODE_RE = re.compile(r"def\s?\d{3,4}", re.IGNORECASE)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s):
    return " ".join(TOKEN_RE.findall(s.lower()))


def toks(s):
    return set(TOKEN_RE.findall(s.lower()))


def codes(s):
    return {c.lower().replace(" ", "") for c in CODE_RE.findall(s)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def compare(evals, train, embedder, label):
    e_ins = [e["instruction"] for e in evals]
    e_ref = [e["reference_answer"] for e in evals]
    t_ins = [t["instruction"] for t in train]
    t_out = [t["output"] for t in train]

    enc = lambda xs: embedder.encode(xs, normalize_embeddings=True, batch_size=64)
    E_ins, E_ref, T_ins, T_out = enc(e_ins), enc(e_ref), enc(t_ins), enc(t_out)
    S_ins = E_ins @ T_ins.T   # eval instruction vs train instruction
    S_ref = E_ref @ T_out.T   # eval reference vs train output
    S_qa = (E_ins @ T_ins.T + E_ref @ T_out.T) / 2  # pair-level similarity

    t_ins_norm = {norm(x): i for i, x in enumerate(t_ins)}
    t_out_norm = {norm(x): i for i, x in enumerate(t_out)}
    t_ins_tok = [toks(x) for x in t_ins]
    t_out_tok = [toks(x) for x in t_out]
    t_codes = [codes(a) | codes(b) for a, b in zip(t_ins, t_out)]

    rows = []
    for k, e in enumerate(evals):
        j_ins = np.array([jaccard(toks(e_ins[k]), t) for t in t_ins_tok])
        j_ref = np.array([jaccard(toks(e_ref[k]), t) for t in t_out_tok])
        best_pair = int(np.argmax(S_qa[k]))
        e_codes = codes(e_ins[k]) | codes(e_ref[k])
        rows.append({
            "set": label,
            "eval_id": e.get("id", k + 1),
            "exact_instruction": e_ins[k] in t_ins,
            "norm_instruction": norm(e_ins[k]) in t_ins_norm,
            "exact_reference": e_ref[k] in t_out,
            "norm_reference": norm(e_ref[k]) in t_out_norm,
            "max_cos_instruction": float(S_ins[k].max()),
            "argmax_cos_instruction": int(S_ins[k].argmax()),
            "max_cos_reference": float(S_ref[k].max()),
            "argmax_cos_reference": int(S_ref[k].argmax()),
            "pair_best_train_idx": best_pair,
            "pair_cos_instruction": float(S_ins[k, best_pair]),
            "pair_cos_reference": float(S_ref[k, best_pair]),
            "max_jaccard_instruction": float(j_ins.max()),
            "max_jaccard_reference": float(j_ref.max()),
            "eval_fault_codes": ",".join(sorted(e_codes)),
            "fault_code_in_train": bool(e_codes) and any(e_codes & c for c in t_codes),
        })
    df = pd.DataFrame(rows)
    df["near_dup_instruction"] = (df.max_cos_instruction > COS_THRESHOLD) | \
        (df.max_jaccard_instruction > JACCARD_THRESHOLD)
    df["near_dup_reference"] = (df.max_cos_reference > COS_THRESHOLD) | \
        (df.max_jaccard_reference > JACCARD_THRESHOLD)
    df["paired_paraphrase"] = (df.pair_cos_instruction > 0.8) & (df.pair_cos_reference > 0.8)
    return df


def summarize(df):
    def ids(mask):
        return [int(x) for x in df.loc[mask, "eval_id"]]
    return {
        "n_eval": int(len(df)),
        "exact_instruction_ids": ids(df.exact_instruction | df.norm_instruction),
        "exact_reference_ids": ids(df.exact_reference | df.norm_reference),
        "near_dup_instruction_ids": ids(df.near_dup_instruction),
        "near_dup_reference_ids": ids(df.near_dup_reference),
        "paired_paraphrase_ids (both cos>0.8 to same train row)": ids(df.paired_paraphrase),
        "instr_cos>0.8_ids": ids(df.max_cos_instruction > 0.8),
        "ref_cos>0.8_ids": ids(df.max_cos_reference > 0.8),
        "fault_code_in_train_ids": ids(df.fault_code_in_train),
        "max_cos_instruction": {"mean": round(df.max_cos_instruction.mean(), 4),
                                "median": round(df.max_cos_instruction.median(), 4),
                                "min": round(df.max_cos_instruction.min(), 4)},
        "max_cos_reference": {"mean": round(df.max_cos_reference.mean(), 4),
                              "median": round(df.max_cos_reference.median(), 4),
                              "min": round(df.max_cos_reference.min(), 4)},
        "max_jaccard_instruction_mean": round(df.max_jaccard_instruction.mean(), 4),
        "max_jaccard_reference_mean": round(df.max_jaccard_reference.mean(), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="evaluation_prompts.jsonl")
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--bak", action="store_true",
                    help="also compare the *.bak (pre-revision) sets")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)

    evals, train = load_jsonl(args.eval), load_jsonl(args.train)
    frames = [compare(evals, train, embedder, "current")]
    report = {"current": summarize(frames[0]),
              "n_train": len(train),
              "thresholds": {"cos": COS_THRESHOLD, "jaccard": JACCARD_THRESHOLD}}

    # Baseline: how similar are eval items to *each other*? (Context for thresholds.)
    E = embedder.encode([e["instruction"] for e in evals], normalize_embeddings=True)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    report["eval_vs_eval_max_cos_instruction_median"] = round(float(np.median(S.max(1))), 4)

    if args.bak:
        eb, tb = load_jsonl(args.eval + ".bak"), load_jsonl(args.train + ".bak")
        report["bak"] = {
            "n_eval_bak": len(eb), "n_train_bak": len(tb),
            "eval_bak_instructions_kept_in_current":
                sum(e["instruction"] in {x["instruction"] for x in evals} for e in eb),
            "train_bak_instructions_kept_in_current":
                sum(t["instruction"] in {x["instruction"] for x in train} for t in tb),
        }
        fb = compare(eb, tb, embedder, "bak")
        frames.append(fb)
        report["bak"]["eval_bak_vs_train_bak"] = summarize(fb)
        # Did the training-set revision add items targeted at the new eval ids?
        cur = frames[0]
        new_ids = [e["id"] for e in evals if e["instruction"] not in
                   {x["instruction"] for x in eb}]
        tb_set = {t["instruction"] for t in tb}
        added_rows = {i for i, t in enumerate(train) if t["instruction"] not in tb_set}
        sub = cur[cur.eval_id.isin(new_ids)]
        report["bak"]["new_eval_ids"] = [int(x) for x in new_ids]
        report["bak"]["n_train_rows_added_in_revision"] = len(added_rows)
        report["bak"]["new_eval_ids_whose_nearest_pair_is_an_added_train_row"] = \
            [int(r.eval_id) for r in sub.itertuples() if r.pair_best_train_idx in added_rows]

    out = pd.concat(frames, ignore_index=True)
    out.to_csv("leakage_per_item.csv", index=False)
    with open("leakage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
