"""
Proxy-construction x flat-chunking sweep (Task 8), with a dev/test split.

Ranks heading-node sections for every question under
  * proxy variants   (proxy_variants.py; delivery = node + direct children)
  * multi-vector     (node body in ~500-char pieces, node score = max piece)
  * flat chunks      RecursiveCharacterTextSplitter, chunk_size x overlap grid
                     (delivery = the chunk itself)
  * hybrid           the same flat ranking, delivery = node + direct children
                     ("rank by body chunks, dereference to the section")
and scores them with retrieval_metrics.score_ranking under
  strict  : the retrieved section itself is gold
  context : what the system delivers contains gold
            (proxy/multivector/hybrid: node + children; flat: the chunk, so
             context == strict).

Selection (no tuning on test): the best proxy variant and the best flat
setting are chosen on the DEV half by strict MRR (ties: higher R@5, then
list order), then compared on the TEST half with a paired bootstrap.
Full-set numbers for every config are reported as descriptive.

Also measures encoder truncation: all-MiniLM-L6-v2 encodes at most
max_seq_length (256) word-pieces incl. [CLS]/[SEP]; the rest is silently cut.

Public run (writes artifacts/public/task8/, docs/img/task8/*.png):
  HF_HUB_OFFLINE=1 python3 sweep_proxy_chunk.py --md corpus/px4_*.md \\
      --questions questions/public_v1.jsonl --split questions/public_v1_split.json \\
      --out-dir artifacts/public/task8 --plots-dir docs/img/task8

Private run (aggregates only; refuses to write inside the repo):
  python3 /path/to/repo/sweep_proxy_chunk.py --private --md operator_manual.md ... \\
      --questions benchmark_questions.jsonl --gold gold_labels.jsonl --out-dir .
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import OrderedDict

import numpy as np

import retrieval_metrics as rm
import proxy_variants as pv

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 3, 5)
NDCG_K = 5
MRR_DEPTH = 10
DEPTH = max(max(KS), NDCG_K, MRR_DEPTH)
METRICS = ("R@1", "R@5", "MRR", "nDCG@5", "AllParts@5")
RULES = ("strict", "context")
DEFAULT_PROXY = "title_firstline"


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def make_split(questions, seed, dev_frac=0.5):
    """Stratified by category: sort ids, shuffle with random.Random(seed), first
    round(n * dev_frac) per category -> dev, rest -> test."""
    rng = random.Random(seed)
    dev, test = [], []
    for cat in sorted({q["category"] for q in questions}):
        ids = sorted(q["id"] for q in questions if q["category"] == cat)
        rng.shuffle(ids)
        n_dev = int(round(len(ids) * dev_frac - 1e-9))
        dev += ids[:n_dev]
        test += ids[n_dev:]
    return {"seed": seed, "method": "stratified by category; per category (sorted order) "
            "ids sorted then random.Random(seed).shuffle, first round(n/2) -> dev",
            "dev": sorted(dev), "test": sorted(test)}


# --------------------------------------------------------------------------- #
# Cached encoding
# --------------------------------------------------------------------------- #
class CachedEncoder:
    def __init__(self, embedder, model_name, cache_path=None):
        self.embedder, self.model = embedder, model_name
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.cache = pickle.load(f)
        self.n_new = 0

    def _key(self, t):
        return hashlib.sha1((self.model + "\0" + t).encode("utf-8")).hexdigest()

    def encode(self, texts):
        keys = [self._key(t) for t in texts]
        todo = OrderedDict()
        for k, t in zip(keys, texts):
            if k not in self.cache and k not in todo:
                todo[k] = t
        if todo:
            vecs = rm._encode(self.embedder, list(todo.values()))
            for k, v in zip(todo, vecs):
                self.cache[k] = v
            self.n_new += len(todo)
        return np.stack([self.cache[k] for k in keys]).astype("float32")

    def save(self):
        if self.cache_path and self.n_new:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)


# --------------------------------------------------------------------------- #
# Truncation measurement
# --------------------------------------------------------------------------- #
def truncation_stats(tokenizer, max_seq_length, texts):
    """Word-pieces and chars the encoder actually sees (budget = max_seq_length - 2)."""
    budget = max_seq_length - 2
    n_tok, seen_tok, n_chr, seen_chr = [], [], [], []
    for t in texts:
        enc = tokenizer(t, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
        n = len(enc["input_ids"])
        s = min(n, budget)
        n_tok.append(n)
        seen_tok.append(s)
        n_chr.append(len(t))
        seen_chr.append(len(t) if n <= budget else enc["offset_mapping"][s - 1][1])
    n_tok, seen_tok = np.array(n_tok), np.array(seen_tok)
    n_chr, seen_chr = np.array(n_chr), np.array(seen_chr)
    return {"n_texts": int(len(texts)),
            "mean_chars": float(n_chr.mean()), "mean_tokens": float(n_tok.mean()),
            "mean_encoded_chars": float(seen_chr.mean()),
            "mean_encoded_tokens": float(seen_tok.mean()),
            "share_truncated": float((n_tok > budget).mean()),
            "share_chars_seen": float(seen_chr.sum() / max(1, n_chr.sum())),
            "mean_share_chars_seen": float(np.mean(seen_chr / np.maximum(1, n_chr))),
            "p90_chars": float(np.percentile(n_chr, 90)), "budget_tokens": budget,
            "_seen_chars": seen_chr}


# --------------------------------------------------------------------------- #
# Rankers (dense numpy scoring; equivalent to FAISS IndexFlatIP)
# --------------------------------------------------------------------------- #
def _scores(Q, M):
    """Q @ M.T. numpy 2.0 + macOS Accelerate emits spurious 'divide by zero/overflow
    in matmul' warnings on finite unit vectors; silence them and assert finiteness."""
    with np.errstate(all="ignore"):
        S = Q @ M.T
    assert np.isfinite(S).all()
    return S


def rank_from_scores(S, node_ids, depth=DEPTH):
    order = np.argsort(-S, axis=1, kind="stable")[:, :depth]
    return [[node_ids[j] for j in row] for row in order]


def rank_multivector(Q, P, owner, n_nodes, node_ids, depth=DEPTH):
    S = _scores(Q, P)
    node_S = np.full((Q.shape[0], n_nodes), -np.inf, dtype="float32")
    owner = np.asarray(owner)
    for j in np.unique(owner):
        node_S[:, j] = S[:, owner == j].max(axis=1)
    return rank_from_scores(node_S, node_ids, depth)


def rank_chunks(Q, C, chunk_sections, depth=DEPTH):
    S = _scores(Q, C)
    order = np.argsort(-S, axis=1, kind="stable")
    out = []
    for row in order:
        ranked, seen = [], set()
        for i in row:
            for nid in chunk_sections[i]:
                if nid not in seen:
                    seen.add(nid)
                    ranked.append(nid)
            if len(ranked) >= depth:
                break
        out.append(ranked[:depth])
    return out


def flat_chunks(tree, md_files, chunk_size, overlap):
    fnodes = rm.file_nodes(tree)
    chunks = []
    for path in md_files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        doc = os.path.basename(path)
        spans = rm._file_line_spans(tree, fnodes[doc], text)
        for chunk, start in rm.split_with_offsets(text, chunk_size, overlap):
            chunks.append((chunk, rm.map_span_to_sections(spans, start, start + len(chunk))))
    return chunks


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def agg_metrics(rows):
    return {m: mean([r[m] for r in rows]) for m in METRICS}


def is_inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
LADDER = ["title", "title_firstline", "title_body300", "title_body600", "title_body1200",
          "body_full"]
SHORT = {"title": "heading", "title_firstline": "+first line (shipped)",
         "title_body300": "+300", "title_body600": "+600", "title_body1200": "+1200 = full body",
         "body_full": "full body", "title_keywords": "keywords", "path_firstline": "path+line",
         "multivector": "multi-vector"}


def plot_size_curve(path, full, trunc, cats, flat_key, hybrid_key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = ["overall"] + cats
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9), sharey=True)
    for ax, cat in zip(axes, panels):
        for rule, color in (("strict", BLUE), ("context", ORANGE)):
            xs = [trunc[f"proxy:{v}"]["mean_encoded_chars"] for v in LADDER]
            ys = [full[(f"proxy:{v}", rule, cat)]["R@5"] for v in LADDER]
            ax.plot(xs, ys, "-o", color=color, lw=2, ms=6, label=f"proxy ladder, {rule}")
            for v, mk in (("title_keywords", "s"), ("path_firstline", "D")):
                ax.plot(trunc[f"proxy:{v}"]["mean_encoded_chars"],
                        full[(f"proxy:{v}", rule, cat)]["R@5"], mk, color=color, ms=7,
                        mfc="white", mew=2)
            ax.plot(trunc["proxy:multivector"]["mean_encoded_chars_per_node"],
                    full[("proxy:multivector", rule, cat)]["R@5"], "*", color=color, ms=13,
                    mec="white", mew=0.8)
        ax.axhline(full[(flat_key, "strict", cat)]["R@5"], color=BLUE, ls="--", lw=1.2)
        ax.axhline(full[(hybrid_key, "context", cat)]["R@5"], color=ORANGE, ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 200, 400, 800])
        ax.set_xticklabels(["25", "50", "100", "200", "400", "800"])
        ax.minorticks_off()
        ax.set_title(cat, fontsize=11)
        ax.set_xlabel("mean chars the encoder sees per node (log)", fontsize=8.5)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if cat == "navigation":
            ax.set_facecolor("#fbf7f0")
    axes[0].set_ylabel("Recall@5")
    ov = axes[0]
    for v in LADDER[:-1]:
        ov.annotate(SHORT[v], (trunc[f"proxy:{v}"]["mean_encoded_chars"],
                               full[(f"proxy:{v}", "strict", "overall")]["R@5"]),
                    textcoords="offset points", fontsize=6.5, color=GREY,
                    xytext=(4, 8) if v == "title_body1200" else (3, -11),
                    ha="right" if v == "title_body1200" else "left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", label="strict rule"),
               Line2D([], [], color=ORANGE, lw=2, marker="o", label="context rule (node + children)"),
               Line2D([], [], color=GREY, ls="", marker="s", mfc="white", label="heading + keywords"),
               Line2D([], [], color=GREY, ls="", marker="D", mfc="white", label="heading path + first line"),
               Line2D([], [], color=GREY, ls="", marker="*", ms=11, label="multi-vector (x = whole node)"),
               Line2D([], [], color=BLUE, ls="--", label=f"tuned flat {flat_key[5:]}, chunk delivery"),
               Line2D([], [], color=ORANGE, ls="--", label=f"flat-rank {flat_key[5:]} + section delivery")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Recall@5 vs proxy size, all 252 public questions (descriptive)", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_flat_grid(path, full, sizes, overlaps, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [("flat", "strict", "R@5", "R@5, strict"), ("flat", "strict", "MRR", "MRR, strict"),
             ("hybrid", "context", "R@5", "R@5, section delivery (context)"),
             ("hybrid", "context", "MRR", "MRR, section delivery (context)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (sys_, rule, m, title) in zip(axes, specs):
        M = np.array([[full[(f"{sys_}:{cs}/{ov}", rule, "overall")][m] for ov in overlaps]
                      for cs in sizes])
        ax.imshow(M, cmap="Blues", vmin=M.min() - 0.08, vmax=M.max() + 0.02, aspect="auto")
        for i, cs in enumerate(sizes):
            for j, ov in enumerate(overlaps):
                v = M[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if v > (M.min() + M.max()) / 2 else "#0b0b0b")
                if f"{cs}/{ov}" == chosen:
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                               ec=ORANGE, lw=2.5))
        ax.set_xticks(range(len(overlaps)), [str(o) for o in overlaps])
        ax.set_yticks(range(len(sizes)), [str(s) for s in sizes])
        ax.set_xlabel("overlap (chars)")
        ax.set_ylabel("chunk size (chars)")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Flat chunking grid, all 252 public questions (descriptive); "
                 f"orange box = chosen on dev ({chosen})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", nargs="+", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--gold", default=None, help="gold JSONL (default: inline `gold`)")
    ap.add_argument("--split", default=None,
                    help="dev/test id list (JSON); created with --split-seed if missing")
    ap.add_argument("--split-seed", type=int, default=8)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 500, 1000])
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 50, 200])
    ap.add_argument("--body-chars", nargs="+", type=int, default=list(pv.BODY_CHARS))
    ap.add_argument("--embed-model", default=rm.EMBED_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plots-dir", default=None)
    ap.add_argument("--summary-copy", default=None,
                    help="also write the summary JSON here (public runs: the committed record)")
    ap.add_argument("--cache", default=None, help="embedding cache (default <out-dir>/emb_cache.pkl)")
    ap.add_argument("--private", action="store_true",
                    help="private corpus: refuse any output path inside the repo; no plots; "
                         "per-question CSV omits section labels")
    args = ap.parse_args(argv)

    if args.private:
        for p in (args.out_dir, args.plots_dir, args.cache, args.summary_copy):
            if p and is_inside(p, HERE):
                sys.exit(f"--private: refusing to write {p} inside the repo {HERE}")
        args.plots_dir = None

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    questions = rm.load_jsonl(args.questions)
    tree = rm.build_tree(args.md)
    gold_records, gold, parts = rm.load_gold(tree, questions, args.gold)
    cats = [c for c in rm.CATEGORIES if any(q["category"] == c for q in questions)]

    split = None
    if args.split:
        fresh = make_split(questions, args.split_seed)
        if os.path.exists(args.split):
            with open(args.split) as f:
                split = json.load(f)
            ids = {q["id"] for q in questions}
            if set(split["dev"]) | set(split["test"]) != ids or set(split["dev"]) & set(split["test"]):
                sys.exit("split file does not partition the question ids")
            if split["dev"] != fresh["dev"] and split.get("seed") == args.split_seed:
                print("note: split file differs from a fresh split with the same seed")
        else:
            split = fresh
            with open(args.split, "w") as f:
                json.dump(split, f, indent=1)
            print(f"wrote split {args.split}")
    part_of = {}
    if split:
        part_of.update({i: "dev" for i in split["dev"]})
        part_of.update({i: "test" for i in split["test"]})

    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass
    embedder = rm.load_embedder(args.embed_model, args.device)
    enc = CachedEncoder(embedder, args.embed_model,
                        args.cache or os.path.join(args.out_dir, "emb_cache.pkl"))
    tok, max_len = embedder.tokenizer, embedder.max_seq_length
    Q = enc.encode([q["question"] for q in questions])

    nodes = rm.indexable_nodes(tree)
    node_ids = [n.node_id for n in nodes]
    systems = OrderedDict()      # name -> (rankings, delivery context fn)
    trunc = OrderedDict()
    ctx_node = lambda nid: {nid, *tree.nodes[nid].child_node_ids}
    ctx_id = rm.SectionRetriever.context_identity

    funcs = pv.make_proxy_funcs(tree, tuple(args.body_chars))
    proxy_texts = {}
    for name, fn in funcs.items():
        texts = [fn(n) for n in nodes]
        proxy_texts[name] = texts
        V = enc.encode(texts)
        systems[f"proxy:{name}"] = (rank_from_scores(_scores(Q, V), node_ids), ctx_node)
        trunc[f"proxy:{name}"] = truncation_stats(tok, max_len, texts)
        proxy_texts[name] = [t[:int(c)] for t, c in zip(texts, trunc[f"proxy:{name}"]["_seen_chars"])]
        print(f"  proxy:{name:<16} {time.time() - t0:6.1f}s")

    pieces, owner = [], []
    for j, n in enumerate(nodes):
        for p in pv.node_pieces(n):
            pieces.append(p)
            owner.append(j)
    P = enc.encode(pieces)
    systems["proxy:multivector"] = (rank_multivector(Q, P, owner, len(nodes), node_ids), ctx_node)
    st = truncation_stats(tok, max_len, pieces)
    seen = st["_seen_chars"]
    per_node = np.zeros(len(nodes))
    np.add.at(per_node, np.asarray(owner), seen)
    st["mean_encoded_chars_per_node"] = float(per_node.mean())
    st["pieces_per_node"] = len(pieces) / len(nodes)
    trunc["proxy:multivector"] = st
    print(f"  proxy:multivector      {time.time() - t0:6.1f}s ({len(pieces)} pieces)")

    flat_names = []
    for cs in args.chunk_sizes:
        for ov in args.overlaps:
            ch = flat_chunks(tree, args.md, cs, ov)
            C = enc.encode([c[0] for c in ch])
            ranks = rank_chunks(Q, C, [c[1] for c in ch])
            key = f"{cs}/{ov}"
            flat_names.append(key)
            systems[f"flat:{key}"] = (ranks, ctx_id)
            systems[f"hybrid:{key}"] = (ranks, ctx_node)
            st = truncation_stats(tok, max_len, [c[0] for c in ch])
            st["n_chunks"] = len(ch)
            trunc[f"flat:{key}"] = st
            print(f"  flat:{key:<10} {time.time() - t0:6.1f}s ({len(ch)} chunks)")
    enc.save()
    for st in trunc.values():
        st.pop("_seen_chars", None)

    # ---- score everything ------------------------------------------------
    rows = []
    for sname, (ranks, ctx) in systems.items():
        for i, q in enumerate(questions):
            for rule in RULES:
                cfn = ctx_id if rule == "strict" else ctx
                m = rm.score_ranking(tree, ranks[i], gold[q["id"]], cfn, KS, NDCG_K, MRR_DEPTH,
                                     parts=parts[q["id"]])
                rows.append({"id": q["id"], "category": q["category"],
                             "split": part_of.get(q["id"], ""), "system": sname, "rule": rule,
                             **{k: m[k] for k in METRICS}, "first_rank": m["first_rank"]})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sweep_per_question.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    idx = {}
    for r in rows:
        idx.setdefault((r["system"], r["rule"]), []).append(r)

    def sel(sname, rule, subset="all", cat="overall"):
        return [r for r in idx[(sname, rule)]
                if (subset == "all" or r["split"] == subset) and (cat == "overall" or r["category"] == cat)]

    subsets = ["all"] + (["dev", "test"] if split else [])
    aggregates = {}
    for subset in subsets:
        for sname in systems:
            for rule in RULES:
                for cat in ["overall"] + cats:
                    rs = sel(sname, rule, subset, cat)
                    aggregates[(subset, sname, rule, cat)] = {"n": len(rs), **agg_metrics(rs)}

    proxy_names = [s for s in systems if s.startswith("proxy:")]
    flat_sys = [f"flat:{k}" for k in flat_names]

    def pick(cands, subset):
        best = max(enumerate(cands), key=lambda t: (
            aggregates[(subset, t[1], "strict", "overall")]["MRR"],
            aggregates[(subset, t[1], "strict", "overall")]["R@5"], -t[0]))
        return best[1]

    sel_subset = "dev" if split else "all"
    best_proxy, best_flat = pick(proxy_names, sel_subset), pick(flat_sys, sel_subset)
    best_single = pick([p for p in proxy_names if p != "proxy:multivector"], sel_subset)
    best_hybrid = "hybrid:" + best_flat.split(":", 1)[1]
    eval_subset = "test" if split else "all"

    def boot(a_sys, a_rule, b_sys, b_rule, cat, metric, subset=eval_subset):
        a = sorted(sel(a_sys, a_rule, subset, cat), key=lambda r: r["id"])
        b = sorted(sel(b_sys, b_rule, subset, cat), key=lambda r: r["id"])
        assert [r["id"] for r in a] == [r["id"] for r in b]
        return rm.paired_bootstrap([r[metric] for r in a], [r[metric] for r in b],
                                   args.bootstrap, args.seed)

    comparisons = OrderedDict([
        ("chosen_proxy_vs_tuned_flat|strict", (best_proxy, "strict", best_flat, "strict")),
        ("chosen_proxy_vs_tuned_flat|context", (best_proxy, "context", best_flat, "context")),
        ("chosen_proxy_vs_hybrid|context", (best_proxy, "context", best_hybrid, "context")),
        ("hybrid_vs_tuned_flat|context", (best_hybrid, "context", best_flat, "context")),
        ("best_single_vector_proxy_vs_tuned_flat|strict", (best_single, "strict", best_flat, "strict")),
        ("best_single_vector_proxy_vs_tuned_flat|context", (best_single, "context", best_flat, "context")),
        ("best_single_vector_proxy_vs_hybrid|context", (best_single, "context", best_hybrid, "context")),
        ("shipped_proxy_vs_tuned_flat|strict", (f"proxy:{DEFAULT_PROXY}", "strict", best_flat, "strict")),
        ("shipped_proxy_vs_tuned_flat|context", (f"proxy:{DEFAULT_PROXY}", "context", best_flat, "context")),
        ("chosen_proxy_vs_shipped_proxy|strict", (best_proxy, "strict", f"proxy:{DEFAULT_PROXY}", "strict")),
    ])
    head2head = {}
    for cname, (a, ar, b, br) in comparisons.items():
        for cat in ["overall"] + cats:
            ms = ["R@1", "R@5", "MRR", "nDCG@5"] + (["AllParts@5"] if cat in ("overall", "multihop") else [])
            for m in ms:
                head2head[f"{cname}|{cat}|{m}"] = {"a": a, "b": b, **boot(a, ar, b, br, cat, m)}
    any_variant = {}
    for p in proxy_names:
        for rule in RULES:
            for m in ("MRR", "R@5"):
                any_variant[f"{p}|{rule}|{m}"] = boot(p, rule, best_flat, rule, "overall", m)

    # keyword visibility (hypothesis test): does any of the question's `keywords`
    # appear in the ENCODED part of a gold node's proxy? Then R@5 (strict) split by it.
    visibility = {}
    kwq = [(i, q) for i, q in enumerate(questions) if q.get("keywords")]
    if kwq:
        pos = {nid: j for j, nid in enumerate(node_ids)}
        for name, enc_texts in proxy_texts.items():
            sname = f"proxy:{name}"
            r5 = {r["id"]: r["R@5"] for r in idx[(sname, "strict")]}
            for cat in ["overall"] + cats:
                vis, hid = [], []
                for i, q in kwq:
                    if cat != "overall" and q["category"] != cat:
                        continue
                    kws = [k.lower() for k in q["keywords"]]
                    seen_ = any(k in enc_texts[pos[g]].lower() for g in gold[q["id"]] for k in kws)
                    (vis if seen_ else hid).append(r5[q["id"]])
                visibility[f"{sname}|{cat}"] = {
                    "n": len(vis) + len(hid), "share_visible": len(vis) / max(1, len(vis) + len(hid)),
                    "R@5_if_visible": mean(vis), "R@5_if_hidden": mean(hid),
                    "n_visible": len(vis), "n_hidden": len(hid)}

    # robustness: the chosen proxy against EVERY flat setting (not just the dev pick)
    vs_every_flat = {}
    for f_ in flat_sys:
        h_ = "hybrid:" + f_.split(":", 1)[1]
        for m in ("MRR", "R@5"):
            vs_every_flat[f"{f_}|strict|{m}"] = boot(best_proxy, "strict", f_, "strict", "overall", m)
            vs_every_flat[f"{h_}|context|{m}"] = boot(best_proxy, "context", h_, "context", "overall", m)

    # ---- environment -----------------------------------------------------
    env = rm.environment(args.device, args.embed_model, args.seed)
    env["max_seq_length"] = int(max_len)
    env["corpus_sha256"] = {os.path.basename(p): sha256_file(p) for p in args.md}
    env["corpus_sha256_combined"] = hashlib.sha256(
        "".join(env["corpus_sha256"][os.path.basename(p)] for p in sorted(args.md)).encode()).hexdigest()
    env["questions_sha256"] = sha256_file(args.questions)
    env["runtime_s"] = round(time.time() - t0, 1)
    if args.private:
        env["corpus_sha256"] = {f"file{i}": v for i, v in enumerate(env["corpus_sha256"].values())}

    def keystr(t):
        return "|".join(t)

    summary = {
        "description": "Task 8 proxy-construction x flat-chunking sweep; see docs/proxy-chunk-sweep.md",
        "corpus": "private (not reproducible)" if args.private else "public",
        "config": {"md": None if args.private else [os.path.basename(p) for p in args.md],
                   "questions": os.path.basename(args.questions), "n_questions": len(questions),
                   "categories": {c: sum(q["category"] == c for q in questions) for c in cats},
                   "n_nodes": len(nodes), "ks": KS, "ndcg_k": NDCG_K, "mrr_depth": MRR_DEPTH,
                   "chunk_sizes": args.chunk_sizes, "overlaps": args.overlaps,
                   "body_chars": args.body_chars, "kw_top": pv.KW_TOP,
                   "kw_table_cells": pv.KW_TABLE_CELLS, "mv_size": pv.MV_SIZE,
                   "mv_overlap": pv.MV_OVERLAP, "bootstrap": args.bootstrap, "seed": args.seed,
                   "split": os.path.basename(args.split) if args.split else None,
                   "split_seed": split["seed"] if split else None,
                   "n_dev": len(split["dev"]) if split else None,
                   "n_test": len(split["test"]) if split else None,
                   "selection": f"best by strict MRR on {sel_subset} (ties: R@5, then list order)"},
        "selection": {"best_proxy": best_proxy, "best_flat": best_flat, "best_hybrid": best_hybrid,
                      "best_single_vector_proxy": best_single,
                      "subset": sel_subset, "evaluated_on": eval_subset},
        "truncation": trunc,
        "aggregates": {keystr(k): v for k, v in aggregates.items()},
        "head_to_head": head2head,
        "any_variant_vs_tuned_flat": any_variant,
        "chosen_proxy_vs_every_flat": vs_every_flat,
        "keyword_visibility": visibility,
        "environment": env,
    }
    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    if args.summary_copy:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_copy)), exist_ok=True)
        with open(args.summary_copy, "w") as f:
            json.dump(summary, f, indent=1)
    tables = render_tables(summary, systems, cats, subsets, aggregates)
    with open(os.path.join(args.out_dir, "sweep_tables.md"), "w") as f:
        f.write(tables)

    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)
        full = {(s, r, c): aggregates[("all", s, r, c)] for s in systems for r in RULES
                for c in ["overall"] + cats}
        plot_size_curve(os.path.join(args.plots_dir, "recall_vs_proxy_size.png"), full, trunc,
                        cats, best_flat, best_hybrid)
        plot_flat_grid(os.path.join(args.plots_dir, "flat_grid.png"), full, args.chunk_sizes,
                       args.overlaps, best_flat.split(":", 1)[1])
    print(f"selected on {sel_subset}: proxy={best_proxy} flat={best_flat}")
    for k, v in head2head.items():
        if "|overall|" in k and k.endswith(("|MRR", "|R@5")):
            print(f"  {k:<58} {v['mean_delta']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    print(f"wrote {args.out_dir}/sweep_summary.json, sweep_tables.md, sweep_per_question.csv "
          f"({time.time() - t0:.0f}s)")


def render_tables(summary, systems, cats, subsets, aggregates):
    L = []
    f3 = lambda x: f"{x:.3f}"
    sel = summary["selection"]
    L.append(f"selection on {sel['subset']}: best proxy = {sel['best_proxy']}, "
             f"best single-vector proxy = {sel['best_single_vector_proxy']}, "
             f"best flat = {sel['best_flat']}; head-to-heads on {sel['evaluated_on']}\n")
    L.append("## Truncation\n")
    L.append("| config | texts | mean chars | mean word-pieces | truncated | chars seen (sum) "
             "| mean encoded chars |")
    L.append("|---|---|---|---|---|---|---|")
    for k, t in summary["truncation"].items():
        extra = f" ({t['mean_encoded_chars_per_node']:.0f}/node)" if "mean_encoded_chars_per_node" in t else ""
        L.append(f"| {k} | {t['n_texts']} | {t['mean_chars']:.0f} | {t['mean_tokens']:.0f} | "
                 f"{100 * t['share_truncated']:.1f}% | {100 * t['share_chars_seen']:.1f}% | "
                 f"{t['mean_encoded_chars']:.0f}{extra} |")
    for subset in subsets:
        for rule in RULES:
            L.append(f"\n## {subset} / {rule} / overall\n")
            L.append("| system | n | R@1 | R@5 | MRR | nDCG@5 | multihop AllParts@5 |")
            L.append("|---|---|---|---|---|---|---|")
            for s in systems:
                if rule == "strict" and s.startswith("hybrid:"):
                    continue
                a = aggregates[(subset, s, rule, "overall")]
                mh = aggregates.get((subset, s, rule, "multihop"), {}).get("AllParts@5", float("nan"))
                L.append(f"| {s} | {a['n']} | " + " | ".join(f3(a[m]) for m in METRICS[:4])
                         + f" | {f3(mh)} |")
        for rule in RULES:
            L.append(f"\n## {subset} / {rule} / per category (R@5 / MRR)\n")
            L.append("| system | " + " | ".join(cats) + " |")
            L.append("|---|" + "---|" * len(cats))
            for s in systems:
                if rule == "strict" and s.startswith("hybrid:"):
                    continue
                L.append(f"| {s} | " + " | ".join(
                    f"{f3(aggregates[(subset, s, rule, c)]['R@5'])} / "
                    f"{f3(aggregates[(subset, s, rule, c)]['MRR'])}" for c in cats) + " |")
            L.append(f"\n## {subset} / {rule} / per category (R@1 / nDCG@5)\n")
            L.append("| system | " + " | ".join(cats) + " |")
            L.append("|---|" + "---|" * len(cats))
            for s in systems:
                if rule == "strict" and s.startswith("hybrid:"):
                    continue
                L.append(f"| {s} | " + " | ".join(
                    f"{f3(aggregates[(subset, s, rule, c)]['R@1'])} / "
                    f"{f3(aggregates[(subset, s, rule, c)]['nDCG@5'])}" for c in cats) + " |")
    L.append("\n## Head-to-head (paired bootstrap, a - b)\n")
    L.append("| comparison | category | metric | a | b | delta [95% CI] |")
    L.append("|---|---|---|---|---|---|")
    for k, v in summary["head_to_head"].items():
        c, cat, m = k.rsplit("|", 2)
        star = "" if v["crosses_zero"] else " *"
        L.append(f"| {c} | {cat} | {m} | {v['a']} | {v['b']} | {v['mean_delta']:+.3f} "
                 f"[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]{star} |")
    L.append(f"\n## Every proxy variant vs tuned flat {sel['best_flat']} ({sel['evaluated_on']}, overall)\n")
    L.append("| variant | rule | metric | delta [95% CI] |")
    L.append("|---|---|---|---|")
    for k, v in summary["any_variant_vs_tuned_flat"].items():
        p, rule, m = k.rsplit("|", 2)
        star = "" if v["crosses_zero"] else " *"
        L.append(f"| {p} | {rule} | {m} | {v['mean_delta']:+.3f} [{v['ci_low']:+.3f}, "
                 f"{v['ci_high']:+.3f}]{star} |")
    if summary.get("keyword_visibility"):
        L.append("\n## Keyword visibility: a question keyword inside the encoded part of a gold "
                 "node's proxy (all questions); strict R@5 if visible / if hidden\n")
        L.append("| proxy | " + " | ".join(["overall"] + cats) + " |")
        L.append("|---|" + "---|" * (len(cats) + 1))
        names = sorted({k.split("|")[0] for k in summary["keyword_visibility"]},
                       key=lambda n: list(systems).index(n))
        for n in names:
            cells = []
            for c in ["overall"] + cats:
                v = summary["keyword_visibility"][f"{n}|{c}"]
                cells.append(f"{100 * v['share_visible']:.0f}% vis; {f3(v['R@5_if_visible'])} / "
                             f"{f3(v['R@5_if_hidden'])}")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append(f"\n## {sel['best_proxy']} vs every flat setting ({sel['evaluated_on']}, overall)\n")
    L.append("| baseline | rule | metric | delta [95% CI] |")
    L.append("|---|---|---|---|")
    for k, v in summary["chosen_proxy_vs_every_flat"].items():
        b, rule, m = k.rsplit("|", 2)
        star = "" if v["crosses_zero"] else " *"
        L.append(f"| {b} | {rule} | {m} | {v['mean_delta']:+.3f} [{v['ci_low']:+.3f}, "
                 f"{v['ci_high']:+.3f}]{star} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
