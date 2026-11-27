"""
Router evaluation: separate routing from retrieval in the "Proxy-Pointer" hybrid.

The published "Proxy-Pointer" column was `hybrid_intent_router`: a regex router
that sends zero-trigger questions to a bare LLM (`fast`, no retrieval) and the
rest to proxy-pointer retrieval + LLM (`deep`). This script measures the two
parts separately.

Subcommands
-----------
  retrieval   Router accuracy (classify_query vs hand labels and vs a category
              oracle) and the routing x retrieval 2x2 at retrieval level:
              rows = routing policy (none = always retrieve, router, oracle =
              hand labels, category oracle), columns = flat 500/50 chunks vs
              proxy-pointer. A fast-routed question retrieves nothing, so its
              retrieval metrics are 0. Paired bootstrap CIs for proxy - flat
              within each row and policy - none within each column.
  contexts    Build the generator context for every question and retriever
              (flat: top-4 chunks joined like naive_rag_baseline; proxy: top-1
              node + direct children, as context_assembler builds it).
  generate    Generate answers for the answer-level 2x2 with ONE prompt
              template, one max_new_tokens, greedy decoding, and the same char
              cap for every cell. Resumable; reuses generations where cells are
              identical (a fast-routed answer is the same in both router cells).
              Imports no faiss, so it is safe to run in its own process.
  answers     Score the generations with the published cosine metric
              (MiniLM(reference) . MiniLM(answer), normalised, 4 dp) and build
              the answer-level 2x2 with paired bootstrap CIs.

Public corpus (reproducible):
  HF_HUB_OFFLINE=1 python3 router_eval.py retrieval --md corpus/px4_*.md \\
      --questions questions/public_v1.jsonl --labels questions/router_labels_public.jsonl \\
      --out-dir artifacts/public/task6 --bootstrap 10000 --seed 0

Labels file: JSONL with `id` and `label` ("fast" | "deep"); see
docs/router-evaluation.md for the labelling criterion.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

CATEGORY_ORACLE = {"tabular": "deep", "procedure": "deep", "multihop": "deep",
                   "navigation": "fast"}
CATEGORIES = ("tabular", "procedure", "navigation", "multihop")
POLICIES = ("none", "router", "oracle", "category")
POLICY_NAMES = {"none": "no router (always retrieve)", "router": "regex router (classify_query)",
                "oracle": "oracle router (hand labels)", "category": "category oracle"}
SYSTEMS = ("flat_md", "proxy_pointer")
SYSTEM_NAMES = {"flat_md": "flat 500/50", "proxy_pointer": "proxy-pointer"}
CTX_SEP = "\n\n---\n\n"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_labels(path, questions):
    labels = {r["id"]: r["label"] for r in load_jsonl(path)}
    qids = [q["id"] for q in questions]
    missing = [i for i in qids if i not in labels]
    if missing:
        raise SystemExit(f"labels missing for {len(missing)} questions, e.g. {missing[:5]}")
    bad = {v for v in labels.values()} - {"fast", "deep"}
    if bad:
        raise SystemExit(f"labels must be fast|deep, got {bad}")
    return labels


def triggers(question):
    from hybrid_intent_router import _DEEP_RE
    return [p.pattern.replace(r"\b", "") for p in _DEEP_RE if p.search(question)]


def routes_for(questions, labels):
    """{policy: {qid: 'fast'|'deep'}}"""
    from hybrid_intent_router import classify_query
    out = {p: {} for p in POLICIES}
    for q in questions:
        out["none"][q["id"]] = "deep"
        out["router"][q["id"]] = classify_query(q["question"])[0]
        out["oracle"][q["id"]] = labels[q["id"]]
        out["category"][q["id"]] = CATEGORY_ORACLE.get(q.get("category"), "deep")
    return out


def bootstrap(a, b, n_boot, seed):
    from retrieval_metrics import paired_bootstrap
    return paired_bootstrap(a, b, n_boot, seed)


def fmt_ci(bs):
    star = " ✱" if not bs["crosses_zero"] else ""
    return f"{bs['mean_delta']:+.3f} [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]{star}"


# --------------------------------------------------------------------------- #
# Router accuracy
# --------------------------------------------------------------------------- #
def agreement(pred, truth, ids, n_boot=10000, seed=0):
    """Confusion matrix and fast/deep precision/recall of `pred` against `truth`."""
    cm = Counter((truth[i], pred[i]) for i in ids)
    n = len(ids)
    acc = sum(cm[(c, c)] for c in ("fast", "deep")) / n if n else float("nan")
    res = {"n": n, "accuracy": acc,
           "confusion": {f"true_{t}|pred_{p}": cm[(t, p)] for t in ("fast", "deep")
                         for p in ("fast", "deep")}}
    for c in ("fast", "deep"):
        tp = cm[(c, c)]
        pp = sum(cm[(t, c)] for t in ("fast", "deep"))
        ap = sum(cm[(c, p)] for p in ("fast", "deep"))
        res[f"{c}_precision"] = tp / pp if pp else float("nan")
        res[f"{c}_recall"] = tp / ap if ap else float("nan")
    # Cohen's kappa
    pe = sum((sum(cm[(c, p)] for p in ("fast", "deep")) / n)
             * (sum(cm[(t, c)] for t in ("fast", "deep")) / n) for c in ("fast", "deep")) if n else 0
    res["kappa"] = (acc - pe) / (1 - pe) if n and pe < 1 else float("nan")
    hit = np.array([pred[i] == truth[i] for i in ids], float)
    if n:
        rng = np.random.default_rng(seed)
        means = hit[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        res["accuracy_ci"] = [float(x) for x in np.percentile(means, [2.5, 97.5])]
    return res


def misroute_patterns(questions, labels, top=12):
    from hybrid_intent_router import classify_query
    trig_fast, trig_deep = Counter(), Counter()
    spurious, missed = [], []
    lead = Counter()
    for q in questions:
        t = triggers(q["question"])
        lab = labels[q["id"]]
        for p in t:
            (trig_fast if lab == "fast" else trig_deep)[p] += 1
        route = classify_query(q["question"])[0]
        if route == "deep" and lab == "fast":
            spurious.append({"id": q["id"], "category": q.get("category"), "triggers": t})
        if route == "fast" and lab == "deep":
            missed.append({"id": q["id"], "category": q.get("category")})
            words = re.findall(r"[a-z']+", q["question"].lower())
            lead[" ".join(words[:2])] += 1
    trig = {p: {"fires_on_deep": trig_deep[p], "fires_on_fast": trig_fast[p]}
            for p in sorted(set(trig_fast) | set(trig_deep),
                            key=lambda x: -(trig_fast[x] + trig_deep[x]))}
    return {"trigger_counts": trig,
            "spurious_deep": spurious,
            "spurious_by_trigger": Counter(p for s in spurious for p in s["triggers"]).most_common(),
            "missed_deep": missed,
            "missed_deep_by_category": Counter(m["category"] for m in missed).most_common(),
            "missed_deep_leading_words": lead.most_common(top)}


def accuracy_report(questions, labels, routes, n_boot, seed):
    ids = [q["id"] for q in questions]
    cats = [c for c in CATEGORIES if any(q.get("category") == c for q in questions)]
    out = {}
    for name, truth in (("hand_labels", labels), ("category_oracle", routes["category"])):
        r = {"overall": agreement(routes["router"], truth, ids, n_boot, seed)}
        for c in cats:
            r[c] = agreement(routes["router"], truth,
                             [q["id"] for q in questions if q.get("category") == c], n_boot, seed)
        out[f"router_vs_{name}"] = r
    out["category_oracle_vs_hand_labels"] = {"overall": agreement(routes["category"], labels, ids,
                                                                  n_boot, seed)}
    out["route_counts"] = {p: dict(Counter(routes[p].values())) for p in POLICIES}
    out["label_counts_by_category"] = {c: dict(Counter(labels[q["id"]] for q in questions
                                                       if q.get("category") == c)) for c in cats}
    out["misroutes"] = misroute_patterns(questions, labels)
    return out


# --------------------------------------------------------------------------- #
# Retrieval-level 2x2
# --------------------------------------------------------------------------- #
def cmd_retrieval(args):
    import retrieval_metrics as rm

    questions = load_jsonl(args.questions)
    labels = load_labels(args.labels, questions)
    routes = routes_for(questions, labels)
    acc = accuracy_report(questions, labels, routes, args.bootstrap, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.accuracy_only:
        _dump(os.path.join(args.out_dir, "router_accuracy.json"), acc)
        print(json.dumps({k: v["overall"] for k, v in acc.items() if k.startswith(("router", "cat"))},
                         indent=1))
        return

    tree = rm.build_tree(args.md)
    gold_records = load_jsonl(args.gold) if args.gold else [{"id": q["id"], "gold": q.get("gold")}
                                                            for q in questions]
    gold = rm.resolve_gold(tree, gold_records, [q["id"] for q in questions])
    parts = {}
    for q in questions:
        if q.get("gold_parts"):
            parts[q["id"]] = [[rm.resolve_heading_path(tree, os.path.basename(g["doc"]),
                                                       g["heading_path"]) for g in part]
                              for part in q["gold_parts"]]
        else:
            parts[q["id"]] = [list(gold[q["id"]])]

    r = rm.SectionRetriever(args.md, device="cpu", tree=tree)
    qv = r.encode_queries([q["question"] for q in questions])
    depth = 10
    rank = {"proxy_pointer": lambda i, q: r.rank_sections_proxy(q, depth, qvec=qv[i]),
            "flat_md": lambda i, q: r.rank_sections_flat(q, depth, 500, 50, qvec=qv[i])}
    ctx = {"proxy_pointer": r.context_proxy, "flat_md": r.context_identity}
    rules = {"strict": lambda s: r.context_identity, "context": lambda s: ctx[s]}
    metrics = ("R@1", "R@5", "MRR", "AllParts@5")

    # base[rule][system][metric][qid] = metric with retrieval always on
    base = {rule: {s: defaultdict(dict) for s in SYSTEMS} for rule in rules}
    per_q = []
    for i, q in enumerate(questions):
        for s in SYSTEMS:
            ranked = rank[s](i, q["question"])
            for rule, cfn in rules.items():
                m = rm.score_ranking(tree, ranked, gold[q["id"]], cfn(s), (1, 5), 5, 10,
                                     parts=parts[q["id"]])
                for k in metrics:
                    base[rule][s][k][q["id"]] = m[k]
                per_q.append({"id": q["id"], "category": q.get("category", ""), "system": s,
                              "rule": rule, **{k: m[k] for k in metrics},
                              **{f"route_{p}": routes[p][q["id"]] for p in POLICIES},
                              "triggers": ",".join(triggers(q["question"]))})

    cats = [c for c in CATEGORIES if any(q.get("category") == c for q in questions)]
    # sorted ids: same resampling order as retrieval_metrics.py, so the no-router CIs match it
    subsets = {"overall": sorted(q["id"] for q in questions)}
    for c in cats:
        subsets[c] = sorted(q["id"] for q in questions if q.get("category") == c)

    def vec(rule, s, k, p, ids):
        return [base[rule][s][k][i] if routes[p][i] == "deep" else 0.0 for i in ids]

    tables = {}
    boots = {}
    for rule in rules:
        for sub, ids in subsets.items():
            for p in POLICIES:
                deep_ids = [i for i in ids if routes[p][i] == "deep"]
                for s in SYSTEMS:
                    key = f"{rule}|{sub}|{p}|{s}"
                    tables[key] = {"n": len(ids), "n_deep": len(deep_ids),
                                   **{k: float(np.mean(vec(rule, s, k, p, ids))) for k in metrics},
                                   **{f"deep_only_{k}": (float(np.mean([base[rule][s][k][i]
                                                                        for i in deep_ids]))
                                                         if deep_ids else float("nan"))
                                      for k in metrics}}
                for k in metrics:
                    boots[f"{rule}|{sub}|{p}|proxy-flat|{k}"] = bootstrap(
                        vec(rule, "proxy_pointer", k, p, ids), vec(rule, "flat_md", k, p, ids),
                        args.bootstrap, args.seed)
                    if deep_ids:
                        boots[f"{rule}|{sub}|{p}|proxy-flat|deep_only_{k}"] = bootstrap(
                            [base[rule]["proxy_pointer"][k][i] for i in deep_ids],
                            [base[rule]["flat_md"][k][i] for i in deep_ids],
                            args.bootstrap, args.seed)
                    if p != "none":
                        for s in SYSTEMS:
                            boots[f"{rule}|{sub}|{s}|{p}-none|{k}"] = bootstrap(
                                vec(rule, s, k, p, ids), vec(rule, s, k, "none", ids),
                                args.bootstrap, args.seed)

    summary = {"config": {"md": [os.path.basename(m) for m in args.md],
                          "questions": os.path.basename(args.questions),
                          "labels": os.path.basename(args.labels),
                          "gold": os.path.basename(args.gold) if args.gold else "inline",
                          "chunk": "500/50", "proxy_mode": "title_firstline", "mrr_depth": depth,
                          "bootstrap": args.bootstrap, "seed": args.seed,
                          "fast_route_metric": "0 (no retrieval)"},
               "router_accuracy": acc, "tables": tables, "bootstrap": boots,
               "environment": rm.environment("cpu", rm.EMBED_MODEL, args.seed)}
    _dump(os.path.join(args.out_dir, "router_eval_summary.json"), summary)
    with open(os.path.join(args.out_dir, "router_eval_per_question.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)
    md = render_markdown(summary, cats)
    with open(os.path.join(args.out_dir, "router_eval_tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def render_markdown(summary, cats):
    acc, T, B = summary["router_accuracy"], summary["tables"], summary["bootstrap"]
    L = []
    L.append("## Router accuracy\n")
    L.append("| Reference | n | Accuracy [95% CI] | kappa | fast P / R | deep P / R |")
    L.append("|---|---|---|---|---|---|")
    for name, key in (("hand labels", "router_vs_hand_labels"),
                      ("category oracle", "router_vs_category_oracle")):
        a = acc[key]["overall"]
        L.append(f"| {name} | {a['n']} | {a['accuracy']:.3f} [{a['accuracy_ci'][0]:.3f}, "
                 f"{a['accuracy_ci'][1]:.3f}] | {a['kappa']:.2f} | {a['fast_precision']:.2f} / "
                 f"{a['fast_recall']:.2f} | {a['deep_precision']:.2f} / {a['deep_recall']:.2f} |")
    a = acc["category_oracle_vs_hand_labels"]["overall"]
    L.append(f"| (category oracle vs hand labels) | {a['n']} | {a['accuracy']:.3f} | "
             f"{a['kappa']:.2f} | {a['fast_precision']:.2f} / {a['fast_recall']:.2f} | "
             f"{a['deep_precision']:.2f} / {a['deep_recall']:.2f} |")
    L.append("\nConfusion matrix, router vs hand labels (rows = hand label, cols = router):\n")
    c = acc["router_vs_hand_labels"]["overall"]["confusion"]
    L.append("| | router fast | router deep |\n|---|---|---|")
    L.append(f"| label fast | {c['true_fast|pred_fast']} | {c['true_fast|pred_deep']} |")
    L.append(f"| label deep | {c['true_deep|pred_fast']} | {c['true_deep|pred_deep']} |")
    L.append("\nPer category (router vs hand labels; router vs category oracle):\n")
    L.append("| Category | n | labels fast/deep | router fast/deep | acc vs labels | "
             "acc vs category oracle |")
    L.append("|---|---|---|---|---|---|")
    for cat in cats:
        h, o = acc["router_vs_hand_labels"][cat], acc["router_vs_category_oracle"][cat]
        hc = h["confusion"]
        lf = hc["true_fast|pred_fast"] + hc["true_fast|pred_deep"]
        rf = hc["true_fast|pred_fast"] + hc["true_deep|pred_fast"]
        L.append(f"| {cat} | {h['n']} | {lf}/{h['n'] - lf} | {rf}/{h['n'] - rf} | "
                 f"{h['accuracy']:.3f} | {o['accuracy']:.3f} |")
    mr = acc["misroutes"]
    L.append("\nSpurious deep routes (label fast, router deep) by trigger: "
             + ", ".join(f"`{p}` {n}" for p, n in mr["spurious_by_trigger"]))
    L.append("\nMissed deep (label deep, router fast) by category: "
             + ", ".join(f"{c} {n}" for c, n in mr["missed_deep_by_category"]))
    L.append("\nMissed deep, leading words: "
             + ", ".join(f"\"{w}\" {n}" for w, n in mr["missed_deep_leading_words"]))
    L.append("\nTrigger firing (on hand-deep / hand-fast questions): "
             + ", ".join(f"`{p}` {v['fires_on_deep']}/{v['fires_on_fast']}"
                         for p, v in mr["trigger_counts"].items()))

    has_mh = "multihop" in cats
    for rule in ("strict", "context"):
        L.append(f"\n## Retrieval 2x2, {rule} rule (fast route = no retrieval = 0)\n")
        hdr = "| Routing | n deep | System | R@1 | R@5 | MRR |" + (" AllParts@5 (multihop) |" if has_mh else "")
        L.append(hdr)
        L.append("|---|---|---|---|---|---|" + ("---|" if has_mh else ""))
        for p in POLICIES:
            for s in SYSTEMS:
                t = T[f"{rule}|overall|{p}|{s}"]
                row = (f"| {POLICY_NAMES[p]} | {t['n_deep']}/{t['n']} | {SYSTEM_NAMES[s]} | "
                       f"{t['R@1']:.3f} | {t['R@5']:.3f} | {t['MRR']:.3f} |")
                if has_mh:
                    row += f" {T[f'{rule}|multihop|{p}|{s}']['AllParts@5']:.3f} |"
                L.append(row)
        L.append("\nproxy - flat within each row (paired bootstrap 95% CI):\n")
        L.append("| Routing | Δ R@1 | Δ R@5 | Δ MRR |" + (" Δ AllParts@5 (multihop) |" if has_mh else ""))
        L.append("|---|---|---|---|" + ("---|" if has_mh else ""))
        for p in POLICIES:
            row = f"| {POLICY_NAMES[p]} | " + " | ".join(
                fmt_ci(B[f"{rule}|overall|{p}|proxy-flat|{k}"]) for k in ("R@1", "R@5", "MRR")) + " |"
            if has_mh:
                row += f" {fmt_ci(B[f'{rule}|multihop|{p}|proxy-flat|AllParts@5'])} |"
            L.append(row)
        L.append("\nrouting - no router within each column (paired bootstrap 95% CI):\n")
        L.append("| Routing | System | Δ R@1 | Δ R@5 | Δ MRR |" + (" Δ AllParts@5 (multihop) |" if has_mh else ""))
        L.append("|---|---|---|---|---|" + ("---|" if has_mh else ""))
        for p in POLICIES[1:]:
            for s in SYSTEMS:
                row = f"| {POLICY_NAMES[p]} | {SYSTEM_NAMES[s]} | " + " | ".join(
                    fmt_ci(B[f"{rule}|overall|{s}|{p}-none|{k}"]) for k in ("R@1", "R@5", "MRR")) + " |"
                if has_mh:
                    row += f" {fmt_ci(B[f'{rule}|multihop|{s}|{p}-none|AllParts@5'])} |"
                L.append(row)
        L.append("\nRestricted to deep-routed questions (what the retriever does when it runs):\n")
        L.append("| Routing | n deep | System | R@1 | R@5 | MRR | Δ MRR proxy-flat [95% CI] |")
        L.append("|---|---|---|---|---|---|---|")
        for p in POLICIES:
            for s in SYSTEMS:
                t = T[f"{rule}|overall|{p}|{s}"]
                d = (fmt_ci(B[f"{rule}|overall|{p}|proxy-flat|deep_only_MRR"])
                     if s == "proxy_pointer" else "")
                L.append(f"| {POLICY_NAMES[p]} | {t['n_deep']} | {SYSTEM_NAMES[s]} | "
                         f"{t['deep_only_R@1']:.3f} | {t['deep_only_R@5']:.3f} | "
                         f"{t['deep_only_MRR']:.3f} | {d} |")
        L.append("\nPer category, MRR (no router / regex router / oracle router):\n")
        L.append("| Category | n | flat | proxy |")
        L.append("|---|---|---|---|")
        for cat in cats:
            cells = []
            for s in SYSTEMS:
                cells.append(" / ".join(f"{T[f'{rule}|{cat}|{p}|{s}']['MRR']:.3f}"
                                        for p in ("none", "router", "oracle")))
            L.append(f"| {cat} | {T[f'{rule}|{cat}|none|flat_md']['n']} | {cells[0]} | {cells[1]} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Answer level: contexts -> generate -> answers
# --------------------------------------------------------------------------- #
def cmd_contexts(args):
    import retrieval_metrics as rm
    questions = load_jsonl(args.questions)
    r = rm.SectionRetriever(args.md, device="cpu")
    qv = r.encode_queries([q["question"] for q in questions])
    rows = []
    for i, q in enumerate(questions):
        chunks = r.rank_chunks_flat(q["question"], 4, 500, 50, qvec=qv[i])
        flat = CTX_SEP.join(c["text"] for c in chunks)
        top = r.rank_sections_proxy(q["question"], 1, qvec=qv[i])[0]
        node = r.tree.nodes[top]
        parts = [f"# {node.heading_title}"]
        if node.body_content.strip():
            parts.append(node.body_content.strip())
        for cid in node.child_node_ids:
            ch = r.tree.nodes[cid]
            parts.append(f"## {ch.heading_title}")
            if ch.body_content.strip():
                parts.append(ch.body_content.strip())
        rows.append({"id": q["id"], "flat_md": flat, "proxy_pointer": "\n\n".join(parts)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out} ({len(rows)} questions)")


SYSTEM_PROMPT = ("{intro} Answer the operator's question concisely and accurately. "
                 "Use the context below when it is relevant.")
NO_CONTEXT = "(no context retrieved)"


def build_messages(intro, question, context):
    return [{"role": "system", "content": SYSTEM_PROMPT.format(intro=intro)},
            {"role": "user", "content": f"Context:\n{context or NO_CONTEXT}\n\nQuestion: {question}"}]


def needed_generations(questions, labels, caps):
    """(qid, source, cap) triples. source in none|flat_md|proxy_pointer. `none` is needed
    only where some routing policy sends the question fast; it ignores the cap."""
    routes = routes_for(questions, labels)
    jobs = []
    for q in questions:
        if any(routes[p][q["id"]] == "fast" for p in ("router", "oracle")):
            jobs.append((q["id"], "none", None))
    for cap in caps:
        for s in SYSTEMS:
            for q in questions:
                jobs.append((q["id"], s, cap))
    return jobs


def _cap_key(cap):
    return "none" if cap is None else str(cap)


def _wait_for(pattern, poll_s=30):
    """Block while a process matching `pattern` runs (e.g. a latency benchmark)."""
    if not pattern:
        return
    import subprocess
    import time
    def running():
        # a process whose argv names `pattern` as a script; skips this process and any
        # command line that only mentions it as the value of --pause-while-running
        ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            pid, _, cmd = line.strip().partition(" ")
            if int(pid) == os.getpid():
                continue
            toks = cmd.split()
            if any(os.path.basename(t) == pattern and (i == 0 or toks[i - 1] != "--pause-while-running")
                   for i, t in enumerate(toks)):
                return True
        return False

    while running():
        print(f"paused: '{pattern}' is running", flush=True)
        time.sleep(poll_s)


def cmd_generate(args):
    if "faiss" in sys.modules:
        raise SystemExit("generate must run in a process that has not imported faiss")
    from corpus_config import assistant_intro
    questions = load_jsonl(args.questions)
    labels = load_labels(args.labels, questions)
    contexts = {r["id"]: r for r in load_jsonl(args.contexts)}
    caps = [None if c == "none" else int(c) for c in args.caps]
    jobs = needed_generations(questions, labels, caps)
    done = {}
    if os.path.exists(args.out):
        for r in load_jsonl(args.out):
            done[(r["id"], r["source"], r["cap"])] = r
    qtext = {q["id"]: q["question"] for q in questions}
    todo = [j for j in jobs if (j[0], j[1], _cap_key(j[2])) not in done]
    print(f"{len(jobs)} generations needed, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run", flush=True)
    if args.dry_run or not todo:
        return
    import time
    import torch
    from transformers import pipeline
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    pipe = pipeline("text-generation", model=args.model, torch_dtype=torch.float32,
                    device_map="cpu")
    intro = assistant_intro()
    t_start = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for n, (qid, src, cap) in enumerate(todo, 1):
            _wait_for(args.pause_while_running)
            ctx = "" if src == "none" else contexts[qid][src]
            if cap is not None:
                ctx = ctx[:cap]
            t0 = time.time()
            out = pipe(build_messages(intro, qtext[qid], ctx), max_new_tokens=args.max_new_tokens,
                       do_sample=False)
            ans = out[0]["generated_text"][-1]["content"].strip()
            rec = {"id": qid, "source": src, "cap": _cap_key(cap), "answer": ans,
                   "context_chars": len(ctx), "latency_s": round(time.time() - t0, 2),
                   "model": args.model, "max_new_tokens": args.max_new_tokens}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t_start
            print(f"[{n}/{len(todo)}] q={qid} {src} cap={_cap_key(cap)} {rec['latency_s']}s "
                  f"(elapsed {el / 60:.1f} min, eta {el / n * (len(todo) - n) / 60:.1f} min)",
                  flush=True)
            if args.budget_min and el / 60 > args.budget_min:
                print(f"budget of {args.budget_min} min exceeded; stopping (resumable)", flush=True)
                break


def cmd_answers(args):
    from sentence_transformers import SentenceTransformer
    questions = load_jsonl(args.questions)
    labels = load_labels(args.labels, questions)
    routes = routes_for(questions, labels)
    gens = {(r["id"], r["source"], r["cap"]): r["answer"] for r in load_jsonl(args.generations)}
    emb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def cos(a, b):
        v = emb.encode([a, b], normalize_embeddings=True)
        return round(float(np.dot(v[0], v[1])), 4)

    ids = [q["id"] for q in questions]
    ref = {q["id"]: q["reference_answer"] for q in questions}
    cat = {q["id"]: q.get("category", "") for q in questions}
    cache = {}

    def score(qid, src, cap):
        key = (qid, src, "none" if src == "none" else cap)
        if key not in gens:
            return None
        if key not in cache:
            cache[key] = cos(ref[qid], gens[key])
        return cache[key]

    out = {"config": {"generations": os.path.basename(args.generations),
                      "metric": "cosine(MiniLM(reference), MiniLM(answer)), normalised, 4 dp",
                      "bootstrap": args.bootstrap, "seed": args.seed},
           "cells": {}, "bootstrap": {}}
    caps = sorted({c for (_, s, c) in gens if s != "none"})
    for cap in caps:
        cell = {}
        for p in ("none", "router", "oracle"):
            for s in SYSTEMS:
                v = [score(i, s if routes[p][i] == "deep" else "none", cap) for i in ids]
                if any(x is None for x in v):
                    continue
                cell[(p, s)] = v
                out["cells"][f"{cap}|{p}|{s}"] = {
                    "n": len(v), "mean": float(np.mean(v)),
                    **{c: float(np.mean([x for x, i in zip(v, ids) if cat[i] == c]))
                       for c in sorted(set(cat.values()))},
                    "n_fast": sum(routes[p][i] == "fast" for i in ids)}
        for p in ("none", "router", "oracle"):
            if (p, "proxy_pointer") in cell and (p, "flat_md") in cell:
                out["bootstrap"][f"{cap}|{p}|proxy-flat"] = bootstrap(
                    cell[(p, "proxy_pointer")], cell[(p, "flat_md")], args.bootstrap, args.seed)
            if p != "none":
                for s in SYSTEMS:
                    if (p, s) in cell and ("none", s) in cell:
                        out["bootstrap"][f"{cap}|{s}|{p}-none"] = bootstrap(
                            cell[(p, s)], cell[("none", s)], args.bootstrap, args.seed)
        # fast-routed subsets: bare LLM vs retrieval on exactly the questions a policy skips
        for p in ("router", "oracle"):
            fast_ids = [i for i in ids if routes[p][i] == "fast"]
            for s in SYSTEMS:
                a = [score(i, "none", cap) for i in fast_ids]
                b = [score(i, s, cap) for i in fast_ids]
                if fast_ids and None not in a and None not in b:
                    out["bootstrap"][f"{cap}|{p}_fast_subset|none-{s}"] = bootstrap(
                        a, b, args.bootstrap, args.seed)
                    out["cells"][f"{cap}|{p}_fast_subset|{s}"] = {"n": len(b), "mean": float(np.mean(b))}
                    out["cells"][f"{cap}|{p}_fast_subset|none"] = {"n": len(a), "mean": float(np.mean(a))}
    os.makedirs(args.out_dir, exist_ok=True)
    _dump(os.path.join(args.out_dir, "answer_level_summary.json"), out)
    for k, v in out["cells"].items():
        print(f"{k:<40} n={v['n']:<3} mean={v['mean']:.4f}")
    for k, v in out["bootstrap"].items():
        print(f"{k:<40} {fmt_ci(v)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, md=True):
        if md:
            p.add_argument("--md", nargs="+", required=True, help="corpus markdown files")
        p.add_argument("--questions", required=True)
        p.add_argument("--bootstrap", type=int, default=10000)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("retrieval", help="router accuracy + retrieval-level 2x2")
    common(p)
    p.add_argument("--labels", required=True, help="hand router labels (id, label)")
    p.add_argument("--gold", default=None, help="gold JSONL; default: inline `gold` in --questions")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--accuracy-only", action="store_true", help="router accuracy only, no embedding")
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser("contexts", help="build generator contexts for flat and proxy")
    common(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_contexts)

    p = sub.add_parser("generate", help="answer-level generations (no faiss in this process)")
    common(p, md=False)
    p.add_argument("--labels", required=True)
    p.add_argument("--contexts", required=True)
    p.add_argument("--out", required=True, help="generations JSONL (appended; resumable)")
    p.add_argument("--caps", nargs="+", default=["1200"], help="context char caps, or 'none'")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = library default)")
    p.add_argument("--budget-min", type=float, default=0, help="stop after this many minutes")
    p.add_argument("--pause-while-running", default="",
                   help="pgrep -f pattern; generation pauses while it matches")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("answers", help="score generations, answer-level 2x2")
    common(p, md=False)
    p.add_argument("--labels", required=True)
    p.add_argument("--generations", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_answers)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
