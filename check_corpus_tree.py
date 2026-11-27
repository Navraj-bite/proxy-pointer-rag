#!/usr/bin/env python3
"""
Sanity report for the heading tree of the selected corpus.

Builds the DocumentTree exactly as proxy_pointer_indexer.py does and reports
the numbers that decide whether "return the whole section" is meaningful:
node counts, depth histogram, empty bodies, table / numbered-step nodes,
oversized sections, short / duplicate headings, skipped heading levels.
Optionally validates a question file's `gold` (doc, heading_path) labels.

    PPRAG_CORPUS=public python3 check_corpus_tree.py
    PPRAG_CORPUS=public python3 check_corpus_tree.py --questions questions/public_seed.jsonl
    PPRAG_CORPUS=public python3 check_corpus_tree.py --show 5     # spot-check nodes

Stdlib only (no models); safe to run on freshly fetched data.
"""

import argparse
import collections
import json
import os
import random
import re
import statistics
import sys

# Make the repo's flat modules importable even under `python3 -I`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus_config import get_config                     # noqa: E402
from document_tree_builder import (                       # noqa: E402
    NUMBERED_STEP_RE, DocumentTree, doc_id_of, find_by_heading_path,
    heading_path, validate_tree,
)

TABLE_DELIM_RE = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$", re.M)
LONG_BODY = 8000          # chars; a "section" bigger than this is not a useful unit
SHORT_HEADING = 3         # chars


def is_content(tree, n):
    return n.node_id != tree.root.node_id and not n.heading_title.startswith("[FILE] ")


def assembled_len(tree, n):
    """Length of what context_assembler.query() returns for this node."""
    parts = [f"# {n.heading_title}"]
    if n.body_content.strip():
        parts.append(n.body_content.strip())
    for cid in n.child_node_ids:
        c = tree.nodes[cid]
        parts.append(f"## {c.heading_title}")
        if c.body_content.strip():
            parts.append(c.body_content.strip())
    return len("\n\n".join(parts))


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", help="jsonl with a `gold` list to validate")
    ap.add_argument("--show", type=int, default=0, help="print N random nodes for eyeballing")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = get_config()
    tree = DocumentTree()
    for path in cfg["input_files"]:
        tree.add_file(path)
    content = [n for n in tree.nodes.values() if is_content(tree, n)]

    print(f"corpus            : {cfg['name']}")
    print(f"input files       : {len(cfg['input_files'])}")
    total_bytes = sum(os.path.getsize(p) for p in cfg["input_files"])
    print(f"markdown bytes    : {total_bytes:,}")
    errors, visited = validate_tree(tree)
    print(f"total nodes       : {len(tree.nodes)} (ROOT + {len(cfg['input_files'])} [FILE] + {len(content)} content)")
    print(f"integrity errors  : {len(errors)}")
    for e in errors[:10]:
        print("   - " + e)

    per_file = collections.Counter(doc_id_of(tree, n) for n in content)
    print("\ncontent nodes per file:")
    for p in cfg["input_files"]:
        print(f"  {per_file[p]:5d}  {p}")

    depth = collections.Counter(n.level for n in content)
    print("\ndepth histogram (heading level -> nodes):")
    for lvl in sorted(depth):
        print(f"  h{lvl}: {depth[lvl]:5d}  " + "#" * max(1, depth[lvl] // 10))

    empty = [n for n in content if not n.body_content.strip()]
    empty_leaf = [n for n in empty if not n.child_node_ids]
    tables = [n for n in content if TABLE_DELIM_RE.search(n.body_content)]
    steps = [n for n in content
             if any(NUMBERED_STEP_RE.match(l) for l in n.body_content.splitlines())]
    lens = [len(n.body_content) for n in content]
    asm = [assembled_len(tree, n) for n in content]
    long_nodes = [n for n in content if len(n.body_content) > LONG_BODY]
    long_asm = [n for n, a in zip(content, asm) if a > LONG_BODY]

    print("\nbody content:")
    print(f"  empty body              : {len(empty)} ({len(empty_leaf)} of them leaves = no content at all)")
    print(f"  table-bearing nodes     : {len(tables)}")
    print(f"  numbered-step nodes     : {len(steps)}")
    print(f"  body chars median/p90/max: {int(statistics.median(lens))} / {pct(lens, .9)} / {max(lens)}")
    print(f"  assembled context (node + direct children) median/p90/max: "
          f"{int(statistics.median(asm))} / {pct(asm, .9)} / {max(asm)}")
    print(f"  body > {LONG_BODY} chars       : {len(long_nodes)}")
    for n in long_nodes:
        print(f"     {len(n.body_content):6d}  {doc_id_of(tree, n)} :: {' > '.join(heading_path(tree, n))}")
    print(f"  assembled > {LONG_BODY} chars  : {len(long_asm)}")

    titles = collections.Counter(n.heading_title for n in content)
    short = [n for n in content if len(n.heading_title) < SHORT_HEADING]
    dups = {t: c for t, c in titles.items() if c > 1}
    paths = collections.Counter((doc_id_of(tree, n), tuple(heading_path(tree, n))) for n in content)
    dup_paths = {k: c for k, c in paths.items() if c > 1}
    skipped = []
    for n in content:
        parent = tree.nodes[n.parent_node_id]
        plevel = parent.level if is_content(tree, parent) else 0
        if n.level > plevel + 1:
            skipped.append((n, plevel))

    print("\nheadings:")
    print(f"  very short (<{SHORT_HEADING} chars)  : {len(short)}" +
          (f"  e.g. {[n.heading_title for n in short[:5]]}" if short else ""))
    print(f"  duplicate titles        : {len(dups)} titles used by {sum(dups.values())} nodes")
    for t, c in sorted(dups.items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
        print(f"     {c:3d}x  {t}")
    print(f"  duplicate (doc, heading_path): {len(dup_paths)}")
    for (d, p), c in list(dup_paths.items())[:5]:
        print(f"     {c}x  {d} :: {' > '.join(p)}")
    print(f"  skipped heading levels  : {len(skipped)}")
    for n, pl in skipped[:5]:
        print(f"     h{pl} -> h{n.level}: {' > '.join(heading_path(tree, n))}")

    if args.show:
        rng = random.Random(args.seed)
        print(f"\n=== spot check: {args.show} random content nodes (seed {args.seed}) ===")
        for n in rng.sample(content, args.show):
            print(f"\n--- {n.node_id}  {doc_id_of(tree, n)} lines {n.start_line}-{n.end_line}")
            print(f"    path: {' > '.join(heading_path(tree, n))}")
            print("    " + n.body_content[:400].replace("\n", "\n    "))

    bad = 0
    if args.questions:
        print(f"\n=== gold validation: {args.questions} ===")
        with open(args.questions, encoding="utf-8") as fh:
            qs = [json.loads(l) for l in fh if l.strip()]
        cats = collections.Counter(q["category"] for q in qs)
        print(f"  {len(qs)} questions  {dict(cats)}")
        for q in qs:
            for g in q.get("gold", []):
                hits = find_by_heading_path(tree, g["doc"], g["heading_path"])
                status = "ok" if len(hits) == 1 else ("MISSING" if not hits else f"AMBIGUOUS x{len(hits)}")
                if len(hits) != 1:
                    bad += 1
                kw = q.get("keywords", [])
                miss = []
                if hits:
                    text = (hits[0].heading_title + "\n" + hits[0].body_content).lower()
                    miss = [k for k in kw if k.lower() not in text]
                    if miss:
                        bad += 1
                        status += f" keywords-not-in-gold={miss}"
                print(f"  Q{q['id']:>3} [{q['category']:<10}] {status:<4}  "
                      f"{g['doc']} :: {' > '.join(g['heading_path'])}")
        print(f"  problems: {bad}")

    return 1 if (errors or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
