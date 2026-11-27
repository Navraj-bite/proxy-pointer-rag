"""
Retrieval metrics — measure retrieval directly instead of via answer similarity.

Scores two retrievers on the same questions, against gold section labels, with
the same metric code:

  * proxy-pointer : one proxy embedding per heading node (default proxy =
                    heading + first meaningful body line, exactly as
                    proxy_pointer_indexer.py builds it); nodes ranked by
                    proxy score.
  * flat-chunk    : RecursiveCharacterTextSplitter(500/50) over the SAME
                    markdown files the tree is built from; every retrieved
                    chunk is mapped back to the section node(s) it overlaps,
                    so both systems emit a ranked list of section node ids.
  * legacy-pypdf  : (optional, --legacy-chunks/--legacy-index) the original
                    faiss.index/chunks.pkl built from pypdf text, with each
                    chunk fuzzy-mapped to a markdown section. Coverage is
                    reported; unmapped chunks contribute nothing.

Relevance rules
---------------
  G = set of gold node ids for a question (resolved from heading paths).

  strict  (primary)  : a retrieved section s is relevant iff s in G.
  context (secondary): s is relevant iff the context the system would actually
                       hand the generator for s contains a gold section.
                       proxy-pointer: context_assembler returns s + its direct
                       children, so ctx(s) = {s} U children(s) and a hit on
                       the parent of a gold node counts.
                       flat-chunk / legacy: the context is the chunk itself,
                       which already maps to the section(s) it overlaps, so
                       ctx(s) = {s} and the context rule equals strict.
  A third, symmetric "context_parent" sensitivity row applies the proxy
  context function to every system (parent credit for all) so nobody has to
  take the asymmetry on trust.

Metrics (per question, then averaged)
-------------------------------------
  Recall@K : hit rate — 1 if at least one relevant section is in the top K,
             else 0. Gold sets list *alternative* sections that each contain
             the answer (e.g. the same table in two manuals), so "any gold in
             top K" is the right notion; it is NOT the fraction of gold found.
  MRR      : 1 / rank of the first relevant section within the top
             --mrr-depth (default 10) sections; 0 if none.
  nDCG@K   : binary relevance, DCG = sum rel_i / log2(i+1) over the top K;
             IDCG assumes min(R, K) relevant sections at the top, where R is
             the number of distinct indexable sections that are relevant
             under the rule (|G| for strict).

Chunk -> section mapping rule (flat baseline)
---------------------------------------------
  Each chunk's character span [start, start+len) in its source file is
  converted to a line range. Heading nodes partition a file into disjoint line
  spans (start_line..end_line, every line belongs to the most recent heading,
  i.e. the deepest containing node), so the chunk maps to every content node
  whose span it overlaps, ordered by overlapping characters (desc), ties in
  document order. Lines before a file's first heading belong to the synthetic
  [FILE] node and map to nothing. The ranked section list is the
  concatenation over chunks in score order, de-duplicated keeping the first
  occurrence. A chunk straddling a boundary therefore puts several sections
  at (consecutive) ranks; K always counts sections, not chunks.

Gold labels
-----------
  JSONL, one record per question id:
    {"id": 1,
     "gold": [{"doc": "manual.md", "heading_path": ["2. SETUP", "2.1. Wiring"]}],
     "label_source": "agent-drafted", "verified": false, "notes": "..."}
  heading_path runs from the file root (titles exactly as in the markdown,
  compared after whitespace collapsing). An element may instead be
  {"title": "...", "occurrence": n} (1-based) to pick the n-th of several
  sibling headings with the same title. Paths are resolved to node ids at
  run time; a path that resolves to zero or to more than one node raises.
  See docs/retrieval-metrics.md.

Run (from a directory holding the corpus):
  python3 retrieval_metrics.py \\
      --md machine_overview_troubleshooting.md operator_manual.md \\
      --questions benchmark_questions.jsonl --gold gold_labels.jsonl \\
      --k 1 3 5 --out-dir retrieval_results
"""

import argparse
import bisect
import csv
import datetime
import json
import os
import pickle
import platform
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

_FAISS_BEFORE_TORCH = "faiss" in sys.modules and "torch" not in sys.modules

from document_tree_builder import DocumentTree

EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = (1, 3, 5)
CATEGORIES = ("tabular", "procedure", "navigation", "multihop")
SYSTEMS_PRIMARY = ("proxy_pointer", "flat_md")


# --------------------------------------------------------------------------- #
# Tree helpers
# --------------------------------------------------------------------------- #
def _norm(s):
    return " ".join(str(s).split())


def build_tree(md_files):
    tree = DocumentTree()
    for path in md_files:
        tree.add_file(path)
    return tree


def is_indexable(tree, node):
    """Same exclusion as proxy_pointer_indexer: no ROOT, no [FILE] containers."""
    return node.node_id != tree.root.node_id and not node.heading_title.startswith("[FILE] ")


def indexable_nodes(tree):
    return [n for n in tree.nodes.values() if is_indexable(tree, n)]


def file_node_of(tree, node_id):
    cur = tree.nodes[node_id]
    while cur is not None:
        if cur.heading_title.startswith("[FILE] "):
            return cur
        cur = tree.nodes[cur.parent_node_id] if cur.parent_node_id else None
    return None


def file_nodes(tree):
    """basename -> [FILE] node."""
    out = {}
    for nid in tree.nodes[tree.root.node_id].child_node_ids:
        n = tree.nodes[nid]
        out[os.path.basename(n.heading_title[len("[FILE] "):])] = n
    return out


def heading_path(tree, node_id):
    """(doc basename, [titles from file root down to node])."""
    titles = []
    cur = tree.nodes[node_id]
    while not cur.heading_title.startswith("[FILE] "):
        titles.append(cur.heading_title)
        cur = tree.nodes[cur.parent_node_id]
    return os.path.basename(cur.heading_title[len("[FILE] "):]), titles[::-1]


def section_label(tree, node_id):
    doc, path = heading_path(tree, node_id)
    return f"{doc}::{' > '.join(_norm(t) for t in path)}"


# --------------------------------------------------------------------------- #
# Gold labels
# --------------------------------------------------------------------------- #
class GoldResolutionError(ValueError):
    pass


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_heading_path(tree, doc, path):
    """Resolve (doc basename, heading path) to exactly one node id or raise."""
    fnodes = file_nodes(tree)
    if doc not in fnodes:
        raise GoldResolutionError(f"doc {doc!r} is not one of the corpus files {sorted(fnodes)}")
    if not path:
        raise GoldResolutionError(f"empty heading_path for doc {doc!r}")
    frontier = [fnodes[doc].node_id]
    for depth, elem in enumerate(path):
        if isinstance(elem, dict):
            title, occ = _norm(elem["title"]), int(elem.get("occurrence", 0))
        else:
            title, occ = _norm(elem), 0
        nxt = []
        for pid in frontier:
            kids = [c for c in tree.nodes[pid].child_node_ids
                    if _norm(tree.nodes[c].heading_title) == title]
            if occ:
                if occ > len(kids):
                    continue
                kids = [kids[occ - 1]]
            nxt.extend(kids)
        if not nxt:
            raise GoldResolutionError(
                f"{doc}: heading path {path!r} does not resolve at element {depth} ({title!r})")
        frontier = nxt
    if len(frontier) > 1:
        raise GoldResolutionError(
            f"{doc}: heading path {path!r} is ambiguous ({len(frontier)} nodes); "
            f"add {{'title': ..., 'occurrence': n}} to disambiguate")
    return frontier[0]


def resolve_gold(tree, gold_records, question_ids=None):
    """-> {qid: [node_id, ...]} ; raises on any unresolved/missing label."""
    out = {}
    for rec in gold_records:
        qid = rec["id"]
        if not rec.get("gold"):
            raise GoldResolutionError(f"question {qid}: empty gold list")
        ids = []
        for g in rec["gold"]:
            nid = resolve_heading_path(tree, os.path.basename(g["doc"]), g["heading_path"])
            if not is_indexable(tree, tree.nodes[nid]):
                raise GoldResolutionError(f"question {qid}: gold {g} is not an indexable section")
            if nid not in ids:
                ids.append(nid)
        out[qid] = ids
    if question_ids is not None:
        missing = sorted(set(question_ids) - set(out))
        extra = sorted(set(out) - set(question_ids))
        if missing or extra:
            raise GoldResolutionError(f"gold/question id mismatch: missing={missing} extra={extra}")
    return out


def load_gold(tree, questions, gold_path=None):
    """
    -> (gold_records, {qid: [node_id]}, {qid: [[alt node_id, ...], ...]}).
    Gold comes from *gold_path* (JSONL) or, if None, the `gold` field inline in
    *questions* (public-corpus format). Required parts (multi-hop): `gold_parts`
    = [[alt, alt, ...], [alt, ...]] per question; questions without it have one
    part = all their gold sections as alternatives.
    """
    gold_records = load_jsonl(gold_path) if gold_path else [
        {"id": q["id"], "gold": q.get("gold"), "label_source": q.get("label_source", ""),
         "verified": q.get("verified", False)} for q in questions]
    gold = resolve_gold(tree, gold_records, [q["id"] for q in questions])
    gold_parts = {}
    for q in questions:
        if q.get("gold_parts"):
            gold_parts[q["id"]] = [
                [resolve_heading_path(tree, os.path.basename(g["doc"]), g["heading_path"]) for g in part]
                for part in q["gold_parts"]]
        else:
            gold_parts[q["id"]] = [list(gold[q["id"]])]
    return gold_records, gold, gold_parts


# --------------------------------------------------------------------------- #
# Proxy construction (pluggable)
# --------------------------------------------------------------------------- #
def proxy_title_firstline(node):
    """Identical to the proxy built by proxy_pointer_indexer.build_proxy_index."""
    title = node.heading_title
    snippet = node.body_content.strip()
    if snippet:
        for line in snippet.splitlines():
            line = line.strip()
            if line and not line.startswith("|") and not line.startswith("#"):
                return f"{title}\n{line[:200]}"
        return title
    return title


def proxy_title(node):
    """Heading text only."""
    return node.heading_title


PROXY_FUNCS = {
    "title_firstline": proxy_title_firstline,
    "title": proxy_title,
}


# --------------------------------------------------------------------------- #
# Embedding / index builders
# --------------------------------------------------------------------------- #
def load_embedder(model=EMBED_MODEL, device="cpu"):
    # macOS/arm64: if faiss (its own libomp) was imported before torch, CPU
    # encoding with multi-threaded torch segfaults. Single-threaded torch avoids
    # it; this module itself imports faiss lazily, after torch, so the guard only
    # triggers for callers that imported faiss first.
    import torch
    if device == "cpu" and _FAISS_BEFORE_TORCH:
        torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model, device=device)


def _encode(embedder, texts, batch_size=64):
    return embedder.encode(list(texts), normalize_embeddings=True, batch_size=batch_size,
                           show_progress_bar=False).astype("float32")


def _flat_ip(vecs):
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def build_proxy_index(tree, proxy_fn=proxy_title_firstline, embedder=None):
    """
    Embed proxy_fn(node) for every indexable node. Nothing is written to disk.
    Returns (faiss.IndexFlatIP, node_ids, proxy_texts); row i <-> node_ids[i].
    """
    if isinstance(proxy_fn, str):
        proxy_fn = PROXY_FUNCS[proxy_fn]
    embedder = embedder or load_embedder()
    nodes = indexable_nodes(tree)
    texts = [proxy_fn(n) for n in nodes]
    return _flat_ip(_encode(embedder, texts)), [n.node_id for n in nodes], texts


def split_with_offsets(text, chunk_size=500, overlap=50):
    """RecursiveCharacterTextSplitter chunks + their start offsets in *text*."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    out, prev_start, prev_len = [], 0, 0
    for chunk in splitter.split_text(text):
        idx = text.find(chunk, max(0, prev_start + prev_len - overlap))
        if idx < 0:
            idx = text.find(chunk)
        if idx < 0:
            raise RuntimeError("chunk not found in source text; cannot compute offsets")
        out.append((chunk, idx))
        prev_start, prev_len = idx, len(chunk)
    return out


def _file_line_spans(tree, file_node, text):
    """Line offsets and the (node_id, char_start, char_end) spans of content nodes."""
    line_starts = [0]
    for m in re.finditer("\n", text):
        line_starts.append(m.end())
    n_lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)

    def line_char_span(l0, l1):
        a = line_starts[l0 - 1]
        b = line_starts[l1] if l1 < len(line_starts) else len(text)
        return a, b

    spans = []
    stack = list(file_node.child_node_ids)
    while stack:
        nid = stack.pop()
        node = tree.nodes[nid]
        stack.extend(node.child_node_ids)
        if node.start_line < 1 or node.end_line < node.start_line:
            continue
        a, b = line_char_span(node.start_line, min(node.end_line, n_lines))
        spans.append((nid, a, b, node.start_line))
    spans.sort(key=lambda s: s[3])
    return spans


def map_span_to_sections(spans, start, end):
    """Section node ids overlapping char span [start, end), by overlap desc then doc order."""
    starts = [s[1] for s in spans]
    i = max(0, bisect.bisect_right(starts, start) - 1)
    hits = []
    while i < len(spans) and spans[i][1] < end:
        nid, a, b, line = spans[i]
        ov = min(b, end) - max(a, start)
        if ov > 0:
            hits.append((-ov, line, nid))
        i += 1
    return [nid for _, _, nid in sorted(hits)]


def build_flat_index(tree, md_files, chunk_size=500, overlap=50, embedder=None):
    """
    Flat chunk index over the same markdown the tree was built from.
    Returns (faiss index, chunks) where chunks[i] = {"text", "doc", "start",
    "sections": [node_id, ...]} (sections ordered per the mapping rule).
    """
    embedder = embedder or load_embedder()
    fnodes = file_nodes(tree)
    chunks = []
    for path in md_files:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        doc = os.path.basename(path)
        spans = _file_line_spans(tree, fnodes[doc], text)
        for chunk, start in split_with_offsets(text, chunk_size, overlap):
            chunks.append({"text": chunk, "doc": doc, "start": start,
                           "sections": map_span_to_sections(spans, start, start + len(chunk))})
    index = _flat_ip(_encode(embedder, [c["text"] for c in chunks]))
    return index, chunks


# --------------------------------------------------------------------------- #
# Legacy pypdf chunks -> markdown sections (fuzzy)
# --------------------------------------------------------------------------- #
_TOK = re.compile(r"[a-z0-9]+")
LEGACY_SHINGLE = 4
LEGACY_MIN_COVER = 0.5      # union share of a chunk's shingles explained by chosen sections
LEGACY_MIN_PART = 0.1       # a section must add at least this share to be chosen
LEGACY_TIE_MARGIN = 0.05    # sections within this of the best single-section share are also kept
LEGACY_MAX_DF = 5           # shingles found in more sections than this (page footers etc.) are ignored


def _shingles(text, n=LEGACY_SHINGLE):
    toks = _TOK.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class FuzzySectionMapper:
    """
    Map arbitrary text (e.g. pypdf chunks) to tree sections by word 4-gram
    shingle overlap. Greedy cover: repeatedly take the section that explains
    the most not-yet-explained shingles of the chunk while it adds at least
    min_part; the chunk is mapped iff the chosen sections jointly explain at
    least min_cover of its shingles (so boundary-straddling chunks map to all
    the sections they span, mirroring the flat_md rule). Sections whose own
    share is within tie_margin of the best one (duplicated content) are kept
    too. Shingles that occur in more than max_df sections (repeated page
    headers/footers) are ignored entirely: they carry no location signal.
    Unmapped chunks contribute nothing to a ranking.
    """

    def __init__(self, tree, min_cover=LEGACY_MIN_COVER, min_part=LEGACY_MIN_PART,
                 tie_margin=LEGACY_TIE_MARGIN, max_df=LEGACY_MAX_DF):
        self.min_cover, self.min_part, self.tie_margin = min_cover, min_part, tie_margin
        self.inv = defaultdict(set)
        self.order = {}
        for i, n in enumerate(indexable_nodes(tree)):
            self.order[n.node_id] = i
            for sh in _shingles(n.heading_title + "\n" + n.body_content):
                self.inv[sh].add(n.node_id)
        self.common = {k for k, v in self.inv.items() if len(v) > max_df}
        self.inv = {k: v for k, v in self.inv.items() if len(v) <= max_df}

    def map(self, text):
        """-> (sections, covered_share, n_shingles)."""
        sh = _shingles(text) - self.common
        if not sh:
            return [], 0.0, 0
        owned = defaultdict(set)
        for s in sh:
            for nid in self.inv.get(s, ()):
                owned[nid].add(s)
        if not owned:
            return [], 0.0, len(sh)
        n = len(sh)
        best_single = max(len(v) for v in owned.values()) / n
        chosen, covered = [], set()
        while True:
            gain, nid = max(((len(v - covered), nid) for nid, v in owned.items() if nid not in chosen),
                            key=lambda t: (t[0], -self.order[t[1]]), default=(0, None))
            if nid is None or gain / n < self.min_part:
                break
            chosen.append(nid)
            covered |= owned[nid]
        cover = len(covered) / n
        if cover < self.min_cover:
            return [], cover, n
        ties = [nid for nid, v in sorted(owned.items(), key=lambda t: self.order[t[0]])
                if nid not in chosen and len(v) / n >= best_single - self.tie_margin]
        return chosen + ties, cover, n


# --------------------------------------------------------------------------- #
# Retriever: the reusable hooks
# --------------------------------------------------------------------------- #
class SectionRetriever:
    """
    Holds a corpus tree + embedder and lazily builds/caches indices.

        r = SectionRetriever(["a.md", "b.md"])
        r.rank_sections_proxy("question", k=5)                # -> [node_id, ...]
        r.rank_sections_proxy("question", k=5, proxy_mode="title")
        r.rank_sections_proxy("question", k=5, proxy_mode=my_fn)  # callable(node)->str
        r.rank_sections_flat("question", k=5, chunk_size=500, overlap=50)
    """

    def __init__(self, md_files, embed_model=EMBED_MODEL, device="cpu", tree=None, embedder=None):
        self.md_files = list(md_files)
        self.embed_model = embed_model
        self.device = device
        self.tree = tree or build_tree(self.md_files)
        self.embedder = embedder or load_embedder(embed_model, device)
        self._proxy = {}
        self._flat = {}
        self._legacy = None

    # -- encoding ---------------------------------------------------------
    def encode_queries(self, questions):
        return _encode(self.embedder, questions)

    def _qvec(self, question, qvec):
        return qvec.reshape(1, -1) if qvec is not None else self.encode_queries([question])

    # -- proxy-pointer ----------------------------------------------------
    def proxy_index(self, proxy_mode="title_firstline"):
        key = proxy_mode if isinstance(proxy_mode, str) else id(proxy_mode)
        if key not in self._proxy:
            fn = PROXY_FUNCS[proxy_mode] if isinstance(proxy_mode, str) else proxy_mode
            self._proxy[key] = build_proxy_index(self.tree, fn, self.embedder)
        return self._proxy[key]

    def rank_sections_proxy(self, question, k, proxy_mode="title_firstline", qvec=None,
                            with_scores=False):
        index, node_ids, _ = self.proxy_index(proxy_mode)
        scores, ids = index.search(self._qvec(question, qvec), min(k, index.ntotal))
        ranked = [(node_ids[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        return ranked if with_scores else [n for n, _ in ranked]

    # -- flat chunks over the same markdown -------------------------------
    def flat_index(self, chunk_size=500, overlap=50):
        key = (chunk_size, overlap)
        if key not in self._flat:
            self._flat[key] = build_flat_index(self.tree, self.md_files, chunk_size, overlap,
                                               self.embedder)
        return self._flat[key]

    @staticmethod
    def _chunks_to_sections(index, chunk_sections, qv, k):
        depth = min(index.ntotal, max(50, 10 * k))
        while True:
            _, ids = index.search(qv, depth)
            ranked, seen = [], set()
            for i in ids[0]:
                if i < 0:
                    continue
                for nid in chunk_sections(int(i)):
                    if nid not in seen:
                        seen.add(nid)
                        ranked.append(nid)
            if len(ranked) >= k or depth >= index.ntotal:
                return ranked[:k]
            depth = min(index.ntotal, depth * 2)

    def rank_sections_flat(self, question, k, chunk_size=500, overlap=50, qvec=None):
        index, chunks = self.flat_index(chunk_size, overlap)
        return self._chunks_to_sections(index, lambda i: chunks[i]["sections"],
                                        self._qvec(question, qvec), k)

    def rank_chunks_flat(self, question, k, chunk_size=500, overlap=50, qvec=None):
        """Raw chunk hits (dicts with text/doc/start/sections), for inspection."""
        index, chunks = self.flat_index(chunk_size, overlap)
        _, ids = index.search(self._qvec(question, qvec), min(k, index.ntotal))
        return [chunks[i] for i in ids[0] if i >= 0]

    # -- legacy pypdf index -----------------------------------------------
    def load_legacy(self, chunks_pkl, index_file):
        import faiss
        with open(chunks_pkl, "rb") as f:
            texts = pickle.load(f)
        index = faiss.read_index(index_file)
        if index.ntotal != len(texts):
            raise ValueError(f"legacy index has {index.ntotal} vectors but {len(texts)} chunks")
        mapper = FuzzySectionMapper(self.tree)
        mapped = [mapper.map(t) for t in texts]
        self._legacy = (index, [m[0] for m in mapped], mapped)
        return self.legacy_coverage()

    def legacy_coverage(self):
        _, sections, mapped = self._legacy
        n = len(sections)
        n_mapped = sum(1 for s in sections if s)
        return {"n_chunks": n, "n_mapped": n_mapped, "coverage": n_mapped / n if n else 0.0,
                "n_multi": sum(1 for s in sections if len(s) > 1),
                "median_cover": float(np.median([m[1] for m in mapped])) if n else 0.0,
                "params": {"shingle": LEGACY_SHINGLE, "min_cover": LEGACY_MIN_COVER,
                           "min_part": LEGACY_MIN_PART, "tie_margin": LEGACY_TIE_MARGIN,
                           "max_df": LEGACY_MAX_DF}}

    def rank_sections_legacy(self, question, k, qvec=None):
        if self._legacy is None:
            raise RuntimeError("call load_legacy(chunks_pkl, index_file) first")
        index, sections, _ = self._legacy
        return self._chunks_to_sections(index, lambda i: sections[i], self._qvec(question, qvec), k)

    # -- context coverage diagnostics -----------------------------------
    def node_char_spans(self):
        """{node_id: (doc, char_start, char_end)} for every content node."""
        if getattr(self, "_spans", None) is None:
            fnodes = file_nodes(self.tree)
            self._spans = {}
            for path in self.md_files:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                doc = os.path.basename(path)
                for nid, a, b, _ in _file_line_spans(self.tree, fnodes[doc], text):
                    self._spans[nid] = (doc, a, b)
        return self._spans

    def context_coverage_proxy(self, question, gold, cap=None, proxy_mode="title_firstline",
                               qvec=None):
        """
        Share of the best gold section present in the context context_assembler
        builds from the top-1 proxy hit (node + direct children), optionally
        truncated to *cap* chars (hybrid_intent_router truncates to 1200).
        """
        top = self.rank_sections_proxy(question, 1, proxy_mode, qvec=qvec)[0]
        node = self.tree.nodes[top]
        pieces = [(top, node)] + [(c, self.tree.nodes[c]) for c in node.child_node_ids]
        offs, pos = {}, 0
        for i, (nid, n) in enumerate(pieces):
            part = ("# " if i == 0 else "## ") + n.heading_title
            if n.body_content.strip():
                part += "\n\n" + n.body_content.strip()
            if i:
                pos += 2  # "\n\n" separator
            offs[nid] = (pos, pos + len(part))
            pos += len(part)
        best = 0.0
        for g in gold:
            if g in offs:
                a, b = offs[g]
                kept = (b - a) if cap is None else max(0, min(b, cap) - a)
                best = max(best, kept / (b - a))
        return best

    def context_coverage_flat(self, question, gold, n_chunks=4, cap=None, chunk_size=500,
                              overlap=50, qvec=None):
        """
        Share of the best gold section's characters present in the top
        *n_chunks* flat chunks joined as naive_rag_baseline/run_day10 do
        ("\n\n---\n\n"), optionally truncated to *cap* chars.
        """
        spans = self.node_char_spans()
        hits = self.rank_chunks_flat(question, n_chunks, chunk_size, overlap, qvec=qvec)
        kept, pos = defaultdict(list), 0
        for i, c in enumerate(hits):
            if i:
                pos += len("\n\n---\n\n")
            keep = len(c["text"]) if cap is None else max(0, min(len(c["text"]), cap - pos))
            if keep:
                kept[c["doc"]].append((c["start"], c["start"] + keep))
            pos += len(c["text"])
        best = 0.0
        for g in gold:
            doc, a, b = spans[g]
            covered = np.zeros(b - a, dtype=bool)
            for s, e in kept.get(doc, []):
                lo, hi = max(a, s), min(b, e)
                if hi > lo:
                    covered[lo - a:hi - a] = True
            best = max(best, float(covered.mean()) if b > a else 0.0)
        return best

    # -- relevance --------------------------------------------------------
    def context_proxy(self, node_id):
        """What context_assembler returns for a hit: the node + direct children."""
        return {node_id, *self.tree.nodes[node_id].child_node_ids}

    @staticmethod
    def context_identity(node_id):
        return {node_id}


# Module-level convenience wrappers around a default retriever.
_default = None


def init(md_files, **kw):
    global _default
    _default = SectionRetriever(md_files, **kw)
    return _default


def _need_default():
    if _default is None:
        raise RuntimeError("call retrieval_metrics.init(md_files) first")
    return _default


def rank_sections_proxy(question, k, proxy_mode="title_firstline"):
    return _need_default().rank_sections_proxy(question, k, proxy_mode=proxy_mode)


def rank_sections_flat(question, k, chunk_size=500, overlap=50):
    return _need_default().rank_sections_flat(question, k, chunk_size=chunk_size, overlap=overlap)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def relevance_vector(ranked, gold, context_fn):
    gold = set(gold)
    return [1 if context_fn(s) & gold else 0 for s in ranked]


def n_relevant_possible(tree, gold, context_fn):
    """Distinct indexable sections whose context contains a gold node."""
    gold = set(gold)
    cands = set(gold)
    for g in gold:
        p = tree.nodes[g].parent_node_id
        if p:
            cands.add(p)
    return sum(1 for c in cands if is_indexable(tree, tree.nodes[c]) and context_fn(c) & gold)


def recall_at_k(rel, k):
    return 1.0 if any(rel[:k]) else 0.0


def reciprocal_rank(rel, depth):
    for i, r in enumerate(rel[:depth], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel, k, n_rel):
    dcg = sum(r / np.log2(i + 1) for i, r in enumerate(rel[:k], start=1))
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(n_rel, k) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


def all_parts_at_k(ranked, parts, context_fn, k):
    """1.0 iff every required part has at least one of its alternative gold sections
    covered by the top-k results (multi-hop "found everything" recall)."""
    covered = set()
    for s in ranked[:k]:
        covered |= context_fn(s)
    return 1.0 if all(covered & set(p) for p in parts) else 0.0


def score_ranking(tree, ranked, gold, context_fn, ks=DEFAULT_K, ndcg_k=5, mrr_depth=10,
                  parts=None):
    """`gold` is the union of all gold sections (any hit = relevant). `parts` (optional) is a
    list of required parts, each a list of alternative node ids; it adds AllParts@k. When a
    question has a single part (every non-multihop question), AllParts@k == R@k."""
    rel = relevance_vector(ranked, gold, context_fn)
    out = {f"R@{k}": recall_at_k(rel, k) for k in ks}
    parts = parts or [list(gold)]
    for k in ks:
        out[f"AllParts@{k}"] = all_parts_at_k(ranked, parts, context_fn, k)
    out["MRR"] = reciprocal_rank(rel, mrr_depth)
    out[f"nDCG@{ndcg_k}"] = ndcg_at_k(rel, ndcg_k, n_relevant_possible(tree, gold, context_fn))
    first = next((i for i, r in enumerate(rel[:mrr_depth], start=1) if r), None)
    out["first_rank"] = first if first is not None else ""
    return out


def paired_bootstrap(a, b, n_boot=10000, seed=0, ci=0.95):
    """Percentile CI for mean(a - b) by resampling questions with replacement."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean_delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "crosses_zero": bool(lo <= 0 <= hi), "n": int(len(d)), "n_boot": n_boot,
            "seed": seed, "p_delta_le_0": float((means <= 0).mean())}


# --------------------------------------------------------------------------- #
# Verification sheet
# --------------------------------------------------------------------------- #
def write_verification_sheet(path, tree, questions, gold_records, gold_ids, preview_chars=300):
    by_id = {r["id"]: r for r in gold_records}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "question", "reference_answer", "gold_headings",
                    "gold_section_previews", "agent_notes", "verified", "verifier_notes"])
        for q in questions:
            ids = gold_ids[q["id"]]
            heads = " || ".join(section_label(tree, n) for n in ids)
            prev = " || ".join(
                f"[{_norm(tree.nodes[n].heading_title)}] "
                + _norm(tree.nodes[n].body_content)[:preview_chars] for n in ids)
            w.writerow([q["id"], q.get("category", ""), q["question"], q.get("reference_answer", ""),
                        heads, prev, by_id[q["id"]].get("notes", ""), "", ""])


# --------------------------------------------------------------------------- #
# Environment record
# --------------------------------------------------------------------------- #
def environment(device, embed_model, seed):
    def ver(mod):
        try:
            from importlib.metadata import version
            return version(mod)
        except Exception:
            return None
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        try:
            return subprocess.run(["git", "-C", here, *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "embed_model": f"sentence-transformers/{embed_model}",
        "seed": seed,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "libraries": {m: ver(m) for m in ("sentence-transformers", "faiss-cpu", "torch",
                                          "transformers", "numpy", "scipy", "pandas",
                                          "langchain-text-splitters")},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", nargs="+", required=True, help="corpus markdown files (tree source)")
    ap.add_argument("--questions", required=True, help="JSONL with id, question, category")
    ap.add_argument("--gold", default=None,
                    help="gold label JSONL (see docs/retrieval-metrics.md); if omitted, the "
                         "`gold` field inline in --questions is used (public-corpus format)")
    ap.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    ap.add_argument("--ndcg-k", type=int, default=5)
    ap.add_argument("--mrr-depth", type=int, default=10)
    ap.add_argument("--proxy-mode", default="title_firstline", choices=sorted(PROXY_FUNCS))
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--flat-context-chunks", type=int, default=4,
                    help="chunks the flat pipeline passes to the generator (coverage diagnostic)")
    ap.add_argument("--context-cap", type=int, default=1200,
                    help="char cap the benchmarked generators applied to context (coverage diagnostic)")
    ap.add_argument("--legacy-chunks", help="legacy pypdf chunks.pkl (optional)")
    ap.add_argument("--legacy-index", help="legacy faiss.index matching --legacy-chunks")
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, help="directory for CSV/JSON outputs")
    ap.add_argument("--verification-sheet", help="also write a human gold-verification CSV here")
    ap.add_argument("--check-gold-only", action="store_true",
                    help="resolve gold labels (and write the sheet) without embedding anything")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    questions = load_jsonl(args.questions)
    tree = build_tree(args.md)
    gold_records, gold, gold_parts = load_gold(tree, questions, args.gold)
    print(f"gold resolved: {sum(len(v) for v in gold.values())} sections for {len(gold)} questions")
    if args.verification_sheet:
        write_verification_sheet(args.verification_sheet, tree, questions, gold_records, gold)
        print(f"wrote {args.verification_sheet}")
    if args.check_gold_only:
        return

    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    r = SectionRetriever(args.md, embed_model=args.embed_model, device=args.device, tree=tree)
    qv = r.encode_queries([q["question"] for q in questions])
    depth = max(max(args.k), args.ndcg_k, args.mrr_depth)

    systems = {
        "proxy_pointer": (lambda i, q: r.rank_sections_proxy(q, depth, args.proxy_mode, qvec=qv[i]),
                          r.context_proxy),
        "flat_md": (lambda i, q: r.rank_sections_flat(q, depth, args.chunk_size, args.overlap,
                                                      qvec=qv[i]),
                    r.context_identity),
    }
    legacy_cov = None
    if args.legacy_chunks and args.legacy_index:
        legacy_cov = r.load_legacy(args.legacy_chunks, args.legacy_index)
        print(f"legacy pypdf mapping coverage: {legacy_cov}")
        systems["legacy_pypdf"] = (lambda i, q: r.rank_sections_legacy(q, depth, qvec=qv[i]),
                                   r.context_identity)

    try:
        from hybrid_intent_router import classify_query
    except Exception:
        classify_query = None

    rules = {"strict": lambda ctx: r.context_identity,
             "context": lambda ctx: ctx,
             "context_parent": lambda ctx: r.context_proxy}
    metric_names = ([f"R@{k}" for k in args.k] + ["MRR", f"nDCG@{args.ndcg_k}"]
                    + [f"AllParts@{k}" for k in args.k])

    def coverage(sname, i, q, cap):
        g = gold[q["id"]]
        if sname == "proxy_pointer":
            return r.context_coverage_proxy(q["question"], g, cap, args.proxy_mode, qvec=qv[i])
        if sname == "flat_md":
            return r.context_coverage_flat(q["question"], g, args.flat_context_chunks, cap,
                                           args.chunk_size, args.overlap, qvec=qv[i])
        return ""

    rows = []
    for i, q in enumerate(questions):
        route = classify_query(q["question"])[0] if classify_query else ""
        for sname, (rank_fn, ctx) in systems.items():
            cov_full = coverage(sname, i, q, None)
            cov_cap = coverage(sname, i, q, args.context_cap)
            ranked = rank_fn(i, q["question"])
            for rule, ctx_for in rules.items():
                m = score_ranking(tree, ranked, gold[q["id"]], ctx_for(ctx), args.k,
                                  args.ndcg_k, args.mrr_depth, parts=gold_parts[q["id"]])
                rows.append({"id": q["id"], "category": q.get("category", ""),
                             "router_route": route, "system": sname, "rule": rule, **m,
                             "n_gold": len(gold[q["id"]]),
                             "ctx_cov": cov_full, "ctx_cov_capped": cov_cap,
                             "top5": " | ".join(section_label(tree, n) for n in ranked[:5])})

    os.makedirs(args.out_dir, exist_ok=True)
    per_q = os.path.join(args.out_dir, "retrieval_per_question.csv")
    with open(per_q, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregates
    def select(system, rule, cat=None):
        return [x for x in rows if x["system"] == system and x["rule"] == rule
                and (cat is None or x["category"] == cat)]

    cats = [c for c in CATEGORIES if any(q.get("category") == c for q in questions)]
    cats += sorted({q.get("category") for q in questions} - set(cats) - {None})
    agg = []
    for sname in systems:
        for rule in rules:
            for cat in [None] + cats:
                sel = select(sname, rule, cat)
                covs = [x["ctx_cov"] for x in sel if x["ctx_cov"] != ""]
                covc = [x["ctx_cov_capped"] for x in sel if x["ctx_cov_capped"] != ""]
                agg.append({"system": sname, "rule": rule, "category": cat or "overall",
                            "n": len(sel), **{m: round(_mean([x[m] for x in sel]), 4)
                                              for m in metric_names},
                            "ctx_cov": round(_mean(covs), 4) if covs else "",
                            "ctx_cov_capped": round(_mean(covc), 4) if covc else ""})
    agg_csv = os.path.join(args.out_dir, "retrieval_summary.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    # paired bootstrap: proxy - each baseline
    boots = {}
    kmax = max(args.k)
    for base in [s for s in systems if s != "proxy_pointer"]:
        for rule in rules:
            for cat in [None] + cats:
                a = sorted(select("proxy_pointer", rule, cat), key=lambda x: x["id"])
                b = sorted(select(base, rule, cat), key=lambda x: x["id"])
                for m in ("MRR", f"R@{kmax}", f"AllParts@{kmax}"):
                    key = f"proxy_pointer-minus-{base}|{rule}|{cat or 'overall'}|{m}"
                    boots[key] = paired_bootstrap([x[m] for x in a], [x[m] for x in b],
                                                  args.bootstrap, args.seed)

    summary = {
        "description": "Section-level retrieval metrics; see config.gold_label_source for label provenance.",
        "config": {"md": [os.path.basename(p) for p in args.md], "questions": os.path.basename(args.questions),
                   "gold": os.path.basename(args.gold) if args.gold else "inline", "k": args.k, "ndcg_k": args.ndcg_k,
                   "mrr_depth": args.mrr_depth, "proxy_mode": args.proxy_mode,
                   "chunk_size": args.chunk_size, "overlap": args.overlap,
                   "flat_context_chunks": args.flat_context_chunks, "context_cap": args.context_cap,
                   "n_flat_chunks": r.flat_index(args.chunk_size, args.overlap)[0].ntotal,
                   "n_proxy_nodes": r.proxy_index(args.proxy_mode)[0].ntotal,
                   "gold_label_source": sorted({g.get("label_source", "") for g in gold_records}),
                   "gold_verified": all(g.get("verified") for g in gold_records)},
        "legacy_pypdf_mapping": legacy_cov,
        "aggregates": agg,
        "bootstrap": boots,
        "environment": environment(args.device, args.embed_model, args.seed),
    }
    summ = os.path.join(args.out_dir, "retrieval_summary.json")
    with open(summ, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # console table
    print(f"\n{'system':<14} {'rule':<15} {'category':<11} {'n':>3} "
          + " ".join(f"{m:>8}" for m in metric_names))
    for a in agg:
        print(f"{a['system']:<14} {a['rule']:<15} {a['category']:<11} {a['n']:>3} "
              + " ".join(f"{a[m]:>8.3f}" for m in metric_names))
    print(f"\ngold-section coverage of the generator context (proxy: top-1 node+children; "
          f"flat: top-{args.flat_context_chunks} chunks), uncapped / capped at {args.context_cap} chars:")
    for a in agg:
        if a["rule"] == "strict" and a["ctx_cov"] != "":
            print(f"  {a['system']:<14} {a['category']:<11} {a['ctx_cov']:.3f} / {a['ctx_cov_capped']:.3f}")
    print("\npaired bootstrap 95% CI (proxy - baseline), overall:")
    for k_, v in boots.items():
        if "|overall|" in k_:
            print(f"  {k_:<60} {v['mean_delta']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
                  f"{'  crosses 0' if v['crosses_zero'] else ''}")
    print(f"\nwrote {per_q}\nwrote {agg_csv}\nwrote {summ}")


if __name__ == "__main__":
    main()
