"""
Day 8 — Proxy-Pointer Indexer.

Walks the DocumentNode tree, embeds every content node's heading_title
as a lightweight proxy, stores those embeddings in pp_faiss.index, and
saves the companion metadata (pp_metadata.pkl) and full tree (pp_tree.pkl)
so the runtime retriever can dereference any hit back to full body_content.

Run:  python3 proxy_pointer_indexer.py
"""

import pickle
import random

import faiss
from sentence_transformers import SentenceTransformer

from document_tree_builder import DocumentTree

INPUT_FILES   = ["machine_overview_troubleshooting.md", "operator_manual.md"]
PP_INDEX_FILE = "pp_faiss.index"
PP_META_FILE  = "pp_metadata.pkl"
PP_TREE_FILE  = "pp_tree.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"


def get_doc_id(node, tree):
    """Walk the parent chain until we find the [FILE] container node."""
    cur = node
    while cur is not None:
        if cur.heading_title.startswith("[FILE] "):
            return cur.heading_title[len("[FILE] "):]
        if cur.parent_node_id is None:
            break
        cur = tree.nodes[cur.parent_node_id]
    return "unknown"


def build_proxy_index(tree):
    # Exclude the synthetic ROOT and [FILE] container nodes — they carry no
    # searchable content and would pollute retrieval results.
    nodes = [
        n for n in tree.nodes.values()
        if n.node_id != tree.root.node_id
        and not n.heading_title.startswith("[FILE] ")
    ]
    print(f"  indexable nodes: {len(nodes)}")

    # Augment each proxy with the opening line(s) of body_content so that
    # section-heading-only nodes (especially def-code alarm entries) carry
    # enough semantic signal for the embedder to distinguish them by topic.
    # The proxy remains short (<300 chars) — the full body is fetched at
    # query time via the pointer, not stored in the index.
    def _proxy_text(node):
        title = node.heading_title
        snippet = node.body_content.strip()
        if snippet:
            # Take the first non-empty, non-pipe-table line (max 200 chars)
            for line in snippet.splitlines():
                line = line.strip()
                if line and not line.startswith("|") and not line.startswith("#"):
                    return f"{title}\n{line[:200]}"
            # Fallback: just use title if body is all table rows
            return title
        return title

    proxy_texts = [_proxy_text(n) for n in nodes]
    metadata = [
        {
            "pointer_node_id": n.node_id,
            "heading_title":   n.heading_title,
            "doc_id":          get_doc_id(n, tree),
            "start_line":      n.start_line,
            "end_line":        n.end_line,
        }
        for n in nodes
    ]

    print("  embedding proxy texts with all-MiniLM-L6-v2 ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    vecs = embedder.encode(
        proxy_texts, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, PP_INDEX_FILE)
    with open(PP_META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    with open(PP_TREE_FILE, "wb") as f:
        pickle.dump(tree, f)

    print(f"  saved {PP_INDEX_FILE}  ({index.ntotal} vectors, dim {vecs.shape[1]})")
    print(f"  saved {PP_META_FILE}   ({len(metadata)} entries)")
    print(f"  saved {PP_TREE_FILE}")
    return index, metadata, embedder


def verify_linkage(tree, metadata, n_samples=3):
    """Round-trip n_samples random proxy entries back to their full nodes."""
    print(f"\n=== Linkage verification (n={n_samples}) ===")
    rng = random.Random(7)
    indices = rng.sample(range(len(metadata)), n_samples)

    for rank, idx in enumerate(indices, start=1):
        meta = metadata[idx]
        node = tree.nodes.get(meta["pointer_node_id"])

        print(f"\n  Sample {rank} (metadata index {idx})")
        if node is None:
            print(f"    [FAIL] pointer_node_id {meta['pointer_node_id']} not found in tree")
            continue

        has_body  = len(node.body_content.strip()) > 0
        has_table = "|" in node.body_content
        has_steps = any(
            ln.strip() and ln.strip()[0].isdigit() and ". " in ln
            for ln in node.body_content.splitlines()
        )
        print(f"    node_id       : {node.node_id}")
        print(f"    heading_title : {node.heading_title}")
        print(f"    doc_id        : {meta['doc_id']}")
        print(f"    lines         : {meta['start_line']} – {meta['end_line']}")
        print(f"    body_content  : {len(node.body_content)} chars  "
              f"[table={has_table}  steps={has_steps}  non-empty={has_body}]")
        print(f"    body preview  : {node.body_content[:150]!r}")


def main():
    print("Building document tree ...")
    tree = DocumentTree()
    for path in INPUT_FILES:
        tree.add_file(path)
    print(f"  total nodes: {len(tree.nodes)}")

    print("\nBuilding proxy-pointer FAISS index ...")
    index, metadata, embedder = build_proxy_index(tree)

    verify_linkage(tree, metadata)
    print("\nDone.")


if __name__ == "__main__":
    main()
