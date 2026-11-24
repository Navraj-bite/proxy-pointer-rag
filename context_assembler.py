"""
Day 8 — Context Assembler (runtime retriever).

Given a natural-language question, embeds it with all-MiniLM-L6-v2,
searches the proxy-pointer FAISS index for the closest matching section
heading, then assembles full context from the matched DocumentNode and all
its direct children.

Usage:
    python3 context_assembler.py "Your question here"

Or import and call query(question) -> str from other modules.
"""

import pickle
import sys

import faiss
from sentence_transformers import SentenceTransformer

PP_INDEX_FILE = "pp_faiss.index"
PP_META_FILE  = "pp_metadata.pkl"
PP_TREE_FILE  = "pp_tree.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Module-level singletons so repeated calls don't reload everything.
_index    = None
_metadata = None
_tree     = None
_embedder = None


def _load():
    global _index, _metadata, _tree, _embedder
    if _index is not None:
        return
    _index = faiss.read_index(PP_INDEX_FILE)
    with open(PP_META_FILE, "rb") as f:
        _metadata = pickle.load(f)
    with open(PP_TREE_FILE, "rb") as f:
        _tree = pickle.load(f)
    _embedder = SentenceTransformer(EMBED_MODEL)


def query(question: str) -> str:
    """
    Retrieve full context for *question* via proxy-pointer lookup.

    Steps:
      1. Embed the question with all-MiniLM-L6-v2.
      2. Search pp_faiss.index for the single closest proxy heading.
      3. Read pointer_node_id from pp_metadata.pkl.
      4. Fetch that DocumentNode from pp_tree.pkl.
      5. Fetch all direct children of that node.
      6. Concatenate: matched node heading + body + each child heading + body.
      7. Return as one clean string.
    """
    _load()

    qv = _embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = _index.search(qv, 1)

    idx   = int(ids[0][0])
    score = float(scores[0][0])
    meta  = _metadata[idx]
    node  = _tree.nodes[meta["pointer_node_id"]]

    parts = [f"# {node.heading_title}"]
    if node.body_content.strip():
        parts.append(node.body_content.strip())

    for child_id in node.child_node_ids:
        child = _tree.nodes[child_id]
        parts.append(f"## {child.heading_title}")
        if child.body_content.strip():
            parts.append(child.body_content.strip())

    return "\n\n".join(parts)


def query_with_meta(question: str):
    """Same as query() but also returns retrieval metadata for inspection."""
    _load()

    qv = _embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = _index.search(qv, 1)

    idx   = int(ids[0][0])
    score = float(scores[0][0])
    meta  = _metadata[idx]
    node  = _tree.nodes[meta["pointer_node_id"]]

    parts = [f"# {node.heading_title}"]
    if node.body_content.strip():
        parts.append(node.body_content.strip())

    for child_id in node.child_node_ids:
        child = _tree.nodes[child_id]
        parts.append(f"## {child.heading_title}")
        if child.body_content.strip():
            parts.append(child.body_content.strip())

    context = "\n\n".join(parts)
    return context, {
        "node_id":       node.node_id,
        "heading_title": node.heading_title,
        "doc_id":        meta["doc_id"],
        "start_line":    meta["start_line"],
        "end_line":      meta["end_line"],
        "score":         score,
        "n_children":    len(node.child_node_ids),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 context_assembler.py \"Your question here\"")
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")

    context, meta = query_with_meta(question)

    print(f"Matched node  : {meta['node_id']}  (score={meta['score']:.4f})")
    print(f"Heading       : {meta['heading_title']}")
    print(f"Source file   : {meta['doc_id']}  lines {meta['start_line']}–{meta['end_line']}")
    print(f"Children      : {meta['n_children']}")
    print("\n" + "=" * 70)
    print("ASSEMBLED CONTEXT")
    print("=" * 70)
    print(context)
    print("=" * 70)


if __name__ == "__main__":
    main()
