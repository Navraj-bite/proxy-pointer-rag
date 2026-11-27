import os
import pickle
import argparse

import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from corpus_config import assistant_intro, ensure_parent_dir, get_config

# Paths come from $PPRAG_CORPUS (default "private" -> the original file names
# and the original pypdf-over-PDFs source).
_CFG = get_config()
PDF_PATHS = _CFG["pdf_paths"]
MARKDOWN_PATHS = _CFG["input_files"]      # same markdown the tree is built from
SOURCE = _CFG["baseline_source"]          # "pdf" | "markdown"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOP_K = 4
INDEX_FILE = _CFG["flat_index_file"]
CHUNKS_FILE = _CFG["flat_chunks_file"]


def load_pdfs(paths):
    docs = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can't find {path}")
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        print(f"  loaded {path}: {len(text)} chars, {len(reader.pages)} pages")
        docs.append(text)
    return docs


def load_markdown(paths):
    """Read the heading-structured markdown files as plain text (no parsing)."""
    docs = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can't find {path}")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        print(f"  loaded {path}: {len(text)} chars")
        docs.append(text)
    return docs


def load_documents(source=None):
    source = source or SOURCE
    if source == "pdf":
        return load_pdfs(PDF_PATHS)
    if source == "markdown":
        return load_markdown(MARKDOWN_PATHS)
    raise ValueError(f"unknown baseline source {source!r}")


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for text in docs:
        chunks.extend(splitter.split_text(text))
    print(f"  produced {len(chunks)} chunks")
    return chunks


def build_index(chunks):
    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(
        chunks, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    ensure_parent_dir(INDEX_FILE)
    ensure_parent_dir(CHUNKS_FILE)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  saved index ({index.ntotal} vectors, dim {vectors.shape[1]})")
    return index, embedder


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, chunks, embedder


def retrieve(question, index, chunks, embedder, k=TOP_K):
    qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, k)
    return [chunks[i] for i in ids[0]]


def generate(question, context_chunks):
    import torch
    from transformers import pipeline

    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        assistant_intro() + " "
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    pipe = pipeline(
        "text-generation", model=GEN_MODEL,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"][-1]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the index even if it already exists")
    parser.add_argument("--source", choices=["pdf", "markdown"], default=None,
                        help=f"override the corpus default (currently {SOURCE!r})")
    args = parser.parse_args()

    if args.rebuild or not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
        print(f"Building index from {args.source or SOURCE} -> {INDEX_FILE} ...")
        docs = load_documents(args.source)
        chunks = chunk_documents(docs)
        build_index(chunks)

    if args.build_only:
        print("Index built. Done.")
        return

    if not args.question:
        print("Pass a question in quotes.")
        return

    index, chunks, embedder = load_index()
    retrieved = retrieve(args.question, index, chunks, embedder)

    print("\n" + "=" * 60)
    print("QUESTION:", args.question)
    print("=" * 60 + "\nRETRIEVED CHUNKS:")
    for n, c in enumerate(retrieved, 1):
        print(f"\n[chunk {n}]\n{c}")
    print("\n" + "=" * 60 + "\nGENERATED ANSWER:\n")
    print(generate(args.question, retrieved))
    print("=" * 60)


if __name__ == "__main__":
    main()