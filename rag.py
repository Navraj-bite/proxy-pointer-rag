import os
import pickle
import argparse

import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------- CONFIG
PDF_PATHS = [
    "resources/machine_overview_troubleshooting.pdf",
    "resources/operator_manual.pdf",
]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/gemma-3-1b-it"
TOP_K = 4
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"


# STAGE 1 -- LOAD
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


# STAGE 2 -- CHUNK
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for text in docs:
        chunks.extend(splitter.split_text(text))
    print(f"  produced {len(chunks)} chunks")
    return chunks


# STAGE 3 -- EMBED + INDEX
def build_index(chunks):
    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(
        chunks, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
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


# STAGE 4 -- RETRIEVE
def retrieve(question, index, chunks, embedder, k=TOP_K):
    qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, k)
    return [chunks[i] for i in ids[0]]


# STAGE 5 -- GENERATE
def generate(question, context_chunks):
    import torch
    from transformers import pipeline

    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a support assistant for Serac filling machines. "
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


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
        print("Building index...")
        docs = load_pdfs(PDF_PATHS)
        chunks = chunk_documents(docs)
        build_index(chunks)

    if args.build_only:
        print("Index built. Done.")
        return

    if not args.question:
        print('Pass a question in quotes.')
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
