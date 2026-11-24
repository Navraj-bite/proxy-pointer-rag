#!/usr/bin/env python3
"""Generate naive-RAG answers for the newly added eval questions (ids > 20)
and append them to rag_baseline_outputs.csv, matching its existing columns
(question, retrieved_chunks, generated_answer).

Pipeline mirrors naive_rag_baseline.py: FAISS top-4 retrieval over the saved
index, TinyLlama generation with the same "answer from context only" prompt.
The generation model is loaded ONCE, outside the loop.
"""
import json
import pickle

import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPTS_FILE = "evaluation_prompts.jsonl"
RAG_CSV = "rag_baseline_outputs.csv"
TOP_K = 4


def main():
    prompts = [json.loads(l) for l in open(PROMPTS_FILE) if l.strip()]
    existing = pd.read_csv(RAG_CSV)
    have = {str(q).strip() for q in existing["question"]}
    todo = [p for p in prompts if p["instruction"].strip() not in have]
    print(f"RAG CSV has {len(existing)} rows; {len(todo)} eval questions missing")
    if not todo:
        print("Nothing to do.")
        return

    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading TinyLlama generation pipeline (once)...")
    pipe = pipeline(
        "text-generation", model=GEN_MODEL,
        torch_dtype=torch.bfloat16, device_map="auto",
    )

    new_rows = []
    for p in todo:
        question = p["instruction"]
        qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
        _, ids = index.search(qv, TOP_K)
        retrieved = [chunks[i] for i in ids[0]]
        context = "\n\n---\n\n".join(retrieved)

        prompt = (
            "You are a support assistant for Serac filling machines. "
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        out = pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=256, do_sample=False,
        )
        answer = out[0]["generated_text"][-1]["content"].strip()
        new_rows.append({
            "question": question,
            "retrieved_chunks": context,
            "generated_answer": answer,
        })
        print(f"[id {p['id']:>2}] done ({len(answer)} chars)")

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined[["question", "retrieved_chunks", "generated_answer"]]
    combined.to_csv(RAG_CSV, index=False)
    print(f"\nAppended {len(new_rows)} rows -> {RAG_CSV} now has {len(combined)} rows")


if __name__ == "__main__":
    main()
