import os
import pickle
import csv

import faiss
from sentence_transformers import SentenceTransformer

TOP_K = 4
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_CSV = "rag_baseline_outputs.csv"

QUESTIONS = [
    "The machine suddenly stopped and the red mushroom button is glowing. What should I do to get the machine running again?",
    "The machine is not running in auto mode. It shows something about jog mode. How do I bring it back to automatic?",
    "I switched the key on the panel earlier and now the machine won't start. What should I do when it is in maintenance mode by mistake?",
    "Air pressure alarm is coming and the machine is not starting. What should I check first?",
    "The screen says cleaning pipe still connected. How do I fix this?",
    "The machine stopped saying draining tank not in correct position. What should I adjust to continue production?",
    "A speed controller fault came and the machine stopped. What should I check for this issue?",
    "Motor overload fault popped up. What action should I take before restarting the machine?",
    "The machine says carousel pressure too low and it stopped. What should I inspect?",
    "Upstream bottles are too few and the gate is not opening. What should I do when the machine waits for bottle accumulation?",
    "Containers are stuck on the accumulation table and machine stopped. What should I clear to resume?",
    "The machine slowed down automatically and shows downstream evacuation problem. What should I check?",
    "Conveyor stopped alarm came and machine won't run. What should I do?",
    "The lid conveyor stopped because too many lids are collected upstream. What do I need to do in this case?",
    "The machine shows lid shortage alarm. What should I check in the lid supply system?",
    "Reject system not ready alarm is coming. What should I check on the diverter or controller?",
    "Product level too low alarm came and machine stopped. What should I check before restarting?",
    "Scale jam alarm is showing. What should I do if a weighing scale is blocked?",
    "Lid not detected after lid applicator and the machine stopped. How do I resume the machine?",
    "Front door open alarm came and machine won't start. What should I check before restarting?",
]


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


def main():
    print("Loading index...")
    index, chunks, embedder = load_index()

    rows = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/20] {question[:70]}...")
        retrieved = retrieve(question, index, chunks, embedder)
        context = " | ".join(c[:200] for c in retrieved)
        answer = generate(question, retrieved)
        rows.append({
            "question_id": i,
            "question": question,
            "retrieved_context": context,
            "rag_answer": answer,
        })
        print(f"  Answer: {answer[:120]}...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question_id", "question", "retrieved_context", "rag_answer"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
