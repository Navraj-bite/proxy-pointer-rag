#!/usr/bin/env python3
"""Task 2 + 3: Score naive RAG vs fine-tuned Gemma against reference answers.

For each of the 20 evaluation questions, embed the reference_answer, the RAG
generated_answer, and the fine-tuned generated_answer with all-MiniLM-L6-v2,
then compute cosine similarity of reference-vs-RAG and reference-vs-finetuned.

Saves comparison.csv with columns:
    id, question, reference_answer, rag_answer, rag_score,
    finetuned_answer, finetuned_score

Then prints the averages, the winner, and the first 3 rows.
"""
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
PROMPTS_FILE = "evaluation_prompts.jsonl"
RAG_FILE = "rag_baseline_outputs.csv"
FINETUNED_FILE = "finetuned_outputs.csv"
OUTPUT_FILE = "comparison.csv"


def load_prompts(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine(a, b):
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    prompts = load_prompts(PROMPTS_FILE)
    rag_df = pd.read_csv(RAG_FILE)
    ft_df = pd.read_csv(FINETUNED_FILE).set_index("id")

    # RAG outputs have no id column; they align by row order with the prompts.
    rag_by_question = {
        str(q).strip(): a
        for q, a in zip(rag_df["question"], rag_df["generated_answer"])
    }

    embedder = SentenceTransformer(EMBED_MODEL)

    rows = []
    for p in prompts:
        qid = p["id"]
        question = p["instruction"]
        reference = p["reference_answer"]
        rag_answer = str(rag_by_question[question.strip()])
        finetuned_answer = str(ft_df.loc[qid, "generated_answer"])

        ref_vec, rag_vec, ft_vec = embedder.encode(
            [reference, rag_answer, finetuned_answer]
        )
        rag_score = cosine(ref_vec, rag_vec)
        finetuned_score = cosine(ref_vec, ft_vec)

        rows.append(
            {
                "id": qid,
                "question": question,
                "reference_answer": reference,
                "rag_answer": rag_answer,
                "rag_score": rag_score,
                "finetuned_answer": finetuned_answer,
                "finetuned_score": finetuned_score,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "id", "question", "reference_answer", "rag_answer", "rag_score",
            "finetuned_answer", "finetuned_score",
        ],
    )
    df.to_csv(OUTPUT_FILE, index=False)

    # --- Task 3: final result ---
    avg_rag = df["rag_score"].mean()
    avg_ft = df["finetuned_score"].mean()

    print("\n" + "=" * 60)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print("=" * 60)
    print(f"Average RAG score        : {avg_rag:.4f}")
    print(f"Average fine-tuned score : {avg_ft:.4f}")

    diff = abs(avg_ft - avg_rag)
    if avg_ft > avg_rag:
        winner = "Fine-tuned Gemma"
    elif avg_rag > avg_ft:
        winner = "Naive RAG"
    else:
        winner = "Tie"
    print(f"\nWinner: {winner} (by {diff:.4f} cosine similarity)")

    print("\nFirst 3 rows of comparison.csv:")
    with pd.option_context(
        "display.max_colwidth", 60, "display.width", 200
    ):
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
