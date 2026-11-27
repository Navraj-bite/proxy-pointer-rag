#!/usr/bin/env python3
"""Rebuild questions/public_v1.jsonl from the seed set, the four writer drafts and the
independent verifier verdicts. Deterministic; documents exactly how the evaluation set was made.

    python3 scripts/build_public_v1.py

Provenance: questions were drafted by four LLM agents (one per corpus slice, questions/drafts/),
then each draft was checked by a *different* LLM agent (questions/verification/verify_*.jsonl):
answer support, missing alternative gold, gold precision, category, ambiguity, heading copying.
Verdict fixes are applied here. A human spot-check of a stratified sample is still pending
(questions/human_verification_sample.csv).
"""
import collections, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "questions")

# For multihop rows, which required part(s) each verifier gold_add is an alternative for
# (0-based, in gold_add order). Assigned by hand from the verifier notes.
ASSIGN = {"A-060": [[0]], "C-047": [[0]], "C-048": [[0]], "C-052": [[0], [0], [1, 2]],
          "C-054": [[0], [1]], "C-058": [[0]], "C-059": [[0]], "C-060": [[1]]}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    rows = []
    for r in load(os.path.join(Q, "public_seed.jsonl")):
        r["id"] = f"S-{int(r['id']):03d}"; r["source"] = "seed"; rows.append(r)
    for f in sorted(glob.glob(os.path.join(Q, "drafts", "group_*.jsonl"))):
        for r in load(f):
            r["source"] = os.path.basename(f)[6]; rows.append(r)
    verdicts = {}
    for f in sorted(glob.glob(os.path.join(Q, "verification", "verify_*.jsonl"))):
        for v in load(f):
            verdicts[v["id"]] = v
    assert set(verdicts) == {r["id"] for r in rows}, "every question must have a verdict"

    key = lambda g: (os.path.basename(g["doc"]), json.dumps(g["heading_path"]))
    norm = lambda g: {"doc": g["doc"] if g["doc"].startswith("corpus/") else "corpus/" + g["doc"],
                      "heading_path": g["heading_path"]}
    out, log = [], collections.Counter()
    for r in rows:
        v = verdicts[r["id"]]
        if v["verdict"] == "drop":
            log["dropped"] += 1; continue
        if v.get("question_new"):
            r["question"] = v["question_new"]; log["question rewritten"] += 1
        if v.get("reference_answer_new"):
            r["reference_answer"] = v["reference_answer_new"]; log["answer rewritten"] += 1
        if v.get("category_new") and v["category_new"] != r["category"]:
            log[f"category {r['category']}->{v['category_new']}"] += 1; r["category"] = v["category_new"]
        rem = {key(g) for g in (v.get("gold_remove") or [])}
        adds = [norm(g) for g in (v.get("gold_add") or [])]
        base = [norm(g) for g in r["gold"] if key(g) not in rem]
        if r["category"] == "multihop":
            parts = [[g] for g in base]
            for g, idxs in zip(adds, ASSIGN.get(r["id"], [])):
                for i in idxs:
                    parts[i].append(g)
            assert len(parts) >= 2 and len(adds) == len(ASSIGN.get(r["id"], [])), r["id"]
            r["gold_parts"] = parts
        union = []
        for g in base + adds:
            if key(g) not in {key(x) for x in union}:
                union.append(g)
        r["gold"] = union
        r["verification"] = {"verifier": "agent", "verdict": v["verdict"], "issues": v["issues"],
                             "heading_copy": v.get("heading_copy")}
        r["label_source"] = "agent-drafted, agent-verified (independent pass); human spot-check pending"
        out.append(r)
    with open(os.path.join(Q, "public_v1.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(log))
    print(len(out), "questions:", dict(collections.Counter(r["category"] for r in out)))


if __name__ == "__main__":
    main()
