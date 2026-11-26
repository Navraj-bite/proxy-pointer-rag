#!/usr/bin/env python3
"""Gemma-vs-Gemma three-arm comparison for the Week 1 fine-tune claim.

Arms (all on the same evaluation prompts, same tokenizer/chat template, same
greedy decoding, same dtype and device -- the model object is loaded once):

  base         google/gemma-3-1b-it, question only
  adapter      base + serac-gemma-adapter (LoRA, unmerged), question only
  rag          base, Week 1 flat retriever (faiss.index/chunks.pkl, top-k 4)
               with the Week 1 grounded prompt
  rag_adapter  (optional) base + adapter with the RAG prompt
  adapter_w1repro  (diagnostic) adapter arm tokenized the way the Week 1
               run_finetuned.py did it: chat-template string re-tokenized with
               add_special_tokens=True, i.e. a doubled <bos>. Used only to check
               whether the stored finetuned_outputs.csv came from Gemma+adapter.

The adapter's training prompt (finetune.py) is the tokenizer chat template with
a single user turn = instruction, no system prompt. base/adapter use exactly
that. rag differs from base only by the user-turn text (context + question).

Outputs (in the cwd, which should be the private run dir):
  gemma_arm_<arm>.jsonl   one record per answer, appended + fsynced as produced
                          (resumable: ids already present are skipped)
  gemma_run_env.json      environment + per-arm wall time and memory high-water

Run:
  cd <private run dir> && HF_HUB_OFFLINE=1 python3 /path/to/run_gemma_arms.py \
      --arms base,adapter,rag --device mps --dtype fp32
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_PATH = "serac-gemma-adapter"
PROMPTS_FILE = "evaluation_prompts.jsonl"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
ENV_FILE = "gemma_run_env.json"
ALL_ARMS = ["base", "adapter", "rag", "rag_adapter", "adapter_w1repro"]
ADAPTER_ARMS = {"adapter", "rag_adapter", "adapter_w1repro"}
RAG_ARMS = {"rag", "rag_adapter"}

# Week 1 grounded prompt (naive_rag_baseline.py / rag_new_questions.py), verbatim.
RAG_TEMPLATE = (
    "You are a support assistant for Serac filling machines. "
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)
CHUNK_SEP = "\n\n---\n\n"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="base,adapter,rag",
                    help=f"comma list from {ALL_ARMS}")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (0 = all)")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                    help="fp16 is deliberately not offered: Gemma 3 overflows in fp16")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="256 matches the Week 1 scripts")
    ap.add_argument("--seed", type=int, default=0, help="decoding is greedy; seed set anyway")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--context-chars", type=int, default=0,
                    help="truncate joined RAG context to N chars (0 = no truncation)")
    ap.add_argument("--suffix", default="", help="output filename suffix, e.g. _sanity, _v2")
    ap.add_argument("--prompts", default=PROMPTS_FILE, help="eval set (JSONL: id, instruction, ...)")
    ap.add_argument("--reuse-from", default=None, metavar="SUFFIX",
                    help="copy an answer from gemma_arm_<arm><SUFFIX>.jsonl instead of generating "
                         "it, but only when the token ids rendered for that id from "
                         "--reuse-prompts are byte-identical to this run's (e.g. --reuse-from '')")
    ap.add_argument("--reuse-prompts", default=PROMPTS_FILE,
                    help="eval set the --reuse-from answers were generated from")
    return ap.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_revision():
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == MODEL_ID:
            revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
            return revs[0].commit_hash if revs else None
    return None


def git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", here, "diff", "--quiet", "HEAD", "--",
                                 os.path.basename(__file__)]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def rss_peak_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # bytes on macOS, KiB on Linux


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except (ValueError, KeyError):
                        pass  # tolerate a torn final line from a crash
    return done


def env_path(suffix):
    return ENV_FILE.replace(".json", f"{suffix}.json")


def load_env(suffix):
    if os.path.exists(env_path(suffix)):
        with open(env_path(suffix)) as f:
            return json.load(f)
    return {}


def save_env(env, suffix):
    tmp = env_path(suffix) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(env, f, indent=2)
    os.replace(tmp, env_path(suffix))


def main():
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        sys.exit(f"unknown arms: {bad}")
    # Base arms first, then wrap with the adapter once: same model object throughout.
    arms.sort(key=lambda a: a in ADAPTER_ARMS)

    import numpy as np
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        sys.exit("MPS requested but not available")

    with open(args.prompts) as f:
        prompts = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    retriever = None
    if any(a in RAG_ARMS for a in arms):
        import faiss
        from sentence_transformers import SentenceTransformer
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        retriever = (index, chunks, embedder)

    def retrieve(question):
        index, chunks, embedder = retriever
        qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(qv, args.top_k)
        return [int(i) for i in ids[0]], [float(s) for s in scores[0]], [chunks[i] for i in ids[0]]

    print(f"Loading {MODEL_ID} ({args.dtype}, {device})...", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map={"": device})
    model.eval()
    print(f"  loaded in {time.time() - t_load:.1f}s", flush=True)
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]

    env = load_env(args.suffix)
    env.update({
        "model_id": MODEL_ID,
        "model_revision": model_revision(),
        "adapter_path": ADAPTER_PATH,
        "adapter_sha256": sha256(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")),
        "device": device,
        "dtype": args.dtype,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                     "eos_token_id": eos_ids, "repetition_penalty": None},
        "seed": args.seed,
        "rag": {"top_k": args.top_k, "context_chars": args.context_chars or None,
                "index": INDEX_FILE, "embed_model": EMBED_MODEL, "chunk_sep": CHUNK_SEP},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
        "n_prompts": len(prompts),
        "prompts_file": args.prompts,
        "prompts_sha256": sha256(args.prompts),
    })
    env.setdefault("arms", {})
    save_env(env, args.suffix)

    def render(arm, q):
        """Exact model input for (arm, question): token ids + retrieval metadata."""
        meta = {}
        if arm in RAG_ARMS:
            ids, scores, texts = retrieve(q)
            context = CHUNK_SEP.join(texts)
            if args.context_chars:
                context = context[: args.context_chars]
            user = RAG_TEMPLATE.format(context=context, question=q)
            meta = {"chunk_ids": ids, "chunk_scores": scores, "context_chars": len(context)}
        else:
            user = q
        messages = [{"role": "user", "content": user}]
        if arm == "adapter_w1repro":
            s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(s, return_tensors="pt")["input_ids"]
        else:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt")
        return input_ids, meta

    reuse_q = {}
    if args.reuse_from is not None:
        with open(args.reuse_prompts) as f:
            reuse_q = {r["id"]: r["instruction"] for r in map(json.loads, filter(str.strip, f))}

    adapter_loaded = False
    mps_hw = 0
    for arm in arms:
        if arm in ADAPTER_ARMS and not adapter_loaded:
            from peft import PeftModel
            print(f"Applying LoRA adapter {ADAPTER_PATH} (unmerged)...", flush=True)
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            model.eval()
            adapter_loaded = True
        if arm not in ADAPTER_ARMS and adapter_loaded:
            sys.exit("internal: base arm after adapter was applied")

        out_path = f"gemma_arm_{arm}{args.suffix}.jsonl"
        done = load_done(out_path)
        todo = [p for p in prompts if p["id"] not in done]
        print(f"\n=== arm {arm}: {len(done)} done, {len(todo)} to go -> {out_path}", flush=True)
        if not todo:
            continue

        reuse_recs = {}
        if args.reuse_from is not None and os.path.exists(f"gemma_arm_{arm}{args.reuse_from}.jsonl"):
            with open(f"gemma_arm_{arm}{args.reuse_from}.jsonl") as f:
                reuse_recs = {r["id"]: r for r in map(json.loads, filter(str.strip, f))}

        t_arm = time.time()
        gen_tokens = 0
        n_reused = 0
        with open(out_path, "a") as fout:
            for p in todo:
                q = p["instruction"]
                rec = {"id": p["id"], "arm": arm}
                input_ids, meta = render(arm, q)
                rec.update(meta)
                prompt_sha = hashlib.sha256(
                    json.dumps(input_ids[0].tolist()).encode()).hexdigest()
                old = reuse_recs.get(p["id"])
                if old is not None and p["id"] in reuse_q:
                    old_ids, _ = render(arm, reuse_q[p["id"]])
                    if torch.equal(old_ids, input_ids):
                        new_rec = dict(old, reused_from=f"gemma_arm_{arm}{args.reuse_from}.jsonl",
                                       prompt_sha256=prompt_sha)
                        fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        os.fsync(fout.fileno())
                        n_reused += 1
                        print(f"[{arm} id {p['id']:>3}] reused (identical prompt)", flush=True)
                        continue
                rec["prompt_sha256"] = prompt_sha
                input_ids = input_ids.to(device)
                attn = torch.ones_like(input_ids)

                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids, attention_mask=attn,
                        max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                        temperature=None, top_p=None, top_k=None,
                        eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
                    )
                if device == "mps":
                    torch.mps.synchronize()
                    mps_hw = max(mps_hw, torch.mps.driver_allocated_memory())
                dt = time.time() - t0
                new = out[0][input_ids.shape[1]:]
                answer = tokenizer.decode(new, skip_special_tokens=True).strip()
                n_new = int(new.shape[0])
                gen_tokens += n_new
                rec.update({
                    "answer": answer,
                    "prompt_tokens": int(input_ids.shape[1]),
                    "new_tokens": n_new,
                    "hit_max_new_tokens": n_new >= args.max_new_tokens,
                    "gen_seconds": round(dt, 3),
                })
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                print(f"[{arm} id {p['id']:>2}] {n_new:>3} tok {dt:5.1f}s "
                      f"({n_new / max(dt, 1e-6):.1f} tok/s) prompt {input_ids.shape[1]} tok",
                      flush=True)

        wall = time.time() - t_arm
        a = env["arms"].get(arm, {})
        a["wall_seconds"] = round(a.get("wall_seconds", 0) + wall, 1)
        a["generated_tokens"] = a.get("generated_tokens", 0) + gen_tokens
        a["n_answers"] = len(load_done(out_path))
        a["reused_answers"] = a.get("reused_answers", 0) + n_reused
        a["runs"] = a.get("runs", 0) + 1
        a["last_finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        env["arms"][arm] = a
        env["peak_process_rss_bytes"] = max(env.get("peak_process_rss_bytes", 0), rss_peak_bytes())
        if device == "mps":
            env["peak_mps_driver_allocated_bytes"] = max(
                env.get("peak_mps_driver_allocated_bytes", 0), mps_hw)
        env["memory_note"] = ("Apple unified memory: peak_process_rss_bytes is ru_maxrss of the "
                              "process; peak_mps_driver_allocated_bytes is the high-water of "
                              "torch.mps.driver_allocated_memory() sampled after each generate. "
                              "Neither is VRAM.")
        env["date"] = datetime.date.today().isoformat()
        save_env(env, args.suffix)
        print(f"=== arm {arm} done: {wall:.1f}s, {gen_tokens} tokens", flush=True)

    print("\nAll requested arms complete.", flush=True)


if __name__ == "__main__":
    main()
