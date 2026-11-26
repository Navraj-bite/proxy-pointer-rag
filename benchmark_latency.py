#!/usr/bin/env python3
"""
Stage-level latency, index build/size and memory benchmark.

The published latencies (vanilla 22.2 s, hybrid 26.7 s) are wall-clock times
around a whole answer call: they include one-time model loading on the first
question and TinyLlama generation, so they mostly measure the generator. This
script measures the retriever on its own, stage by stage, and measures
generation separately with identical settings for both systems.

Systems (the production code paths, same on-disk indices):
  proxy_pointer : MiniLM query embedding -> FAISS top-1 over heading proxies
                  -> dereference pointer_node_id to the DocumentNode
                  -> assemble node + direct children (context_assembler.query)
  flat          : MiniLM query embedding -> FAISS top-4 over 500/50 chunks
                  -> look up + join the chunks with "\\n\\n---\\n\\n"
                  (naive_rag_baseline.retrieve + run_day10_benchmark)

Modes (each runs in its own process so peak RSS is per mode):

  retrieval  time embed / search / dereference / assemble per query, over
             every question x --repeats, after --warmup untimed queries.
             Models and indices are loaded once, before timing; the load is
             timed separately. Indices must exist (run `build` first).
  build      time index builds: tree build / proxy texts / embedding /
             FAISS add / write for proxy-pointer; load / chunk / embedding /
             FAISS add / write for flat. Writes the indices to the corpus'
             configured paths, reports on-disk size and vector counts.
  generate   time TinyLlama generation on a small sample with identical
             settings for both systems, context capped at 1,200 chars (as
             published) and uncapped. Reports tokens/s, time to first token
             and the effect of context length.

Usage (public corpus, from the repo root):
    export HF_HUB_OFFLINE=1
    python3 benchmark_latency.py build     --corpus public --repeats 3
    python3 benchmark_latency.py retrieval --corpus public --repeats 3 --warmup 20
    python3 benchmark_latency.py retrieval --corpus public --device mps
    OMP_NUM_THREADS=1 python3 benchmark_latency.py generate --corpus public --gen-n 10

Private corpus: run from a directory holding the private inputs (see
corpus_config.py) and write --out there; the script refuses to write
private-corpus results inside this repository.

On macOS arm64, torch is imported before faiss here on purpose (faiss first
+ multi-threaded torch segfaults, see docs/corpus.md). Set OMP_NUM_THREADS=1
for `generate` anyway, as documented for generation runs. Thread settings are
recorded in every output file because they change the timings.

Method and results: docs/latency-memory.md.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time

_T_IMPORT0 = time.perf_counter()
_LOADAVG_START = list(os.getloadavg())

import numpy as np
import torch  # before faiss, deliberately

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PUBLIC_CORPUS_COMMIT = "d6f12ad1c4f70ad3230afd7d86e971421e02fef4"  # docs/corpus.md
FLAT_TOP_K = 4
PROXY_TOP_K = 1
PUBLISHED_CAP = 1200
SEED = 0

pc = time.perf_counter


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def peak_rss_bytes():
    """Process peak resident set size. ru_maxrss is bytes on macOS, KiB on Linux."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def mps_mem():
    if not torch.backends.mps.is_available():
        return None
    try:
        return {"driver_allocated": int(torch.mps.driver_allocated_memory()),
                "current_allocated": int(torch.mps.current_allocated_memory())}
    except Exception:
        return None


class MpsHighWater:
    """torch.mps exposes current values only; track the max over samples."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.driver = 0
        self.current = 0

    def sample(self):
        if not self.enabled:
            return
        m = mps_mem()
        if m:
            self.driver = max(self.driver, m["driver_allocated"])
            self.current = max(self.current, m["current_allocated"])

    def as_dict(self):
        if not self.enabled:
            return None
        return {"driver_allocated_high_water_bytes": self.driver,
                "current_allocated_high_water_bytes": self.current,
                "note": "max over samples taken after load and after every query; "
                        "unified memory, not VRAM"}


def stats_ms(values_s):
    a = np.asarray(values_s, dtype=float) * 1000.0
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def stats_plain(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def _ver(mod):
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return None


def _git(*args):
    try:
        return subprocess.run(["git", "-C", HERE, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _hf_revision(repo_id):
    """Commit hash of the cached snapshot (refs/main) for a hub model, if cached."""
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    ref = os.path.join(root, "models--" + repo_id.replace("/", "--"), "refs", "main")
    try:
        with open(ref) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpu_brand():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return platform.processor() or None


def _sha256_files(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def environment(cfg, devices, models):
    import faiss
    env = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "cpu_count": os.cpu_count(),
        "devices": devices,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "faiss_omp_max_threads": faiss.omp_get_max_threads(),
        },
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "models": {m: {"revision": _hf_revision(m)} for m in models},
        "seed": SEED,
        "libraries": {m: _ver(m) for m in ("torch", "transformers", "sentence-transformers",
                                          "faiss-cpu", "numpy", "langchain-text-splitters",
                                          "pypdf", "tokenizers")},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "corpus": cfg["name"],
        "loadavg_start": _LOADAVG_START,
        "machine_idle": False,
        "machine_note": "other agents were doing light file reading/grep during the run",
    }
    if cfg["name"] == "public":
        env["corpus_commit"] = PUBLIC_CORPUS_COMMIT
        files = [p for p in cfg["input_files"] if os.path.exists(p)]
        if files:
            env["corpus_markdown_sha256"] = _sha256_files(files)
    else:
        env["corpus_note"] = "private corpus; numbers are not reproducible from a clone"
    return env


def load_questions(paths, limit=None):
    seen, out = set(), []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                key = " ".join(q["question"].split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append({"id": str(q.get("id")), "question": q["question"],
                            "category": q.get("category")})
    if limit:
        out = out[:limit]
    return out


def _sources(paths):
    """Question files used, by basename + sha256 (the full path may be private)."""
    return [{"file": os.path.basename(p), "sha256": _sha256_files([p])} for p in paths]


def resolve_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def check_out_path(cfg, out):
    """Never write private-corpus results inside the repository."""
    if cfg["name"] != "private" or not out:
        return
    o = os.path.realpath(os.path.abspath(out))
    repo = os.path.realpath(HERE)
    if o == repo or o.startswith(repo + os.sep):
        raise SystemExit(f"refusing to write private-corpus results inside the repo: {out}")


def write_out(path, obj):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"\nwrote {path}")


def fmt_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Retrieval: the exact production operations, split into stages
# --------------------------------------------------------------------------- #
def assemble_node_context(tree, node):
    """Byte-identical to context_assembler.query()'s assembly step."""
    parts = [f"# {node.heading_title}"]
    if node.body_content.strip():
        parts.append(node.body_content.strip())
    for child_id in node.child_node_ids:
        child = tree.nodes[child_id]
        parts.append(f"## {child.heading_title}")
        if child.body_content.strip():
            parts.append(child.body_content.strip())
    return "\n\n".join(parts)


def embed_query(embedder, q):
    # same call as context_assembler.query / naive_rag_baseline.retrieve
    return embedder.encode([q], normalize_embeddings=True).astype("float32")


def run_proxy(stack, q):
    t0 = pc()
    qv = embed_query(stack["embedder"], q)
    t1 = pc()
    scores, ids = stack["pp_index"].search(qv, PROXY_TOP_K)
    t2 = pc()
    meta = stack["pp_meta"][int(ids[0][0])]
    node = stack["tree"].nodes[meta["pointer_node_id"]]
    t3 = pc()
    ctx = assemble_node_context(stack["tree"], node)
    t4 = pc()
    return ctx, {"embed": t1 - t0, "search": t2 - t1, "deref": t3 - t2,
                 "assemble": t4 - t3, "retrieval_total": t4 - t0}


def run_flat(stack, q):
    t0 = pc()
    qv = embed_query(stack["embedder"], q)
    t1 = pc()
    scores, ids = stack["flat_index"].search(qv, FLAT_TOP_K)
    t2 = pc()
    chunks = stack["flat_chunks"]
    ctx = "\n\n---\n\n".join([chunks[i] for i in ids[0]])
    t3 = pc()
    return ctx, {"embed": t1 - t0, "search": t2 - t1, "assemble": t3 - t2,
                 "retrieval_total": t3 - t0}


def load_retrieval_stack(cfg, device):
    """Load MiniLM + both indices once. Returns (stack, load timings in s)."""
    import pickle
    from sentence_transformers import SentenceTransformer
    import faiss

    load = {}
    t = pc()
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    load["embedder"] = pc() - t

    t = pc()
    pp_index = faiss.read_index(cfg["pp_index_file"])
    with open(cfg["pp_meta_file"], "rb") as f:
        pp_meta = pickle.load(f)
    with open(cfg["pp_tree_file"], "rb") as f:
        tree = pickle.load(f)
    load["proxy_index_meta_tree"] = pc() - t

    t = pc()
    flat_index = faiss.read_index(cfg["flat_index_file"])
    with open(cfg["flat_chunks_file"], "rb") as f:
        flat_chunks = pickle.load(f)
    load["flat_index_chunks"] = pc() - t

    stack = {"embedder": embedder, "pp_index": pp_index, "pp_meta": pp_meta, "tree": tree,
             "flat_index": flat_index, "flat_chunks": flat_chunks}
    return stack, load


def check_equivalence(cfg, stack, questions):
    """Confirm the staged code returns exactly what the production functions return."""
    os.environ["PPRAG_CORPUS"] = cfg["name"]
    import context_assembler as ca
    import naive_rag_baseline as nrb
    ca._index, ca._metadata, ca._tree, ca._embedder = (
        stack["pp_index"], stack["pp_meta"], stack["tree"], stack["embedder"])
    for q in questions:
        p_ctx, _ = run_proxy(stack, q["question"])
        f_ctx, _ = run_flat(stack, q["question"])
        if p_ctx != ca.query(q["question"]):
            raise SystemExit(f"proxy context differs from context_assembler.query for {q['id']}")
        ref = "\n\n---\n\n".join(nrb.retrieve(q["question"], stack["flat_index"],
                                              stack["flat_chunks"], stack["embedder"], k=FLAT_TOP_K))
        if f_ctx != ref:
            raise SystemExit(f"flat context differs from naive_rag_baseline.retrieve for {q['id']}")
    return len(questions)


def mode_retrieval(args, cfg):
    device = resolve_device(args.device)
    questions = load_questions(args.questions or [cfg["questions_file"]], args.limit)
    if not questions:
        raise SystemExit("no questions")
    mps_hw = MpsHighWater(device == "mps")
    rss_after_import = peak_rss_bytes()

    stack, load = load_retrieval_stack(cfg, device)
    rss_after_load = peak_rss_bytes()
    mps_hw.sample()
    print(f"loaded on {device}: " + ", ".join(f"{k} {v*1000:.0f} ms" for k, v in load.items()))
    print(f"proxy vectors {stack['pp_index'].ntotal}, flat vectors {stack['flat_index'].ntotal}, "
          f"questions {len(questions)}")

    # warmup (untimed) + equivalence check against the production functions
    rng = random.Random(SEED)
    warm = [questions[i % len(questions)] for i in range(args.warmup)]
    for q in warm:
        run_proxy(stack, q["question"])
        run_flat(stack, q["question"])
    n_eq = check_equivalence(cfg, stack, questions[: min(len(questions), 25)])
    print(f"warmup {len(warm)} queries/system; staged == production for {n_eq} questions")

    records = []
    ctx_len = {"proxy_pointer": {}, "flat": {}}
    t_all = pc()
    for r in range(args.repeats):
        order = list(range(len(questions)))
        rng.shuffle(order)
        for j, qi in enumerate(order):
            q = questions[qi]
            systems = ("proxy_pointer", "flat") if (j + r) % 2 == 0 else ("flat", "proxy_pointer")
            for s in systems:
                ctx, t = (run_proxy if s == "proxy_pointer" else run_flat)(stack, q["question"])
                records.append({"system": s, "repeat": r, "qid": q["id"], **t})
                ctx_len[s][q["id"]] = len(ctx)
            mps_hw.sample()
    wall = pc() - t_all

    out_stats = {}
    for s, stages in (("proxy_pointer", ("embed", "search", "deref", "assemble", "retrieval_total")),
                      ("flat", ("embed", "search", "assemble", "retrieval_total"))):
        rs = [x for x in records if x["system"] == s]
        out_stats[s] = {st: stats_ms([x[st] for x in rs]) for st in stages}
        lens = list(ctx_len[s].values())
        out_stats[s]["context_chars"] = stats_plain(lens)
        out_stats[s]["context_over_cap"] = int(sum(1 for v in lens if v > PUBLISHED_CAP))

    result = {
        "mode": "retrieval",
        "env": environment(cfg, {"embedder": device}, [EMBED_MODEL]),
        "config": {"repeats": args.repeats, "warmup_per_system": args.warmup,
                   "n_questions": len(questions),
                   "question_sources": _sources(args.questions or [cfg["questions_file"]]),
                   "proxy_top_k": PROXY_TOP_K, "flat_top_k": FLAT_TOP_K,
                   "proxy_vectors": int(stack["pp_index"].ntotal),
                   "flat_vectors": int(stack["flat_index"].ntotal),
                   "order": "questions shuffled per repeat (seed 0); system order alternates",
                   "embedder_shared": True},
        "load_ms": {k: v * 1000 for k, v in load.items()},
        "import_s": _IMPORT_S,
        "stats_ms": out_stats,
        "timed_wall_s": wall,
        "memory": {"peak_rss_bytes": peak_rss_bytes(),
                   "rss_high_water_after_imports_bytes": rss_after_import,
                   "rss_high_water_after_load_bytes": rss_after_load,
                   "mps": mps_hw.as_dict(),
                   "label": "unified-memory high-water mark / peak RSS (not VRAM)"},
        "records": records,
    }
    result["env"]["loadavg_end"] = list(os.getloadavg())

    rows = []
    for s, st in out_stats.items():
        for stage in ("embed", "search", "deref", "assemble", "retrieval_total"):
            if stage in st:
                v = st[stage]
                rows.append([s, stage, f"{v['p50']:.3f}", f"{v['p95']:.3f}", f"{v['mean']:.3f}", v["n"]])
    print()
    print(fmt_table(rows, ["system", "stage", "p50 ms", "p95 ms", "mean ms", "n"]))
    print(f"\npeak RSS {peak_rss_bytes()/2**20:.0f} MiB; torch threads {torch.get_num_threads()}")
    if mps_hw.enabled:
        print(f"MPS driver high-water {mps_hw.driver/2**20:.0f} MiB")
    write_out(args.out, result)


# --------------------------------------------------------------------------- #
# Build: mirrors proxy_pointer_indexer.main and naive_rag_baseline.build_index
# --------------------------------------------------------------------------- #
def mode_build(args, cfg):
    import pickle
    import faiss
    from sentence_transformers import SentenceTransformer
    from corpus_config import ensure_parent_dir
    from document_tree_builder import DocumentTree
    from proxy_pointer_indexer import get_doc_id
    from retrieval_metrics import proxy_title_firstline  # == proxy_pointer_indexer's proxy
    import naive_rag_baseline as nrb

    device = resolve_device(args.device)
    flat_source = args.flat_source or cfg["baseline_source"]
    mps_hw = MpsHighWater(device == "mps")

    t = pc()
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    embedder_load = pc() - t
    mps_hw.sample()

    def files_size(paths):
        return {os.path.basename(p): os.path.getsize(p) for p in paths}

    runs = {"proxy_pointer": [], "flat": []}
    sizes, counts = {}, {}
    for r in range(args.repeats):
        # ---- proxy-pointer ----
        t0 = pc()
        tree = DocumentTree()
        for path in cfg["input_files"]:
            tree.add_file(path)
        t1 = pc()
        nodes = [n for n in tree.nodes.values()
                 if n.node_id != tree.root.node_id and not n.heading_title.startswith("[FILE] ")]
        texts = [proxy_title_firstline(n) for n in nodes]
        metadata = [{"pointer_node_id": n.node_id, "heading_title": n.heading_title,
                     "doc_id": get_doc_id(n, tree), "start_line": n.start_line,
                     "end_line": n.end_line} for n in nodes]
        t2 = pc()
        vecs = embedder.encode(texts, normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
        t3 = pc()
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        t4 = pc()
        for p in (cfg["pp_index_file"], cfg["pp_meta_file"], cfg["pp_tree_file"]):
            ensure_parent_dir(p)
        faiss.write_index(index, cfg["pp_index_file"])
        with open(cfg["pp_meta_file"], "wb") as f:
            pickle.dump(metadata, f)
        with open(cfg["pp_tree_file"], "wb") as f:
            pickle.dump(tree, f)
        t5 = pc()
        mps_hw.sample()
        runs["proxy_pointer"].append({"tree_build": t1 - t0, "proxy_texts": t2 - t1,
                                      "embed": t3 - t2, "faiss_add": t4 - t3,
                                      "write": t5 - t4, "total": t5 - t0})
        counts["proxy_pointer"] = {"vectors": int(index.ntotal), "dim": int(vecs.shape[1]),
                                   "tree_nodes": len(tree.nodes)}
        sizes["proxy_pointer"] = files_size([cfg["pp_index_file"], cfg["pp_meta_file"],
                                             cfg["pp_tree_file"]])

        # ---- flat ----
        t0 = pc()
        docs = nrb.load_documents(flat_source)
        t1 = pc()
        chunks = nrb.chunk_documents(docs)
        t2 = pc()
        vecs = embedder.encode(chunks, normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
        t3 = pc()
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        t4 = pc()
        for p in (cfg["flat_index_file"], cfg["flat_chunks_file"]):
            ensure_parent_dir(p)
        faiss.write_index(index, cfg["flat_index_file"])
        with open(cfg["flat_chunks_file"], "wb") as f:
            pickle.dump(chunks, f)
        t5 = pc()
        mps_hw.sample()
        runs["flat"].append({"load_source": t1 - t0, "chunk": t2 - t1, "embed": t3 - t2,
                             "faiss_add": t4 - t3, "write": t5 - t4, "total": t5 - t0})
        counts["flat"] = {"vectors": int(index.ntotal), "dim": int(vecs.shape[1]),
                          "source": flat_source,
                          "mean_chunk_chars": float(np.mean([len(c) for c in chunks]))}
        sizes["flat"] = files_size([cfg["flat_index_file"], cfg["flat_chunks_file"]])
        print(f"build repeat {r + 1}/{args.repeats}: proxy {runs['proxy_pointer'][-1]['total']:.2f} s, "
              f"flat {runs['flat'][-1]['total']:.2f} s")

    summary = {}
    for s, rs in runs.items():
        summary[s] = {k: stats_ms([x[k] for x in rs]) for k in rs[0]}
        summary[s]["disk_bytes"] = sizes[s]
        summary[s]["disk_bytes_total"] = int(sum(sizes[s].values()))
        summary[s].update(counts[s])

    result = {
        "mode": "build",
        "env": environment(cfg, {"embedder": device}, [EMBED_MODEL]),
        "config": {"repeats": args.repeats, "flat_source": flat_source,
                   "chunk_size": nrb.CHUNK_SIZE, "chunk_overlap": nrb.CHUNK_OVERLAP,
                   "encode_batch_size": 32,
                   "note": "embedder load timed separately; repeat 1 includes cold file cache"},
        "embedder_load_ms": embedder_load * 1000,
        "summary_ms": summary,
        "runs_s": runs,
        "memory": {"peak_rss_bytes": peak_rss_bytes(), "mps": mps_hw.as_dict(),
                   "label": "unified-memory high-water mark / peak RSS (not VRAM)"},
    }
    result["env"]["loadavg_end"] = list(os.getloadavg())
    rows = []
    for s, sm in summary.items():
        stages = [k for k in runs[s][0]]
        rows.append([s, sm["vectors"], f"{sm['disk_bytes_total']/1024:.0f} KiB"] +
                    [f"{k} {sm[k]['p50']:.0f}" for k in stages])
    print()
    print(fmt_table(rows, ["system", "vectors", "disk", "stage p50 ms ..."] + [""] * 5))
    write_out(args.out, result)


# --------------------------------------------------------------------------- #
# Generation: identical settings for both systems
# --------------------------------------------------------------------------- #
GEN_SYSTEM_SUFFIX = ("Answer the question using ONLY the context below. "
                     "If the answer is not in the context, say you don't know.")
FAST_SYSTEM_SUFFIX = "Answer the operator's question concisely and accurately."


def mode_generate(args, cfg):
    from corpus_config import assistant_intro
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    gen_device = args.gen_device
    mps_hw = MpsHighWater(gen_device == "mps")
    questions = load_questions(args.questions or [cfg["questions_file"]])
    rng = random.Random(SEED)
    sample = rng.sample(questions, min(args.gen_n, len(questions)))

    # 1. retrieval contexts (CPU embedder; retrieval is not what is timed here)
    stack, _ = load_retrieval_stack(cfg, "cpu")
    contexts = {}
    for q in sample:
        contexts[q["id"]] = {"proxy_pointer": run_proxy(stack, q["question"])[0],
                             "flat": run_flat(stack, q["question"])[0]}
    rss_retrieval = peak_rss_bytes()

    # 2. generator, loaded once (same weights/dtype/device as the published run)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.gen_dtype]
    t = pc()
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=dtype)
    model.to(gen_device)
    model.eval()
    gen_load = pc() - t
    rss_after_gen_load = peak_rss_bytes()
    mps_hw.sample()
    window = int(getattr(model.config, "max_position_embeddings", 2048))
    max_prompt = window - args.max_new_tokens

    class FirstToken(StoppingCriteria):
        def __init__(self):
            self.t = None

        def __call__(self, input_ids, scores, **kw):
            if self.t is None:
                if gen_device == "mps":
                    torch.mps.synchronize()
                self.t = pc()
            return False

    def build_ids(question, context, system_suffix):
        if context is None:
            msgs = [{"role": "system", "content": assistant_intro(cfg) + " " + system_suffix},
                    {"role": "user", "content": question}]
        else:
            msgs = [{"role": "system", "content": assistant_intro(cfg) + " " + system_suffix},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")

    def fit(question, context):
        """Shrink the context until prompt + max_new_tokens fits TinyLlama's window."""
        ids = build_ids(question, context, GEN_SYSTEM_SUFFIX)
        cut = False
        while ids.shape[1] > max_prompt:
            over = ids.shape[1] - max_prompt
            context = context[: max(0, len(context) - max(64, int(over * 3.5)))]
            ids = build_ids(question, context, GEN_SYSTEM_SUFFIX)
            cut = True
        return ids, context, cut

    def generate(ids):
        ids = ids.to(gen_device)
        ft = FirstToken()
        t0 = pc()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id,
                                 stopping_criteria=StoppingCriteriaList([ft]))
        if gen_device == "mps":
            torch.mps.synchronize()
        t1 = pc()
        new = int(out.shape[1] - ids.shape[1])
        ttft = (ft.t - t0) if ft.t else None
        decode_s = (t1 - ft.t) if ft.t else None
        answer = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return {"prompt_tokens": int(ids.shape[1]), "new_tokens": new, "total_s": t1 - t0,
                "ttft_s": ttft,
                "decode_tok_s": ((new - 1) / decode_s) if decode_s and new > 1 else None,
                "end_to_end_tok_s": new / (t1 - t0), "answer_chars": len(answer.strip())}

    # warmup, untimed
    generate(build_ids("Warm-up question.", "Warm-up context.", GEN_SYSTEM_SUFFIX))

    caps = [None if c in ("none", "0") else int(c) for c in args.caps]
    records = []
    t_all = pc()
    for i, q in enumerate(sample):
        for cap in caps:
            systems = ("proxy_pointer", "flat") if i % 2 == 0 else ("flat", "proxy_pointer")
            for s in systems:
                full = contexts[q["id"]][s]
                ctx = full if cap is None else full[:cap]
                ids, used, cut = fit(q["question"], ctx)
                rec = generate(ids)
                mps_hw.sample()
                rec.update({"system": s, "cap": cap if cap is not None else "none",
                            "qid": q["id"], "context_chars_full": len(full),
                            "context_chars_used": len(used), "window_truncated": cut})
                records.append(rec)
                print(f"  q{q['id']:>6} {s:13s} cap={rec['cap']!s:5s} ctx={len(used):5d}ch "
                      f"prompt={rec['prompt_tokens']:4d}tok new={rec['new_tokens']:3d} "
                      f"{rec['total_s']:6.2f}s ttft={rec['ttft_s']:.2f}s", flush=True)
        if args.fast_arm:
            ids = build_ids(q["question"], None, FAST_SYSTEM_SUFFIX)
            rec = generate(ids)
            rec.update({"system": "fast_no_retrieval", "cap": "n/a", "qid": q["id"],
                        "context_chars_full": 0, "context_chars_used": 0,
                        "window_truncated": False})
            records.append(rec)
            print(f"  q{q['id']:>6} fast(no ctx)  prompt={rec['prompt_tokens']:4d}tok "
                  f"new={rec['new_tokens']:3d} {rec['total_s']:6.2f}s", flush=True)
    wall = pc() - t_all

    groups = {}
    for rec in records:
        groups.setdefault(f"{rec['system']}|cap={rec['cap']}", []).append(rec)
    summary = {}
    for g, rs in groups.items():
        summary[g] = {
            "n": len(rs),
            "total_s": stats_plain([x["total_s"] for x in rs]),
            "ttft_s": stats_plain([x["ttft_s"] for x in rs if x["ttft_s"] is not None]),
            "decode_tok_s": stats_plain([x["decode_tok_s"] for x in rs if x["decode_tok_s"]]),
            "end_to_end_tok_s": stats_plain([x["end_to_end_tok_s"] for x in rs]),
            "prompt_tokens": stats_plain([x["prompt_tokens"] for x in rs]),
            "new_tokens": stats_plain([x["new_tokens"] for x in rs]),
            "context_chars_used": stats_plain([x["context_chars_used"] for x in rs]),
            "window_truncated": int(sum(1 for x in rs if x["window_truncated"])),
        }
    pt = np.array([x["prompt_tokens"] for x in records], dtype=float)
    tt = np.array([x["ttft_s"] or np.nan for x in records], dtype=float)
    ok = ~np.isnan(tt)
    fit_ttft = None
    if ok.sum() >= 3:
        slope, intercept = np.polyfit(pt[ok], tt[ok], 1)
        fit_ttft = {"ms_per_prompt_token": float(slope * 1000), "intercept_ms": float(intercept * 1000),
                    "r": float(np.corrcoef(pt[ok], tt[ok])[0, 1])}

    result = {
        "mode": "generate",
        "env": environment(cfg, {"embedder": "cpu", "generator": gen_device}, [EMBED_MODEL, GEN_MODEL]),
        "config": {"gen_model": GEN_MODEL, "dtype": args.gen_dtype, "device": gen_device,
                   "max_new_tokens": args.max_new_tokens, "do_sample": False,
                   "caps": [c if c is not None else "none" for c in caps],
                   "window_tokens": window,
                   "question_sources": _sources(args.questions or [cfg["questions_file"]]),
                   "sample_ids": [q["id"] for q in sample],
                   "sample_seed": SEED, "fast_arm": bool(args.fast_arm),
                   "prompt": "same system prompt and user template for both systems "
                             "(vanilla_answer's grounded prompt); only the context differs",
                   "uncapped_note": "'none' = no 1,200-char cap; context is still shortened, "
                                    "if needed, so prompt + max_new_tokens fits the model window"},
        "generator_load_s": gen_load,
        "summary": summary,
        "ttft_vs_prompt_tokens": fit_ttft,
        "timed_wall_s": wall,
        "memory": {"peak_rss_bytes": peak_rss_bytes(),
                   "rss_high_water_after_retrieval_load_bytes": rss_retrieval,
                   "rss_high_water_after_generator_load_bytes": rss_after_gen_load,
                   "mps": mps_hw.as_dict(),
                   "label": "unified-memory high-water mark / peak RSS (not VRAM)"},
        "records": records,
    }
    result["env"]["loadavg_end"] = list(os.getloadavg())
    rows = []
    for g, sm in summary.items():
        rows.append([g, sm["n"], f"{sm['context_chars_used']['mean']:.0f}",
                     f"{sm['prompt_tokens']['mean']:.0f}", f"{sm['new_tokens']['mean']:.0f}",
                     f"{sm['ttft_s']['p50']:.2f}" if sm["ttft_s"] else "-",
                     f"{sm['total_s']['p50']:.2f}", f"{sm['total_s']['mean']:.2f}",
                     f"{sm['decode_tok_s']['p50']:.1f}" if sm["decode_tok_s"] else "-",
                     sm["window_truncated"]])
    print()
    print(fmt_table(rows, ["arm", "n", "ctx chars", "prompt tok", "new tok", "TTFT p50 s",
                           "total p50 s", "total mean s", "decode tok/s", "window-cut"]))
    print(f"\ngenerator load {gen_load:.1f} s; peak RSS {peak_rss_bytes()/2**20:.0f} MiB; "
          f"timed wall {wall/60:.1f} min")
    write_out(args.out, result)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["retrieval", "build", "generate"])
    ap.add_argument("--corpus", choices=["public", "private"], default=None,
                    help="default: $PPRAG_CORPUS (then 'private')")
    ap.add_argument("--questions", nargs="+", default=None,
                    help="question JSONL file(s); duplicates by text are dropped "
                         "(default: the corpus' questions_file)")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20, help="untimed queries per system")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"],
                    help="embedding device (SentenceTransformer's own default on a Mac is mps)")
    ap.add_argument("--flat-source", choices=["pdf", "markdown"], default=None,
                    help="build: flat-baseline source (default: the corpus config)")
    ap.add_argument("--gen-n", type=int, default=10, help="generate: sample size")
    ap.add_argument("--caps", nargs="+", default=[str(PUBLISHED_CAP), "none"],
                    help="generate: context caps in chars; 'none' = uncapped")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--gen-device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--gen-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--fast-arm", action="store_true",
                    help="generate: also time the router's fast route (no retrieval, "
                         "ungrounded prompt) on the same questions")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    name = args.corpus or os.environ.get("PPRAG_CORPUS", "private")
    os.environ["PPRAG_CORPUS"] = name  # modules imported below read it
    sys.path.insert(0, HERE)
    from corpus_config import get_config
    cfg = get_config(name)
    check_out_path(cfg, args.out)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    {"retrieval": mode_retrieval, "build": mode_build, "generate": mode_generate}[args.mode](args, cfg)


_IMPORT_S = time.perf_counter() - _T_IMPORT0

if __name__ == "__main__":
    main()
