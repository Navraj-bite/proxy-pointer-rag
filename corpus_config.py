"""
Shared corpus configuration.

Select the corpus with the PPRAG_CORPUS environment variable:

    PPRAG_CORPUS=private   (default) the original proprietary manuals; file
                           names are exactly what the scripts used before
    PPRAG_CORPUS=public    PX4 User Guide subset fetched by
                           scripts/fetch_corpus.sh (see docs/corpus.md)

Every script that reads a corpus, builds an index or reads questions takes its
paths from get_config(). Public artifacts are namespaced under
artifacts/public/ so they can never overwrite the private ones.
"""

import os

ENV_VAR = "PPRAG_CORPUS"
DEFAULT_CORPUS = "private"

_PX4_SECTIONS = [
    "getting_started",
    "config",
    "config_mc",
    "flying",
    "flight_modes",
    "flight_modes_mc",
    "advanced_config",
]

CORPORA = {
    "private": {
        "name": "private",
        # markdown produced by parse_documents.py (LlamaParse)
        "input_files": [
            "machine_overview_troubleshooting.md",
            "operator_manual.md",
        ],
        # flat baseline source: "pdf" (pypdf text of the PDFs) or "markdown"
        "baseline_source": "pdf",
        "pdf_paths": [
            "resources/machine_overview_troubleshooting.pdf",
            "resources/operator_manual.pdf",
        ],
        "pp_index_file": "pp_faiss.index",
        "pp_meta_file": "pp_metadata.pkl",
        "pp_tree_file": "pp_tree.pkl",
        "flat_index_file": "faiss.index",
        "flat_chunks_file": "chunks.pkl",
        "questions_file": "benchmark_questions.jsonl",
        "results_csv": "proxy_rag_vs_vanilla_rag_results.csv",
        "report_md": "proxy_rag_vs_vanilla_rag.md",
        "retrieval_csv": "proxy_rag_vs_vanilla_rag_retrieval.csv",
        # used as "You are a support assistant for <assistant_subject>."
        "assistant_subject": "Serac filling machines",
    },
    "public": {
        "name": "public",
        "input_files": [f"corpus/px4_{s}.md" for s in _PX4_SECTIONS],
        "baseline_source": "markdown",
        "pdf_paths": [],
        "pp_index_file": "artifacts/public/pp_faiss.index",
        "pp_meta_file": "artifacts/public/pp_metadata.pkl",
        "pp_tree_file": "artifacts/public/pp_tree.pkl",
        "flat_index_file": "artifacts/public/faiss.index",
        "flat_chunks_file": "artifacts/public/chunks.pkl",
        "questions_file": "questions/public_seed.jsonl",
        "results_csv": "artifacts/public/proxy_rag_vs_vanilla_rag_results.csv",
        "report_md": "artifacts/public/proxy_rag_vs_vanilla_rag.md",
        "retrieval_csv": "artifacts/public/proxy_rag_vs_vanilla_rag_retrieval.csv",
        "assistant_subject": "the PX4 Autopilot flight stack (drones and other uncrewed vehicles)",
    },
}


def corpus_name():
    name = os.environ.get(ENV_VAR, DEFAULT_CORPUS).strip().lower() or DEFAULT_CORPUS
    if name not in CORPORA:
        raise SystemExit(f"{ENV_VAR}={name!r} is not one of {sorted(CORPORA)}")
    return name


def get_config(name=None):
    """Return a copy of the config dict for `name` (default: $PPRAG_CORPUS)."""
    return dict(CORPORA[name or corpus_name()])


def ensure_parent_dir(path):
    """Create the parent directory of an output file if it has one."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def assistant_intro(cfg=None):
    """'You are a support assistant for <subject>.' — byte-identical for private."""
    cfg = cfg or get_config()
    return f"You are a support assistant for {cfg['assistant_subject']}."
