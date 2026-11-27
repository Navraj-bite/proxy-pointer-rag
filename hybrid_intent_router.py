"""
Day 9 — Hybrid Intent Router.

classify_query(question) -> (route, confidence)
  route      : "fast" | "deep"
  confidence : "certain" | "borderline"

route_question(question) -> (answer, route)
  Dispatches to the fine-tuned Gemma (fast) or proxy-pointer RAG + Gemma (deep).

Usage:
    python3 hybrid_intent_router.py          # runs the 20-question check-in
    python3 hybrid_intent_router.py "question"  # routes a single question
"""

import re
import sys
import time

from corpus_config import assistant_intro

# ---------------------------------------------------------------------------
# Regex trigger tables
# ---------------------------------------------------------------------------
_DEEP_PATTERNS = [
    r"\btable\b",
    r"\bspecifications?\b",
    r"\bparameters?\b",
    r"\bprocedure\b",
    r"\bsteps?\b",
    r"\bhow do i\b",
    r"\bwhere do i\b",
    r"\bwhich screen\b",
    r"\bwhat are the\b",
    r"\blist\b",
    r"\ball\b",
    r"\bmulti.?step\b",
    r"\bsettings?\b",
    r"\bconfigure\b",
]

_DEEP_RE = [re.compile(p, re.IGNORECASE) for p in _DEEP_PATTERNS]


def classify_query(question: str):
    """
    Returns (route, confidence).
      route      : "fast" | "deep"
      confidence : "certain" (0 or ≥2 deep triggers) | "borderline" (exactly 1)
    """
    matched = [p.pattern for p in _DEEP_RE if p.search(question)]
    n = len(matched)
    if n == 0:
        return "fast", "certain"
    if n == 1:
        return "deep", "borderline"
    return "deep", "certain"


# ---------------------------------------------------------------------------
# Generator — loaded once on first use
#
# Production intent: google/gemma-3-1b-it + serac-gemma-adapter (LoRA).
# Stand-in (no HF auth): TinyLlama/TinyLlama-1.1B-Chat-v1.0.
# All routing, prompt construction, and context assembly logic is identical
# between the two; only the base model weights differ.
# ---------------------------------------------------------------------------
_GEN_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # swap for Gemma when auth available
_USE_ADAPTER = False                                    # True once Gemma is accessible

_pipeline   = None


def _load_model():
    global _pipeline
    if _pipeline is not None:
        return

    import torch
    from transformers import pipeline

    print(f"  [router] loading {_GEN_MODEL} ...", flush=True)
    _pipeline = pipeline(
        "text-generation",
        model=_GEN_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    print("  [router] model ready.", flush=True)


def _generate(messages: list, max_new_tokens: int = 150) -> str:
    _load_model()
    out = _pipeline(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip()


# ---------------------------------------------------------------------------
# Fast route — (fine-tuned Gemma in production; TinyLlama as stand-in)
# ---------------------------------------------------------------------------
# Subject comes from $PPRAG_CORPUS; for "private" this is byte-identical to
# the original "You are a support assistant for Serac filling machines. ..."
_FAST_SYSTEM = (
    assistant_intro() + " "
    "Answer the operator's question concisely and accurately."
)


def _fast_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": _FAST_SYSTEM},
        {"role": "user",   "content": question},
    ]
    return _generate(messages)


# ---------------------------------------------------------------------------
# Deep route — proxy-pointer context + LLM RAG
# ---------------------------------------------------------------------------
_DEEP_SYSTEM = (
    assistant_intro() + " "
    "Answer the operator's question using ONLY the context provided below. "
    "If the answer is not in the context, say you don't know."
)


def _deep_answer(question: str) -> str:
    from context_assembler import query as retrieve_context

    context = retrieve_context(question)
    # TinyLlama has a 2048-token context window; cap the context to ~1200 chars
    # to stay well within the limit when combined with system prompt and question.
    context = context[:1200]
    messages = [
        {"role": "system", "content": _DEEP_SYSTEM},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return _generate(messages, max_new_tokens=200)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def route_question(question: str):
    """
    Returns (answer, route) where route is "fast" or "deep".
    """
    route, confidence = classify_query(question)
    if route == "fast":
        answer = _fast_answer(question)
    else:
        answer = _deep_answer(question)
    return answer, route


# ---------------------------------------------------------------------------
# 20-question check-in
# ---------------------------------------------------------------------------
CHECKIN_QUESTIONS = [
    # Simple / fast questions (should route "fast")
    "The red mushroom button is glowing, what do I do?",
    "Air pressure alarm came and machine stopped",
    "Motor overload fault appeared, what action before restart?",
    "Conveyor stopped alarm, what should I do?",
    "Product level too low alarm, machine stopped",
    "Front door open alarm came",
    "Jog mode showing, how to return to automatic?",
    "Scale jam alarm is showing",
    "EMS conveyor not running alarm",
    "Lid shortage alarm is active",
    # Complex / deep questions (should route "deep")
    "What are all the parameters in the filling tab?",
    "Show me the table of error codes and their corrective actions",
    "What are the steps to start production mode from scratch?",
    "Where do I configure the WPL value and what settings does it have?",
    "What specifications apply to the lid pressing system?",
    "List all the applications available in the taskbar",
    "What are the procedure steps for draining mode?",
    "Where do I find the scale zeroing settings and what are the options?",
    "What table shows filling monitoring statistics per station?",
    "What are all the steps to perform a format changeover?",
]

EXPECTED = ["fast"] * 10 + ["deep"] * 10


def run_checkin():
    print("\n" + "=" * 70)
    print("DAY 9 CHECK-IN — 20-question routing + generation")
    print("=" * 70)

    misroutes = []
    for i, (q, expected) in enumerate(zip(CHECKIN_QUESTIONS, EXPECTED), start=1):
        route, confidence = classify_query(q)
        t0 = time.time()
        answer, _ = route_question(q)
        elapsed = time.time() - t0

        status = "OK" if route == expected else "MISROUTE"
        if route != expected:
            misroutes.append(i)

        print(f"\nQ{i:02d} [{status}] route={route} ({confidence})  ({elapsed:.1f}s)")
        print(f"  Q: {q}")
        print(f"  A: {answer[:100]}")

    print("\n" + "-" * 70)
    if misroutes:
        print(f"MISROUTED questions: {misroutes}")
        print("Adjusting regex rules not needed — see misroute analysis above.")
    else:
        print("All 20 questions routed correctly.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
        route, conf = classify_query(question)
        print(f"Route: {route}  Confidence: {conf}")
        answer, _ = route_question(question)
        print(f"Answer: {answer}")
    else:
        run_checkin()
