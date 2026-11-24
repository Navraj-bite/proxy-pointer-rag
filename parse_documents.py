"""
Structure-aware PDF -> Markdown extraction for the Serac support chatbot.

Replaces Week 1's naive pypdf text dump (which flattened the CAUSE /
CONSEQUENCE / CORRECTIVE ACTION def-code tables into ambiguous paragraphs)
with LlamaParse markdown output, so that:

  * Multi-column troubleshooting tables come out as real `| pipe |` markdown
    tables (cause vs. corrective action stay in their own columns).
  * Nested step-by-step operator procedures keep their indentation as proper
    nested markdown lists.

IMPORTANT: llama-parse / llama-index-core require Python >= 3.10. The system
python3 here is 3.9, so run this with Homebrew's 3.11:

    /opt/homebrew/bin/python3.11 parse_documents.py
"""
import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------- CONFIG
# (input PDF, output markdown) pairs
DOCUMENTS = [
    ("resources/machine_overview_troubleshooting.pdf", "machine_overview_troubleshooting.md"),
    ("resources/operator_manual.pdf", "operator_manual.md"),
]

# Steer the parser toward faithful tables + nested procedures.
PARSING_INSTRUCTION = (
    "This is a technical manual for a Serac filling machine. "
    "Preserve every multi-column table (especially the def-code error tables "
    "with CAUSE, CONSEQUENCE, and CORRECTIVE ACTION columns) as a proper "
    "markdown pipe table with one column per source column - never merge or "
    "flatten the columns into a paragraph. "
    "Preserve numbered and bulleted step-by-step procedures as markdown lists, "
    "keeping sub-steps indented as nested list items so the hierarchy is intact."
)


def main():
    # Load LLAMA_CLOUD_API_KEY from .env
    load_dotenv()
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: LLAMA_CLOUD_API_KEY not found.\n"
            "Add it to a .env file in this directory:\n"
            "    LLAMA_CLOUD_API_KEY=llx-...\n"
            "Get the key from https://cloud.llamaindex.ai"
        )

    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        premium_mode=True,           # best fidelity for complex tables / layout
        parsing_instruction=PARSING_INSTRUCTION,
        verbose=True,
    )

    for pdf_path, out_path in DOCUMENTS:
        if not os.path.exists(pdf_path):
            print(f"  [SKIP] {pdf_path} not found.")
            continue

        print(f"Parsing {pdf_path} ...")
        documents = parser.load_data(pdf_path)
        markdown = "\n\n".join(doc.text for doc in documents if doc.text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"  -> wrote {out_path} ({len(documents)} pages, {len(markdown):,} chars)")


if __name__ == "__main__":
    main()
