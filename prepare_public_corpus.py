#!/usr/bin/env python3
"""
Deterministic conversion of the pinned PX4 User Guide subset into
heading-structured markdown for the proxy-pointer pipeline.

Input : the raw VitePress markdown fetched by scripts/fetch_corpus.sh
        (corpus/raw/px4-<commit>/docs/en/<section>/<page>.md)
Output: one markdown file per guide section, corpus/px4_<section>.md, plus
        corpus/ATTRIBUTION.md and a copy of the upstream licence.

Layout decision (see docs/corpus.md):
  * one output file per section directory -> one [FILE] node per section
  * each page keeps its own '#' title as a level-1 heading; the page's
    '##'..'######' headings keep their relative depth
  * skipped heading levels inside a page are closed up (an '###' directly
    under '#' becomes '##'), which does not change any parent/child link
  * pages are ordered index.md first, then alphabetically by file name

Cleaning (all deterministic, stdlib only, no network):
  * YAML front matter removed (its 'title' is used if a page has no '#')
  * VitePress includes (<!--@include: _partial.md-->) inlined; Vue
    v-if/v-else-if/v-else <div>s on $frontmatter.frame evaluated for the
    including page, everything else about <div> dropped
  * HTML comments, <script>, <iframe>, <audio>, <lite-youtube>, <Redirect>,
    images and <a id> anchors removed; <Badge text="X"/> becomes "(X)"
  * links and reference links reduced to their text; link definitions dropped
  * ':::tip' / '::: info' / ':::warning' containers become a bold label line
  * '{#custom-id}' heading anchors removed
  * pipe tables normalised to '| a | b |' rows with '| --- |' delimiters and
    trimmed cells (the upstream files pad cells with hundreds of spaces)
  * numbered and bulleted lists are left exactly as written
  * inside fenced code blocks nothing is changed except that a line starting
    with '#' gets one leading space, so shell comments are not parsed as
    headings by document_tree_builder.py
  * pages left with no body text after cleaning (redirect stubs) are skipped

The input is untrusted data: this script only reads it, never executes it.
Run it with `python3 -I`.
"""

import argparse
import html
import os
import re
import shutil
import sys

# Guide sections, in reading order. Output file = corpus/px4_<section>.md
SECTIONS = [
    "getting_started",
    "config",
    "config_mc",
    "flying",
    "flight_modes",
    "flight_modes_mc",
    "advanced_config",
]
# Auto-generated 1.6 MB parameter dump: a reference table, not manual prose.
EXCLUDE = {"advanced_config/parameter_reference.md"}

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
CONTAINER_RE = re.compile(r"^(\s*):::+\s*([A-Za-z]*)\s*(.*?)\s*$")
LINK_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S+")
DELIM_CELL_RE = re.compile(r"^\s*:?-{1,}:?\s*$")
INCLUDE_RE = re.compile(r"<!--\s*@include:\s*([^\s>]+?)\s*-->")

CONTAINER_LABELS = {
    "info": "Note", "note": "Note", "tip": "Tip", "warning": "Warning",
    "danger": "Danger", "caution": "Caution", "details": "Details",
}


# --------------------------------------------------------------------------- #
# front matter, includes, Vue conditionals
# --------------------------------------------------------------------------- #
def split_front_matter(text):
    """Return (front_matter_dict, body). Only flat 'key: value' lines."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    fm = {}
    for line in text[4:end].splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$", line)
        if m:
            fm[m.group(1)] = m.group(2).strip("'\"")
    return fm, text[end + 5:]


def resolve_includes(text, page_dir, available):
    """Inline '<!--@include: _x.md-->' partials from the same section dir."""
    def repl(m):
        name = m.group(1)
        if "/" in name or not name.startswith("_") or not name.endswith(".md"):
            raise ValueError(f"unsupported include target {name!r}")
        path = os.path.join(page_dir, name)
        if path not in available:
            raise ValueError(f"include target {path} not in fetched set")
        with open(path, encoding="utf-8") as fh:
            _, body = split_front_matter(normalise_newlines(fh.read()))
        return body
    return INCLUDE_RE.sub(repl, text)


DIV_TOKEN_RE = re.compile(r"<div\b[^>]*>|</div\s*>", re.I)
VCOND_RE = re.compile(r"""\b(v-if|v-else-if|v-else)\b(?:\s*=\s*"([^"]*)")?""")
FRAME_COND_RE = re.compile(r"""^\s*\$frontmatter\.frame\s*===\s*'([^']*)'\s*$""")


def eval_vue_divs(text, front_matter):
    """
    Keep or drop the content of <div v-if/v-else-if/v-else> blocks according
    to the page's front matter, then drop every <div>/</div> tag.
    Only `$frontmatter.frame === '...'` conditions occur in this corpus;
    anything else raises so a new construct cannot be silently mangled.
    """
    out = []
    pos = 0
    # stack entries: [keep_this_div, chain_taken_at_this_depth_for_children]
    stack = []
    # chain_taken[depth] = whether an earlier branch of the current
    # v-if/v-else-if chain at that depth was taken.
    chain_taken = {}

    def visible():
        return all(entry[0] for entry in stack)

    for m in DIV_TOKEN_RE.finditer(text):
        if visible():
            out.append(text[pos:m.start()])
        pos = m.end()
        tag = m.group(0)
        depth = len(stack)
        if tag.startswith("</"):
            if not stack:
                raise ValueError("unbalanced </div>")
            stack.pop()
            continue
        cm = VCOND_RE.search(tag)
        if not cm:
            stack.append([True])
            chain_taken[depth] = None
            continue
        kind, cond = cm.group(1), cm.group(2)
        if kind in ("v-if", "v-else-if"):
            fm = FRAME_COND_RE.match(cond or "")
            if not fm:
                raise ValueError(f"unsupported Vue condition {cond!r}")
            result = front_matter.get("frame") == fm.group(1)
        else:
            result = True
        if kind == "v-if":
            keep = result
            chain_taken[depth] = keep
        else:
            if chain_taken.get(depth) is None:
                raise ValueError(f"{kind} without preceding v-if")
            keep = (not chain_taken[depth]) and result
            chain_taken[depth] = chain_taken[depth] or keep
        stack.append([keep])
    if stack:
        raise ValueError("unbalanced <div>")
    out.append(text[pos:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# inline cleaning (never applied inside code spans or fenced blocks)
# --------------------------------------------------------------------------- #
CODE_SPAN_RE = re.compile(r"(``[^`].*?``|`[^`]*`)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?:<[^>]*>|[^()\s]*(?:\([^()]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)")
LINK_RE = re.compile(r"\[([^\[\]]*)\]\((?:<[^>]*>|[^()\s]*(?:\([^()]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)")
REFLINK_RE = re.compile(r"\[([^\[\]]+)\]\[[^\[\]]*\]")
AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>\s]+)>")
BADGE_RE = re.compile(r"<Badge\b[^>]*?\btext=\"([^\"]*)\"[^>]*/?>", re.I)
DROP_BLOCK_RE = re.compile(r"<(audio|iframe)\b.*?</\1\s*>", re.I | re.S)
DROP_TAG_RE = re.compile(r"</?(?:img|source|lite-youtube|Redirect|audio|iframe)\b[^>]*>", re.I)
BADGE_ONLY_RE = re.compile(r"^\s*(<Badge\b[^>]*/>\s*)+$", re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)
UNWRAP_TAG_RE = re.compile(r"</?(?:a|span|sub|sup|p|kbd|b|i|em|strong|u|small|center|font)\b[^>]*>", re.I)


def clean_inline(segment):
    s = IMAGE_RE.sub("", segment)
    # Images nested in links leave '[](url)'; the link rule then removes it.
    for _ in range(2):  # nested '[text [inner](u)](u)' is rare; two passes cover it
        s = LINK_RE.sub(r"\1", s)
    s = REFLINK_RE.sub(r"\1", s)
    s = AUTOLINK_RE.sub(r"\1", s)
    s = BADGE_RE.sub(lambda m: f"({m.group(1)})", s)
    s = DROP_BLOCK_RE.sub("", s)
    s = DROP_TAG_RE.sub("", s)
    s = re.sub(r"^(\s*)<br\s*/?>\s*", r"\1", s, flags=re.I)   # leading <br>: drop
    s = BR_RE.sub(" ", s)
    s = UNWRAP_TAG_RE.sub("", s)
    return html.unescape(s)


def clean_line(line):
    # Swap code spans for placeholders so link text like [`PARAM`](url) is
    # still reduced to `PARAM` while nothing inside backticks is touched.
    spans = []

    def stash(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    s = clean_inline(CODE_SPAN_RE.sub(stash, line))
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)


def clean_heading_text(title):
    title = re.sub(r"\s*\{#[^}]*\}\s*$", "", title)
    title = clean_line(title)
    title = title.replace("`", "")
    title = re.sub(r"\*+", "", title)
    title = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", title)   # _emphasis_
    title = re.sub(r"\s+", " ", title).strip().rstrip("#").strip()
    return title


# --------------------------------------------------------------------------- #
# pipe tables
# --------------------------------------------------------------------------- #
def split_row(row):
    """Split a pipe-table row into cells, honouring `code` spans and \\|."""
    s = row.strip()
    cells, cur, in_code, i = [], [], False, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("\\|")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    if s.startswith("|"):
        cells = cells[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        cells = cells[:-1]
    return [re.sub(r"\s+", " ", c).strip() for c in cells]


def is_delimiter_row(line):
    if "-" not in line or "|" not in line:
        return False
    cells = split_row(line)
    return bool(cells) and all(DELIM_CELL_RE.match(c) for c in cells)


def normalise_tables(lines, in_code_flags):
    out = list(lines)
    i = 1
    n = len(lines)
    while i < n:
        if (not in_code_flags[i] and not in_code_flags[i - 1]
                and is_delimiter_row(lines[i]) and "|" in lines[i - 1]
                and lines[i - 1].strip()):
            indent = re.match(r"^\s*", lines[i - 1]).group(0)
            header = split_row(lines[i - 1])
            out[i - 1] = indent + "| " + " | ".join(header) + " |"
            out[i] = indent + "| " + " | ".join("---" for _ in header) + " |"
            j = i + 1
            while j < n and not in_code_flags[j] and lines[j].strip() and "|" in lines[j]:
                cells = split_row(lines[j])
                out[j] = indent + "| " + " | ".join(cells) + " |"
                j += 1
            i = j
        else:
            i += 1
    return out


PLAIN_TEXT_RE = re.compile(r"^\s*(?![|#>*+-]|\d+\.|`{3}|~{3})\S")


def merge_labels(lines, code_flags, label_idx):
    """
    Fold a bare '**Tip:**' container label into the following paragraph line
    ('**Tip:** Position mode is ...') so the label never stands alone as a
    section's first body line. Labels followed by a list/table/code stay alone.
    """
    out, flags, skip = [], [], set()
    for i, (line, flag) in enumerate(zip(lines, code_flags)):
        if i in skip:
            continue
        if i in label_idx:
            j = i + 1
            while j < len(lines) and not lines[j].strip() and not code_flags[j]:
                j += 1
            if j < len(lines) and not code_flags[j] and PLAIN_TEXT_RE.match(lines[j]) \
                    and not HEADER_RE.match(lines[j]):
                indent = re.match(r"^\s*", line).group(0)
                lines[j] = f"{indent}{line.strip()} {lines[j].strip()}"
                skip.update(range(i + 1, j))
                continue
        out.append(line)
        flags.append(flag)
    return out, flags


# --------------------------------------------------------------------------- #
# page conversion
# --------------------------------------------------------------------------- #
def normalise_newlines(text):
    return text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")


def convert_page(path, available, stats):
    with open(path, encoding="utf-8") as fh:
        raw = normalise_newlines(fh.read())
    fm, text = split_front_matter(raw)
    text = resolve_includes(text, os.path.dirname(path), available)
    text = eval_vue_divs(text, fm)

    lines_in = text.split("\n")
    out, code_flags = [], []
    in_fence, fence_tok = False, None
    in_comment = in_script = False
    label_lines = set()

    for line in lines_in:
        fm_match = FENCE_RE.match(line)
        if in_fence:
            if fm_match and fm_match.group(1)[0] == fence_tok[0] \
                    and len(fm_match.group(1)) >= len(fence_tok) \
                    and line.strip() == fm_match.group(1):
                in_fence = False
            elif re.match(r"^#{1,6}(\s|$)", line):
                line = " " + line
                stats["code_hash_lines_escaped"] += 1
            out.append(line.rstrip())
            code_flags.append(True)
            continue

        # multi-line HTML comments / script blocks (outside code only)
        if in_comment:
            if "-->" in line:
                in_comment = False
                line = line.split("-->", 1)[1]
            else:
                continue
        if in_script:
            if re.search(r"</script\s*>", line, re.I):
                in_script = False
            continue
        line = re.sub(r"<!--.*?-->", "", line)
        if "<!--" in line:
            line, in_comment = line.split("<!--", 1)[0], True
        if re.match(r"^\s*<script\b", line, re.I):
            in_script = not re.search(r"</script\s*>", line, re.I)
            continue

        if fm_match:
            in_fence, fence_tok = True, fm_match.group(1)
            out.append(line.rstrip())
            code_flags.append(True)
            continue

        cm = CONTAINER_RE.match(line)
        if cm:
            indent, kind, title = cm.groups()
            if kind:
                label = CONTAINER_LABELS.get(kind.lower(), kind.capitalize())
                title = clean_line(title).strip()
                if kind.lower() == "details":
                    out.append(f"{indent}**{title or label}:**")
                elif title:
                    out.append(f"{indent}**{label}:** {title}")
                else:
                    label_lines.add(len(out))   # merged into next line below
                    out.append(f"{indent}**{label}:**")
                code_flags.append(False)
            continue  # closing ':::' dropped

        if BADGE_ONLY_RE.match(line):
            continue  # page-level version badges carry no section content

        if LINK_DEF_RE.match(line):
            continue

        hm = HEADER_RE.match(line)
        if hm:
            title = clean_heading_text(hm.group(2))
            if title:
                out.append(("#" * len(hm.group(1))) + " " + title)
                code_flags.append(False)
            continue

        out.append(clean_line(line).rstrip())
        code_flags.append(False)

    out, code_flags = merge_labels(out, code_flags, label_lines)
    out = normalise_tables(out, code_flags)

    # Heading hierarchy: synthesise a page title if needed, close skipped levels.
    first_heading = next((i for i, (l, c) in enumerate(zip(out, code_flags))
                          if not c and HEADER_RE.match(l)), None)
    if first_heading is None or len(HEADER_RE.match(out[first_heading]).group(1)) != 1:
        title = fm.get("title") or os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
        out.insert(0, "# " + title)
        code_flags.insert(0, False)
        synthesised = True
    else:
        synthesised = False

    stack = []  # (source_level, output_level)
    closed_up = 0
    for i, (line, is_code) in enumerate(zip(out, code_flags)):
        if is_code:
            continue
        hm = HEADER_RE.match(line)
        if not hm:
            continue
        src = len(hm.group(1))
        while stack and stack[-1][0] >= src:
            stack.pop()
        new = (stack[-1][1] + 1) if stack else 1
        if new != src:
            closed_up += 1
        stack.append((src, new))
        out[i] = "#" * new + " " + hm.group(2)

    # Collapse blank-line runs outside code; trim.
    final, blank = [], False
    for line, is_code in zip(out, code_flags):
        if not is_code and not line.strip():
            if blank:
                continue
            blank = True
            final.append("")
            continue
        blank = False
        final.append(line)
    page = "\n".join(final).strip("\n")

    has_body = any(l.strip() and not HEADER_RE.match(l) for l in final)
    if not has_body:
        return None
    stats["synthesised_titles"] += int(synthesised)
    stats["levels_closed_up"] += closed_up
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True,
                    help="fetched tree root, e.g. corpus/raw/px4-<commit>")
    ap.add_argument("--out", default="corpus", help="output directory")
    ap.add_argument("--commit", default="<commit>",
                    help="upstream commit SHA, recorded in ATTRIBUTION.md")
    args = ap.parse_args()

    en = os.path.join(args.src, "docs", "en")
    available = set()
    for sec in SECTIONS:
        d = os.path.join(en, sec)
        for name in sorted(os.listdir(d)):
            if name.endswith(".md"):
                available.add(os.path.join(d, name))

    os.makedirs(args.out, exist_ok=True)
    stats = {"pages_kept": 0, "pages_skipped_empty": 0, "partials": 0,
             "synthesised_titles": 0, "levels_closed_up": 0,
             "code_hash_lines_escaped": 0}
    total_bytes = 0
    for sec in SECTIONS:
        d = os.path.join(en, sec)
        names = sorted(n for n in os.listdir(d) if n.endswith(".md"))
        names.sort(key=lambda n: (n != "index.md", n))
        pages = []
        for name in names:
            rel = f"{sec}/{name}"
            if rel in EXCLUDE:
                continue
            if name.startswith("_"):
                stats["partials"] += 1       # only rendered via @include
                continue
            page = convert_page(os.path.join(d, name), available, stats)
            if page is None:
                stats["pages_skipped_empty"] += 1
                print(f"  skip (no body after cleaning): {rel}")
                continue
            stats["pages_kept"] += 1
            pages.append(page)
        doc = "\n\n".join(pages) + "\n"
        out_path = os.path.join(args.out, f"px4_{sec}.md")
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(doc)
        total_bytes += len(doc.encode("utf-8"))
        print(f"  wrote {out_path}: {len(pages)} pages, {len(doc.encode('utf-8')):,} bytes")

    lic = os.path.join(args.src, "docs", "LICENSE")
    shutil.copyfile(lic, os.path.join(args.out, "LICENSE-PX4-DOCS"))
    with open(os.path.join(args.out, "ATTRIBUTION.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "The px4_*.md files in this directory are adapted from the PX4 User Guide\n"
            "(https://docs.px4.io), Copyright (c) PX4 Dev Team / Dronecode Project,\n"
            "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).\n"
            f"Source: https://github.com/PX4/PX4-Autopilot/tree/{args.commit}/docs/en\n"
            "(see docs/corpus.md in proxy-pointer-rag). Changes: converted by\n"
            "prepare_public_corpus.py (markup stripped, links reduced to text, images\n"
            "removed, tables normalised, pages concatenated per section).\n"
        )
    print("  " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    print(f"  total markdown: {total_bytes:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
