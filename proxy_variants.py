"""
Proxy constructions for the proxy-pointer index (Task 8 sweep).

Every builder returns a callable(node) -> str, the text that gets embedded for
that heading node. Delivery is unchanged by the proxy: a hit still dereferences
to the node + its direct children (context_assembler).

    make_proxy_funcs(tree)  -> {name: callable(node) -> str}
    node_pieces(node, ...)  -> [str]   (multi-vector: one embedding per piece)

Variants
--------
  title            heading text only
  title_firstline  heading + first body line not starting with '|' or '#',
                   cut to 200 chars (the shipped proxy; proxy_pointer_indexer.py)
  title_body<N>    heading + first N chars of the body, whitespace collapsed
                   (raw markdown otherwise: tables, links and URLs included,
                   exactly what a flat chunk would see)
  title_keywords   heading + table cells + top-K TF-IDF body terms (below)
  body_full        heading + the whole body, whitespace collapsed
  path_firstline   ancestor heading titles joined with " > " (file root down
                   to the node) + the shipped first line. Many headings repeat
                   ("Parameters" x15 in the public corpus), so ancestry
                   disambiguates them.
  multivector      (not a single text; see node_pieces) heading + body split
                   with RecursiveCharacterTextSplitter(500, 50); the node's
                   score is the max over its pieces.

Keyword method (title_keywords), deterministic:
  * Clean the body: drop URLs, markdown link targets "(...)" after "]", HTML
    tags, backticks and emphasis markers.
  * Tokens: [A-Za-z][A-Za-z0-9_]+ lowercased, length >= 3, not in a fixed
    English stopword list.
  * TF-IDF with each indexable node's body as one document: tf = raw count in
    the node, idf = ln((1 + N) / (1 + df)) + 1 (sklearn's smoothed idf).
    Keep the top KW_TOP terms by tf*idf, ties broken alphabetically.
  * Table nodes: every pipe table's header-row cells plus the first-column cell
    of each data row (separator rows skipped), cleaned the same way,
    de-duplicated in order, capped at KW_TABLE_CELLS cells.
  * Text = heading, then table cells, then keywords (so if the encoder
    truncates, the keywords go first).
"""

import math
import re
from collections import Counter

KW_TOP = 20
KW_TABLE_CELLS = 40
MV_SIZE = 500
MV_OVERLAP = 50
BODY_CHARS = (300, 600, 1200)

_STOP = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each either else
etc few for from further get gets had has have having he her here hers him his how however
i if in into is it its itself just may might more most must my no nor not now of off on
once one only or other our out over own per same see set sets she should so some such than
that the their them then there these they this those through thus to too under until up
use used uses using very via was we were what when where which while who whom why will with
would you your yes can't cannot don't
""".split())

_URL = re.compile(r"https?://\S+")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_MARK = re.compile(r"[`*]+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_SEP_ROW = re.compile(r"^\|?\s*:?-{2,}")


def _collapse(s):
    return " ".join(s.split())


def _clean(s):
    s = _URL.sub(" ", s)
    s = _LINK_TARGET.sub("]", s)
    s = _HTML.sub(" ", s)
    s = _MARK.sub("", s)
    s = s.replace("[", "").replace("]", "").replace("!", " ")
    return _collapse(s)


def first_line(node):
    """The shipped proxy's body line: first line not starting with '|' or '#', <= 200 chars."""
    for line in node.body_content.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("|") and not line.startswith("#"):
            return line[:200]
    return ""


def tokens(text):
    return [t for t in (m.lower() for m in _TOKEN.findall(_clean(text)))
            if len(t) >= 3 and t not in _STOP]


def table_cells(body, cap=KW_TABLE_CELLS):
    """Header-row cells + first-column cells of every pipe table in *body*."""
    cells, in_table, header_done = [], False, False
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_table = False
            continue
        if _SEP_ROW.match(line):
            continue
        parts = [_clean(c) for c in line.strip("|").split("|")]
        if not in_table:
            in_table, header_done = True, False
        if not header_done:
            cells.extend(parts)
            header_done = True
        elif parts:
            cells.append(parts[0])
    out, seen = [], set()
    for c in cells:
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:cap]


class KeywordExtractor:
    def __init__(self, nodes, top=KW_TOP):
        self.top = top
        self.tf = {n.node_id: Counter(tokens(n.body_content)) for n in nodes}
        df = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(nodes)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}

    def keywords(self, node):
        tf = self.tf.get(node.node_id) or Counter(tokens(node.body_content))
        scored = sorted(((-c * self.idf.get(t, 1.0), t) for t, c in tf.items()))
        return [t for _, t in scored[:self.top]]


def heading_titles(tree, node):
    titles, cur = [], node
    while cur is not None and not cur.heading_title.startswith("[FILE] ") \
            and cur.node_id != tree.root.node_id:
        titles.append(cur.heading_title)
        cur = tree.nodes[cur.parent_node_id] if cur.parent_node_id else None
    return titles[::-1]


def make_proxy_funcs(tree, body_chars=BODY_CHARS, kw_top=KW_TOP):
    """name -> callable(node) -> proxy text, in a fixed display order (small to large)."""
    from retrieval_metrics import indexable_nodes, proxy_title, proxy_title_firstline
    kx = KeywordExtractor(indexable_nodes(tree), kw_top)

    def body_n(n):
        def fn(node):
            body = _collapse(node.body_content)
            return f"{node.heading_title}\n{body[:n]}" if body else node.heading_title
        fn.__name__ = f"title_body{n}"
        return fn

    def title_keywords(node):
        parts = [node.heading_title]
        cells = table_cells(node.body_content)
        if cells:
            parts.append(" ".join(cells))
        kws = kx.keywords(node)
        if kws:
            parts.append(" ".join(kws))
        return "\n".join(parts)

    def body_full(node):
        body = _collapse(node.body_content)
        return f"{node.heading_title}\n{body}" if body else node.heading_title

    def path_firstline(node):
        path = " > ".join(_collapse(t) for t in heading_titles(tree, node))
        line = first_line(node)
        return f"{path}\n{line}" if line else path

    funcs = {"title": proxy_title, "title_firstline": proxy_title_firstline}
    funcs["path_firstline"] = path_firstline
    funcs["title_keywords"] = title_keywords
    for n in body_chars:
        funcs[f"title_body{n}"] = body_n(n)
    funcs["body_full"] = body_full
    return funcs


def node_pieces(node, size=MV_SIZE, overlap=MV_OVERLAP):
    """Multi-vector pieces: heading + body split like a flat chunker, confined to the node."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    body = node.body_content.strip()
    text = f"{node.heading_title}\n{body}" if body else node.heading_title
    if len(text) <= size:
        return [text]
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)
