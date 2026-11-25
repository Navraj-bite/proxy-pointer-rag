"""
Day 7 — Object-Oriented Tree Database Construction.

Parses markdown documents (real #/##/### ATX headers + | pipe tables) into a
nested tree of DocumentNode objects, then recursively validates that the
parent/child links are internally consistent.

Run:  python3 document_tree_builder.py
"""

import re

# A header line looks like:  "## 6.1. SELECT PRODUCTION FILE"
# 1-6 leading '#' tokens, then a space, then the heading text.
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# A line that begins a markdown numbered procedure step, e.g. "1. Do the thing".
NUMBERED_STEP_RE = re.compile(r"^\s*\d+\.\s+\S")

from corpus_config import get_config

# Selected by $PPRAG_CORPUS (default "private": the two original manuals).
INPUT_FILES = get_config()["input_files"]


class DocumentNode:
    """A single heading and the body that belongs to it."""

    def __init__(self, node_id, heading_title, parent_node_id, level, start_line=0):
        self.node_id = node_id
        self.heading_title = heading_title
        self.body_content = ""        # everything between this header and the
                                      # next header of equal-or-higher level
        self.parent_node_id = parent_node_id
        self.child_node_ids = []
        # 'level' is bookkeeping for nesting; not one of the required fields
        # but harmless to carry and useful for the integrity checks.
        self.level = level
        self.start_line = start_line  # line number of the heading in its source file
        self.end_line = 0             # filled in by add_file after full parse

    def __repr__(self):
        return f"DocumentNode({self.node_id!r}, {self.heading_title!r})"


class DocumentTree:
    """Holds every node and the synthetic ROOT that ties the forest together."""

    def __init__(self):
        self.nodes = {}               # node_id -> DocumentNode
        self._counter = 0
        # Synthetic root so traversal has a single starting point even though
        # the corpus spans multiple files each with many level-1 headers.
        self.root = self._new_node(heading_title="ROOT", parent_node_id=None, level=-1)

    def _new_node(self, heading_title, parent_node_id, level, start_line=0):
        node_id = f"n{self._counter:04d}"
        self._counter += 1
        node = DocumentNode(node_id, heading_title, parent_node_id, level, start_line)
        self.nodes[node_id] = node
        if parent_node_id is not None:
            self.nodes[parent_node_id].child_node_ids.append(node_id)
        return node

    def add_file(self, path):
        """Parse one markdown file and graft its headings into the tree."""
        # Read all lines up front so we know the total count for end_line.
        with open(path, "r", encoding="utf-8") as fh:
            all_lines = fh.readlines()
        total_lines = len(all_lines)

        # Each file gets a level-0 container node directly under ROOT so the
        # file boundary is visible in the printed hierarchy.
        file_node = self._new_node(heading_title=f"[FILE] {path}",
                                   parent_node_id=self.root.node_id, level=0,
                                   start_line=1)
        file_node.end_line = total_lines

        # creation_order tracks (start_line, node_id) in document order so we
        # can compute each node's end_line as next_start - 1 after parsing.
        creation_order = [(1, file_node.node_id)]

        # stack holds (level, node_id) for the current ancestry chain.
        # We seed it with the file node at level 0 so any level-1 header
        # ('#') attaches under the file.
        stack = [(0, file_node.node_id)]
        current = file_node           # node currently collecting body lines

        for lineno, raw_line in enumerate(all_lines, start=1):
            line = raw_line.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()

                # Pop ancestors until the top of the stack is strictly
                # shallower than this header -> that node is our parent.
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent_id = stack[-1][1] if stack else file_node.node_id

                node = self._new_node(heading_title=title,
                                      parent_node_id=parent_id, level=level,
                                      start_line=lineno)
                creation_order.append((lineno, node.node_id))
                stack.append((level, node.node_id))
                current = node
            else:
                # Body line: belongs to the most recent header (tables,
                # lists, numbered steps and prose all land here).
                current.body_content += line + "\n"

        # Assign end_line: each content node ends just before the next one.
        for i, (start, nid) in enumerate(creation_order):
            if nid == file_node.node_id:
                continue  # file node already got end_line = total_lines
            next_start = creation_order[i + 1][0] if i + 1 < len(creation_order) else total_lines + 1
            self.nodes[nid].end_line = next_start - 1

        # Tidy up trailing whitespace on every body we just built.
        for node in self.nodes.values():
            node.body_content = node.body_content.strip("\n")


def doc_id_of(tree, node):
    """Source file of a node: the path its [FILE] ancestor was added with."""
    cur = node
    while cur is not None:
        if cur.heading_title.startswith("[FILE] "):
            return cur.heading_title[len("[FILE] "):]
        cur = tree.nodes.get(cur.parent_node_id) if cur.parent_node_id else None
    return None


def heading_path(tree, node):
    """Heading titles from the top-level heading down to `node` (no ROOT/[FILE])."""
    path = []
    cur = node
    while cur is not None and cur.parent_node_id is not None \
            and not cur.heading_title.startswith("[FILE] "):
        path.append(cur.heading_title)
        cur = tree.nodes[cur.parent_node_id]
    return list(reversed(path))


def find_by_heading_path(tree, doc, path):
    """All nodes in file `doc` whose heading_path equals `path` (list of titles)."""
    return [
        n for n in tree.nodes.values()
        if n.parent_node_id is not None and not n.heading_title.startswith("[FILE] ")
        and heading_path(tree, n) == list(path) and doc_id_of(tree, n) == doc
    ]


# --------------------------------------------------------------------------- #
# Task 2 — recursive hierarchy integrity validation
# --------------------------------------------------------------------------- #
def validate_tree(tree):
    """
    Walk the whole tree from ROOT and confirm:
      * every child's parent_node_id points back at its real parent
      * every parent's child_node_ids lists exactly its real children
    Returns (errors, visited_count).
    """
    errors = []
    visited = set()

    def walk(node_id):
        node = tree.nodes[node_id]
        visited.add(node_id)

        for child_id in node.child_node_ids:
            child = tree.nodes.get(child_id)
            if child is None:
                errors.append(f"{node_id} lists missing child {child_id}")
                continue
            # Upward link must agree with the downward link.
            if child.parent_node_id != node_id:
                errors.append(
                    f"child {child_id} ('{child.heading_title}') has "
                    f"parent_node_id={child.parent_node_id} but is listed "
                    f"under {node_id} ('{node.heading_title}')"
                )
            walk(child_id)

    walk(tree.root.node_id)

    # Independent cross-check: rebuild children from the parent pointers and
    # compare against each node's declared child_node_ids.
    rebuilt = {nid: [] for nid in tree.nodes}
    for nid, node in tree.nodes.items():
        if node.parent_node_id is not None:
            rebuilt[node.parent_node_id].append(nid)
    for nid, node in tree.nodes.items():
        if set(rebuilt[nid]) != set(node.child_node_ids):
            errors.append(
                f"{nid} ('{node.heading_title}') child set mismatch: "
                f"declared={node.child_node_ids} actual={rebuilt[nid]}"
            )

    # Every node reachable from ROOT?
    unreached = set(tree.nodes) - visited
    if unreached:
        errors.append(f"{len(unreached)} node(s) unreachable from ROOT: {sorted(unreached)}")

    return errors, len(visited)


def check_def0006_and_steps(tree):
    """Targeted checks the task called out explicitly."""
    print("\n=== Targeted checks: def0006 table & numbered procedure steps ===")

    def0006 = next(
        (n for n in tree.nodes.values() if n.heading_title.lower().startswith("def0006")),
        None,
    )
    if def0006 is None:
        print("  ! def0006 node not found")
    else:
        parent = tree.nodes[def0006.parent_node_id]
        has_table = "| CAUSE" in def0006.body_content or "|---" in def0006.body_content \
            or re.search(r"^\|", def0006.body_content, re.M) is not None
        print(f"  def0006 node id      : {def0006.node_id}")
        print(f"  heading_title        : {def0006.heading_title}")
        print(f"  attached to parent   : {parent.node_id} ('{parent.heading_title}')")
        print(f"  table in body_content: {has_table} "
              f"({len(def0006.body_content)} chars of body retained)")
        print(f"  not orphaned         : {def0006.parent_node_id is not None}")

    # Find nodes whose body holds numbered procedure steps and confirm those
    # steps stayed inside the body rather than becoming stray nodes.
    step_nodes = [
        n for n in tree.nodes.values()
        if any(NUMBERED_STEP_RE.match(ln) for ln in n.body_content.splitlines())
    ]
    print(f"\n  nodes whose body_content retains numbered steps: {len(step_nodes)}")
    for n in step_nodes[:3]:
        steps = [ln.strip() for ln in n.body_content.splitlines()
                 if NUMBERED_STEP_RE.match(ln)]
        print(f"    - '{n.heading_title}' keeps {len(steps)} step line(s); "
              f"e.g. {steps[0][:60]!r}")


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def print_tree(tree, node_id=None, depth=0):
    node = tree.nodes[node_id or tree.root.node_id]
    if depth == 0:
        print(node.heading_title)
    else:
        print("    " * (depth - 1) + "└── " + node.heading_title)
    for child_id in node.child_node_ids:
        print_tree(tree, child_id, depth + 1)


def main():
    tree = DocumentTree()
    for path in INPUT_FILES:
        tree.add_file(path)

    print("=" * 70)
    print("PARSED DOCUMENT HIERARCHY")
    print("=" * 70)
    print_tree(tree)

    print("\n" + "=" * 70)
    print("HIERARCHY INTEGRITY VALIDATION")
    print("=" * 70)
    errors, visited = validate_tree(tree)
    print(f"Total nodes        : {len(tree.nodes)}")
    print(f"Reached from ROOT  : {visited}")
    if errors:
        print(f"INTEGRITY ERRORS   : {len(errors)}")
        for e in errors:
            print("  - " + e)
    else:
        print("INTEGRITY ERRORS   : 0  -> every parent/child link is consistent")

    check_def0006_and_steps(tree)


if __name__ == "__main__":
    main()
