#!/usr/bin/env bash
# Fetch the pinned public corpus (PX4 User Guide subset, CC BY 4.0), verify
# every file against the committed sha256 manifest, and convert it to
# heading-structured markdown in corpus/ (gitignored).
#
#   ./scripts/fetch_corpus.sh            # fetch (if needed) + verify + convert
#   ./scripts/fetch_corpus.sh --force    # re-download even if a verified copy exists
#
# Requirements: bash, curl, python3 (stdlib only), and shasum or sha256sum.
# Downloaded files are treated as untrusted data: they go into a fresh empty
# directory, nothing in them is executed, and the converter runs as `python3 -I`.
set -euo pipefail

PX4_REPO="PX4/PX4-Autopilot"
PX4_TAG="v1.17.0"
PX4_COMMIT="d6f12ad1c4f70ad3230afd7d86e971421e02fef4"   # commit the tag points to
MANIFEST="scripts/px4_docs_v1.17.0.sha256"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_DIR="corpus/raw/px4-${PX4_COMMIT}"
OUT_DIR="corpus"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if command -v sha256sum >/dev/null 2>&1; then
  SHA_CHECK=(sha256sum --check --quiet --strict)
elif command -v shasum >/dev/null 2>&1; then
  SHA_CHECK=(shasum -a 256 --check --quiet --strict)
else
  echo "error: need sha256sum or shasum" >&2; exit 1
fi

verify() {  # verify <dir>: manifest hashes match and no extra files exist
  local dir="$1"
  ( cd "$dir" && "${SHA_CHECK[@]}" "$REPO_ROOT/$MANIFEST" ) || return 1
  local expected actual
  expected="$(wc -l < "$MANIFEST" | tr -d ' ')"
  actual="$(find "$dir" -type f | wc -l | tr -d ' ')"
  if [ "$expected" != "$actual" ]; then
    echo "error: $dir has $actual files, manifest lists $expected" >&2
    return 1
  fi
}

if [ "$FORCE" = 0 ] && [ -d "$RAW_DIR" ] && verify "$RAW_DIR" 2>/dev/null; then
  echo "== verified copy already present: $RAW_DIR"
else
  echo "== downloading PX4 User Guide subset @ ${PX4_TAG} (${PX4_COMMIT})"
  mkdir -p corpus/raw
  TMP_DIR="$(mktemp -d corpus/raw/.download.XXXXXX)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  CFG="$(mktemp)"
  trap 'rm -rf "$TMP_DIR" "$CFG"' EXIT
  while read -r _sha path; do
    case "$path" in
      docs/*) ;;                                  # manifest paths are fixed
      *) echo "error: bad manifest path $path" >&2; exit 1 ;;
    esac
    case "$path" in *..*) echo "error: bad manifest path $path" >&2; exit 1 ;; esac
    printf 'url = "https://raw.githubusercontent.com/%s/%s/%s"\noutput = "%s/%s"\n' \
      "$PX4_REPO" "$PX4_COMMIT" "$path" "$TMP_DIR" "$path" >> "$CFG"
  done < "$MANIFEST"
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
       --retry 3 --create-dirs --parallel --config "$CFG"
  echo "== verifying sha256 of $(wc -l < "$MANIFEST" | tr -d ' ') files"
  verify "$TMP_DIR"
  rm -rf "$RAW_DIR"
  mkdir -p "$(dirname "$RAW_DIR")"
  mv "$TMP_DIR" "$RAW_DIR"
  echo "== verified: $RAW_DIR"
fi

echo "== converting to heading-structured markdown in $OUT_DIR/"
python3 -I prepare_public_corpus.py --src "$RAW_DIR" --out "$OUT_DIR" --commit "$PX4_COMMIT"
echo "== done. Next: python3 check_corpus_tree.py"
