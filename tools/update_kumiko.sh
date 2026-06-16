#!/usr/bin/env bash
# Re-vendor the official kumiko into pannello/kumiko_vendor/.
#
# We vendor kumiko (rather than fetch it at install time) so that
# `pip install pannello` and `pipx install pannello` produce a self-contained,
# offline, reproducible tool. kumiko is AGPL-3.0 and so is pannello, so
# redistributing it here is compliant. Only the runtime files are kept.
#
# Usage: tools/update_kumiko.sh [GIT_REF]
set -euo pipefail

REF="${1:-9d587ae}"   # pinned upstream commit (kumiko master @ 2024)
REPO="https://github.com/njean42/kumiko.git"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/pannello/kumiko_vendor"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --quiet "$REPO" "$TMP/kumiko"
git -C "$TMP/kumiko" checkout --quiet "$REF"
COMMIT="$(git -C "$TMP/kumiko" rev-parse HEAD)"

rm -rf "$DEST"
mkdir -p "$DEST"
# runtime files only: the library + its lib/ package + license + readme
cp "$TMP/kumiko/kumikolib.py" "$DEST/"
cp -r "$TMP/kumiko/lib" "$DEST/"
cp "$TMP/kumiko/LICENSE" "$DEST/"
cp "$TMP/kumiko/README.md" "$DEST/"
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} +

# Local patch: guard self.panels.remove() against double-remove. Upstream's
# merge/deoverlap iterates a set() of panels whose hash order varies per run, so
# it nondeterministically throws "list.remove(x): x not in list". (upstream bug)
sed -i -E 's/(^\s*)self\.panels\.remove\((p[0-9]*)\)/\1if \2 in self.panels: self.panels.remove(\2)/' \
    "$DEST/lib/page.py"

cat > "$DEST/VENDOR.md" <<EOF
# Vendored kumiko

Source:  $REPO
Commit:  $COMMIT
License: AGPL-3.0 (see LICENSE)

This is the official upstream kumiko, unmodified, trimmed to the files pannello
imports at runtime (kumikolib.py + lib/). Re-vendor with tools/update_kumiko.sh.
EOF

echo "vendored kumiko @ $COMMIT -> $DEST"
