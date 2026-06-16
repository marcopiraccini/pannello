# Vendored kumiko

Source:  https://github.com/njean42/kumiko.git
Commit:  9d587ae9498bc84dfda06fc19c6ad89f421bec14
License: AGPL-3.0 (see LICENSE)

This is the official upstream kumiko, trimmed to the files pannello imports at
runtime (kumikolib.py + lib/). Re-vendor with tools/update_kumiko.sh.

Local patch: lib/page.py guards every `self.panels.remove(...)` with an
`if ... in self.panels` check. Upstream iterates a `set()` of panels whose hash
order varies per run, so it nondeterministically crashes with
"list.remove(x): x not in list". update_kumiko.sh re-applies this patch.
