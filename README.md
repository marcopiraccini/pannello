# pannello

Detect comic-book panels and write the per-page JSON used by KOReader
panel-zoom plugins. Point it at a `.cbz`/`.cbr` (or a whole folder) and it
produces a `<comic>.json` next to each file.

Panel detection uses the official [kumiko](https://github.com/njean42/kumiko)
(vendored), with an automatic CPU model fallback (when installed) for pages where
kumiko fails.

Licensed AGPL-3.0-or-later (see [Licensing](#licensing)).

## What it does

- Reads `.cbz .cbr .cb7 .cbt .pdf` archives, or a folder of page images.
- Detects panels per page, normalized to 0..1, stored in reading order.
- Writes JSON named after the comic, in the format the plugin reads:

```json
{
  "reading_direction": "ltr",
  "total_pages": 2,
  "pages": [
    {"page": 1, "image": "p001.jpg",
     "panels": [{"x": 0.04, "y": 0.01, "w": 0.92, "h": 0.19}]}
  ]
}
```

Temporary files (archive extraction) go to the system temp dir (`/tmp` on Linux)
and are cleaned up afterwards.

## Install

System tools (only what your archives need):

- `.cbr` -> `unrar`   (`apt install unrar`)
- `.cb7` -> `7z`      (`apt install p7zip-full`)
- `.pdf` -> `pdftoppm` (`apt install poppler-utils`)
- `.cbz` -> nothing extra

Then, from a checkout:

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

This installs the `pannello` command. The official kumiko is bundled, so there
is nothing else to install for the base detector.

### Model fallback (recommended)

kumiko handles most pages; a model cleans up the ones it fails on ("weak" pages:
no panels, one full-page box, or a kumiko crash). Once the `[model]` extra is
installed, pannello uses it **automatically** on weak pages -- no flag needed.
Without the extra, pannello runs kumiko-only and degrades quietly.

CPU-only, no GPU required:

```sh
.venv/bin/pip install -e '.[model]'      # torch + ultralytics + huggingface_hub
# if you have no CUDA GPU, install CPU torch first to avoid the big CUDA wheels:
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The default fallback model is `general` (Western/general comics). Weights
download on first use into the Hugging Face cache.

## Usage

```sh
pannello "My Comic.cbz"              # -> "My Comic.json" next to it
pannello /path/to/library            # batch: one JSON per cbr/cbz found (recursive)
pannello comic.cbz --rtl             # manga (right-to-left reading order)
pannello library/ -o out/            # write all JSON into out/
pannello manga.cbz --rtl --model manga   # manga: right-to-left + manga model on weak pages
pannello comic.cbr --preview         # also write contact sheets to inspect the panels
pannello --help
```

Input can be a single comic, a folder of comics (each gets its own JSON), or a
folder of loose page images (treated as one comic named after the folder).

### Fixing wrongly-ordered archives (`--repack`)

The KOReader panel plugin matches panels to pages by **page index**, so
pannello's page order must equal KOReader's. KOReader orders archive pages by the
**raw byte order** of their entry paths. Most CBZ/CBR files are flat and
zero-padded, so this is fine. But an archive with chapter subfolders or odd names
can sort differently than its intended reading order -- e.g. a `From Hell - Appendice/`
folder byte-sorts before `From Hell 00/` (because `-` < `0`), so KOReader shows
43 appendix pages first. The book then reads out of order *and* every panel set
lands on the wrong page.

`--repack` fixes this by rewriting the comic into a CBZ with flat, zero-padded
names (`0001.jpg`, `0002.jpg`, ...) in reading order, then writing the JSON for
it. Byte order now equals reading order, so KOReader reads it correctly and
panels line up:

    pannello --repack "From Hell.cbr"      # -> "From Hell.cbz" + "From Hell.json"

Put **both** files on the device and open the `.cbz` (not the original). Repacking
is lossless (images are copied, only renamed). Archives that are already flat and
ordered (most comics) don't need it.

Key flags: `--rtl`/`--ltr` (force reading order), `--preview`, `-o/--out-dir`,
`-j/--jobs` (default cores-2), `--limit N` (first N pages, for testing),
`--fallback {auto,model,none}`, `--model`, `--model-conf`, `-V/--version`.

### Reading direction

pannello auto-detects reading direction: it reads the `<Manga>` field of a
`ComicInfo.xml` inside the archive if present (`rtl` for manga), otherwise
defaults to `ltr`. Force it with `--rtl` or `--ltr`. When there's no metadata and
the pages look black-and-white (manga-like), it prints a hint suggesting `--rtl`
-- it never auto-flips on color alone (that would wrongly flip B&W Western books).

### Preview (`--preview`)

Writes contact-sheet PNGs to `<name>.preview/` with every panel boxed and
numbered in reading order, so you can verify detection (and that `--rtl` is
right -- panel 1 should be top-right for manga) on your computer instead of
round-tripping to the device. Green boxes = kumiko, red = model-rescued.
Sheets render in parallel with a `sheet k/N` progress line; use `--limit N` to
preview only the first N pages of a big book.

### Ordering check

Every run checks whether KOReader's byte-sort page order matches pannello's, and
prints a `WARNING: KOReader will read this archive out of order` with a
`--repack` suggestion when they differ -- so the silent misalignment described
below is caught automatically.

### Choosing the model

The model runs on weak pages automatically (when installed). `--model` picks
which one:

```sh
pannello comic.cbz                         # default: general (Western) model, automatic
pannello manga.cbz --rtl --model manga     # manga model (manga109), for manga
pannello comic.cbz --model ./my.pt         # a local .pt file
pannello comic.cbz --model owner/name:weights.pt   # any HF YOLO repo
```

`general` (default) detects panels on Western comics. `manga` only fires on
manga-style ruled panels (it finds nothing on Western/European color art), so use
it for manga. Passing `--model` *requires* the `[model]` extra (errors with an
install hint if missing); the bare default degrades to kumiko-only.

`--fallback`: `auto` (default) uses the model on weak pages if installed;
`model` requires it; `none` disables it (kumiko only).

## How it works

1. Extract the archive to a temp dir, list pages in natural order.
2. Run kumiko per page in parallel; normalize panels to 0..1.
3. Flag "weak" pages (no panels, one near-full-page box, or a kumiko crash).
4. Re-run weak pages through the model (automatically when the `[model]` extra is
   installed; `--fallback none` to skip), replacing kumiko's result only when the
   model finds a better segmentation (tagged `"source":"model"`).

## Benchmark

`benchmark/` compares kumiko vs the model against human-labeled ground truth
(true number of navigable panels per page). Metric: mean absolute count error
(MAE, lower is better) and exact-count matches.

Reproduce the free part (CC-BY Pepper & Carrot pages):

```sh
cd benchmark
python3 fetch_pc.py
../.venv/bin/python run_benchmark.py
```

| set | pages | kumiko MAE | model MAE | kumiko exact | model exact |
|---|---|---|---|---|---|
| Pepper & Carrot (CC-BY, reproducible) | 6 | 2.17 | 2.00 | 1/6 | 3/6 |
| From Hell (private, numbers only) | 12 | 0.58 | 1.75 | 9/12 | 2/12 |
| Providence Vol.1 (private, numbers only) | 12 | 0.83 | 1.50 | 7/12 | 4/12 |

Only the Pepper & Carrot set is redistributable (CC-BY), so only it is
reproducible from this repo. The From Hell / Providence numbers come from
copyrighted books (owned in print); pages are not distributed, only the counts.

Takeaways: kumiko is the better general engine, including on the "complex"
books, because they use clear panel borders (grids, splashes) that kumiko
handles well. The model tends to over-segment grids (From Hell) but does win
specific pages where kumiko under-segments low-contrast colored gutters
(e.g. Providence p.091: true 7, kumiko 4, model 7). So the model is a targeted
fallback, not a general replacement. Panel count is a coarse proxy and labels
are from a single human, so treat small gaps (Pepper & Carrot) as a wash.

## Licensing

pannello is **AGPL-3.0-or-later**. It must be, because it bundles kumiko
(AGPL-3.0) and the optional model path depends on ultralytics (AGPL-3.0).

Components and their licenses (full detail in [NOTICE](NOTICE)):

| Component | Bundled? | License |
|---|---|---|
| kumiko (njean42/kumiko) | yes, vendored | AGPL-3.0 |
| mosesb/best-comic-panel-detection (default model) | no, downloaded at runtime | Apache-2.0 |
| ultralytics (model inference, `[model]` extra) | no, optional dep | AGPL-3.0 |
| OpenCV, NumPy, Requests | no, deps | Apache-2.0 / BSD / Apache-2.0 |
| Pepper & Carrot (benchmark pages) | no, downloaded by benchmark | CC-BY 4.0 |

### Inspiration and the "did you copy code?" question

pannello was inspired by the KOReader **panelreader.koplugin /
panelzoom_integration.koplugin** (AGPL-3.0), and it emits the same on-disk JSON
schema so the plugin can read its output.

No source code was copied from that plugin. pannello is a from-scratch
reimplementation. The only thing shared is the JSON format, which is a data
interface (not copyrightable expression). Where the plugin bundled a fork of
kumiko, pannello vendors the official upstream kumiko instead.

### Why kumiko is vendored (not downloaded at setup)

`pannello/kumiko_vendor/` is the official kumiko, pinned to one commit
(see `kumiko_vendor/VENDOR.md`), trimmed to the runtime files. It is committed
on purpose: it keeps `pip install` / `pipx install` self-contained, offline, and
reproducible (kumiko is not on PyPI and has no post-install hook). Since pannello
is already AGPL-3.0, redistributing AGPL-3.0 kumiko here is compliant. Update it
with `tools/update_kumiko.sh [git-ref]`.
