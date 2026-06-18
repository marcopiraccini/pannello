'use strict'
"""pannello command-line interface."""

import os
import sys
import argparse

from . import __version__
from . import core


def _log(msg):
    print(msg, file=sys.stderr)


def _process(comic, args, label=''):
    if args.repack:
        try:
            comic, _ = core.repack(comic, out_dir=args.out_dir, dpi=args.dpi, log=_log)
        except Exception as e:
            _log(f'{label}repack failed: {comic}: {e}')
            return False
    rtl = True if args.rtl else (False if args.ltr else None)  # None = auto-detect
    try:
        st = core.generate(
            comic, rtl=rtl, jobs=args.jobs, fallback=args.fallback,
            model_path=args.model, model_conf=args.model_conf, out_dir=args.out_dir,
            limit=args.limit, preview=args.preview, review=args.review, dpi=args.dpi, log=_log)
    except Exception as e:
        _log(f'{label}error: {comic}: {e}')
        return False
    dirn = st['reading_direction'] + (' (ComicInfo)' if st['rtl_source'] == 'ComicInfo.xml' else '')
    _log(f'{label}{st["comic"]}: {st["pages"]} pages, {st["panels"]} panels, '
         f'{dirn}  ({st["seconds"]:.1f}s) -> {st["out"]}')
    if st['order_mismatch']:
        _log(f'{label}  WARNING: KOReader will read this archive out of order '
             f'(panels misalign) -- regenerate with --repack')
    if st['gray_hint'] and st['rtl_source'] == 'default':
        _log(f'{label}  note: looks black-and-white/manga; add --rtl if it reads right-to-left')
    _print_low_confidence(st['low_confidence'], st['pages'], label)
    if st['preview_dir']:
        _log(f'{label}  preview: {st["preview_sheets"]} sheet(s) -> {st["preview_dir"]}')
    if st.get('review_dir'):
        _log(f'{label}  review: {len(st["low_confidence"])} page(s) -> {st["review_dir"]}')
    return True


def _print_low_confidence(lc, total, label=''):
    """Table of low-confidence pages: reason + how each ended up."""
    if not lc:
        return
    fixed = sum(1 for x in lc if x['fixed'])
    _log(f'{label}  low-confidence: {len(lc)}/{total} pages '
         f'({fixed} fixed by model, {len(lc) - fixed} to review)')
    _log(f'{label}    page  reason     result')
    for x in lc[:20]:
        if x['fixed']:
            result = f'fixed -> {x["panels"]} panels (model)'
        elif x.get('fullpage'):
            result = 'full page'
        else:
            result = f'tiled -> {x["panels"]} panels'
        _log(f'{label}    {x["page"]:>4}  {x["reason"]:<10} {result}')
    if len(lc) > 20:
        _log(f'{label}    ... +{len(lc) - 20} more')


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='pannello',
        description='Detect comic panels and write KOReader panel-zoom JSON '
                    '(<comic>.json) using the official kumiko, with an optional model fallback.')
    direction = ap.add_mutually_exclusive_group()
    ap.add_argument('input',
                    help='a comic (cbz/cbr/cb7/cbt/pdf), a folder of comics '
                         '(one JSON per archive), or a folder of page images')
    ap.add_argument('--repack', action='store_true',
                    help='first normalize the comic to a CBZ with flat reading-order '
                         'page names (fixes books KOReader sorts wrong), then write JSON for it')
    direction.add_argument('--rtl', action='store_true',
                    help='force right-to-left reading order (manga)')
    direction.add_argument('--ltr', action='store_true',
                    help='force left-to-right reading order')
    ap.add_argument('--preview', action='store_true',
                    help='also write contact-sheet PNGs (<name>.preview/) with numbered '
                         'panel boxes for visual QA (green=kumiko, red=model)')
    ap.add_argument('--review', action='store_true',
                    help='write a focused contact sheet (<name>.review/) of just the '
                         'low-confidence pages, captioned with their reason')
    ap.add_argument('--fallback', choices=['auto', 'model', 'none'], default='auto',
                    help="model fallback for weak pages (kumiko found nothing / one "
                         "full-page box / crashed): 'auto' (default) uses the model if "
                         "the [model] extra is installed, else kumiko-only; 'model' "
                         "requires it; 'none' disables it")
    ap.add_argument('-o', '--out-dir',
                    help='write JSON files here (default: next to each comic)')
    ap.add_argument('-j', '--jobs', type=int, default=None,
                    help='parallel workers (default: CPU cores - 2)')
    ap.add_argument('--limit', type=int, help='only process the first N pages (testing)')
    ap.add_argument('--dpi', type=int, default=150,
                    help='resolution for rendering PDF pages to images (default: 150)')
    ap.add_argument('--model', default=None,
                    help="which model for --fallback: a preset ('general' default, "
                         "'manga'), a local .pt path, or a Hub repo id "
                         "('owner/name' or 'owner/name:weights.pt')")
    ap.add_argument('--model-conf', type=float, default=0.25, help='model confidence threshold')
    ap.add_argument('-V', '--version', action='version', version=f'pannello {__version__}')
    args = ap.parse_args(argv)

    # Specifying a model means "require it": surface the install hint if the
    # [model] extra is missing instead of silently degrading.
    if args.model is not None and args.fallback == 'auto':
        args.fallback = 'model'

    src = os.path.expanduser(args.input)
    if not os.path.exists(src):
        _log(f'error: not found: {src}')
        return 1

    # A single archive/pdf, a folder of archives (batch), or a folder of images.
    if os.path.isfile(src):
        return 0 if _process(src, args) else 1

    comics = core.find_comics(src)
    if comics:
        _log(f'batch: {len(comics)} comics in {src}')
        ok = 0
        for i, c in enumerate(comics, 1):
            ok += _process(c, args, label=f'[{i}/{len(comics)}] ')
        _log(f'done: {ok}/{len(comics)} comics')
        return 0 if ok else 1

    # No archives inside: treat the folder itself as one comic (folder of pages).
    return 0 if _process(src, args) else 1


if __name__ == '__main__':
    sys.exit(main())
