'use strict'
"""
pannello.core - panel detection and JSON generation.

Detects comic panels with the official kumiko (vendored) and writes the JSON
format consumed by KOReader panel-zoom plugins:

    {"reading_direction": "ltr"|"rtl", "total_pages": N,
     "pages": [{"page": 1, "image": "p001.jpg",
                "panels": [{"x":..,"y":..,"w":..,"h":..}]}]}

Coordinates are normalized to 0..1, panels are stored in reading order.
Temporary extraction uses the system temp dir (/tmp on Linux).
"""

import os
import re
import sys
import json
import time
import shutil
import zipfile
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

VENDOR = Path(__file__).resolve().parent / 'kumiko_vendor'

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
ARCHIVE_EXTS = ('.cbz', '.cbr', '.cb7', '.cbt', '.pdf')

# A page is "weak" (kumiko likely failed) when it has no panels, or a single
# panel covering at least this fraction of the page area.
WEAK_FULLPAGE_AREA = 0.90


def natural_key(path):
    parts = re.split(r'(\d+)', str(path).lower())
    return [int(p) if p.isdigit() else p for p in parts]


def detect_archive_type(path):
    """Detect type from magic bytes (extensions lie: .cbr is often a zip)."""
    try:
        out = subprocess.run(['file', '-b', str(path)], capture_output=True,
                             text=True, check=True).stdout.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ''
    # Order matters: the broad 'zip' check must come LAST -- other formats mention
    # "zip" in their description (e.g. a PDF "(zip deflate encoded)"), and 'zip'
    # must also catch "Zip multi-volume archive data" (mislabeled .cbr files).
    if 'rar archive' in out:
        return 'rar'
    if '7-zip' in out:
        return '7z'
    if 'pdf document' in out:
        return 'pdf'
    if 'posix tar' in out or 'tar archive' in out:
        return 'tar'
    if 'zip' in out and 'gzip' not in out:
        return 'zip'
    return {'.cbz': 'zip', '.cbr': 'rar', '.cb7': '7z', '.cbt': 'tar',
            '.pdf': 'pdf'}.get(path.suffix.lower())


def extract_archive(path, dest, dpi=150):
    """Extract a comic archive into dest. Returns the detected type. PDFs are
    rendered to JPEGs at `dpi` (pdftoppm)."""
    kind = detect_archive_type(path)
    try:
        if kind == 'zip':
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest)
        elif kind == 'rar':
            subprocess.run(['unrar', 'x', '-o+', '-y', '-inul', str(path), str(dest) + os.sep],
                           check=True)
        elif kind == '7z':
            subprocess.run(['7z', 'x', '-y', f'-o{dest}', str(path)],
                           check=True, stdout=subprocess.DEVNULL)
        elif kind == 'tar':
            subprocess.run(['tar', '-xf', str(path), '-C', str(dest)], check=True)
        elif kind == 'pdf':
            subprocess.run(['pdftoppm', '-jpeg', '-r', str(dpi), str(path),
                            str(Path(dest) / 'page')], check=True)
        else:
            raise ValueError(f'unsupported archive: {path} (detected: {kind})')
    except FileNotFoundError as e:
        tool = {
            'rar': 'unrar',
            '7z': '7z',
            'tar': 'tar',
            'pdf': 'pdftoppm',
        }.get(kind, kind or 'extractor')
        raise RuntimeError(f'missing required tool {tool!r} to extract {path.name}') from e
    return kind


def list_pages(root):
    """All image files under root, naturally sorted, skipping macOS junk."""
    files = [p for p in Path(root).rglob('*')
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS
             and '__MACOSX' not in p.parts and not p.name.startswith('._')]
    files.sort(key=natural_key)
    return files


def kumiko_one(image_path, rtl):
    """Run official kumiko on a single image (in-process). Returns its info dict."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from kumikolib import Kumiko
    k = Kumiko({'debug': False, 'progress': False, 'rtl': rtl,
                'min_panel_size_ratio': None, 'panel_expansion': True})
    k.parse_image(str(image_path))
    return k.get_infos()[0]


def normalize_panels(info):
    """Convert kumiko pixel [x,y,w,h] panels to normalized {x,y,w,h} dicts."""
    w, h = info['size'][0], info['size'][1]
    if not w or not h:
        return []
    out = []
    for x, y, pw, ph in info['panels']:
        out.append({
            'x': round(max(0.0, min(1.0, x / w)), 4),
            'y': round(max(0.0, min(1.0, y / h)), 4),
            'w': round(max(0.0, min(1.0, pw / w)), 4),
            'h': round(max(0.0, min(1.0, ph / h)), 4),
        })
    return out


def is_weak(panels):
    """True when kumiko likely failed: no panels, or one near-full-page panel."""
    if not panels:
        return True
    if len(panels) == 1 and panels[0]['w'] * panels[0]['h'] >= WEAK_FULLPAGE_AREA:
        return True
    return False


def detect_anomalies(panels):
    """High-precision flags for clearly-wrong panels (calibrated to NOT fire on
    clean grids): a sliver (extreme aspect ratio), a tiny noise panel, or two
    panels overlapping (a kumiko de-overlap bug). Returns a sorted list of tags.

    Note: this does NOT catch borderless grid-over-segmentation (normal-aspect
    cells) -- no cheap high-precision signal exists for that.
    """
    flags = set()
    for p in panels:
        w, h = p['w'], p['h']
        if w > 0 and h > 0 and max(w / h, h / w) >= 10:
            flags.add('sliver')
        if w * h < 0.005:
            flags.add('tiny')
    for i, a in enumerate(panels):
        for b in panels[i + 1:]:
            ix = max(0.0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x']))
            iy = max(0.0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
            if ix * iy > 0.05 * min(a['w'] * a['h'], b['w'] * b['h']):
                flags.add('overlap')
    return sorted(flags)


def page_issue(panels, crashed=False):
    """Unified 'kumiko probably got this page wrong' reason, or None.

    One concept ("low confidence") with a reason: under-detection
    (crash / empty / full_page) or an anomaly (sliver / overlap / tiny). The model
    is tried on all of them; the reason just tells the user what looked off.
    """
    if crashed:
        return 'crash'
    if not panels:
        return 'empty'
    if len(panels) == 1:
        # A lone panel: the whole page should be that one panel. If it doesn't
        # cover the page (kumiko boxed just an illustration, leaving a title/credits
        # strip), the model gets a chance to find a real split; otherwise it
        # collapses to a true full-page so no strip is left unreachable.
        return 'full_page'
    anomalies = detect_anomalies(panels)
    if anomalies:
        return anomalies[0]
    if has_hole(panels):
        return 'hole'  # a region of the page is uncovered -> let the model try
    return None


def _worker(args):
    idx, image_path, rtl = args
    try:
        panels = normalize_panels(kumiko_one(image_path, rtl))
        return idx, panels, is_weak(panels), None
    except Exception as e:  # keep going on a bad page
        return idx, [], True, f'{type(e).__name__}: {e}'


def detect_pages(pages, rtl, jobs, progress=None):
    """Detect panels for every page in parallel. Returns (pages_data, weak, errors)."""
    pages_data = [None] * len(pages)
    weak, errors = [], []
    tasks = [(i, p, rtl) for i, p in enumerate(pages)]
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            idx, panels, weak_flag, err = fut.result()
            pages_data[idx] = {'page': idx + 1, 'image': pages[idx].name, 'panels': panels}
            if weak_flag:
                weak.append(idx + 1)
            if err:
                errors.append((idx + 1, err))
            done += 1
            if progress:
                progress(done, len(pages))
    weak.sort()
    return pages_data, weak, errors


def detect_pages_model(pages, rtl, model_path, conf, progress=None, engine=None):
    """Engine-primary detection: run the engine (YOLO model by default, or Magi)
    on EVERY page, skipping kumiko.

    Sequential (the engine runs in this process, not the worker pool). Returns
    pages_data like detect_pages but with NO 'source' marker, so the coverage
    guarantee treats the engine's boxes as raw detections (lone panels collapse to
    full page, multi-panel pages get tiled). The YOLO model under-detects on many
    comics; Magi is far more accurate (see --thorough).
    """
    if engine is None:
        from . import model as engine
    engine.load_model(model_path)  # fail fast if the extra is missing
    pages_data, total = [], len(pages)
    for i, p in enumerate(pages):
        try:
            boxes, size = engine.detect_panels(str(p), rtl=rtl, model_path=model_path, conf=conf)
            panels = engine.normalize(boxes, size)
        except Exception:
            panels = []
        pages_data.append({'page': i + 1, 'image': pages[i].name, 'panels': panels})
        if progress:
            progress(i + 1, total)
    return pages_data


def deoverlap(panels):
    """Clip overlapping panels apart so the output is strictly disjoint. Each
    overlapping pair is split along the thinner overlap dimension at the overlap
    midpoint (the two panels then meet edge-to-edge -- no overlap, no gap).
    A reader must never get overlapping zoom regions.
    """
    p = [dict(q) for q in panels]
    for _ in range(50):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                a, b = p[i], p[j]
                ox = min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x'])
                oy = min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y'])
                if ox <= 1e-4 or oy <= 1e-4:
                    continue
                if ox <= oy:  # thinner overlap is horizontal -> split in x
                    mid = (max(a['x'], b['x']) + min(a['x'] + a['w'], b['x'] + b['w'])) / 2
                    left, right = (a, b) if a['x'] <= b['x'] else (b, a)
                    left['w'] = mid - left['x']
                    right['w'] = right['x'] + right['w'] - mid
                    right['x'] = mid
                else:  # split in y
                    mid = (max(a['y'], b['y']) + min(a['y'] + a['h'], b['y'] + b['h'])) / 2
                    top, bot = (a, b) if a['y'] <= b['y'] else (b, a)
                    top['h'] = mid - top['y']
                    bot['h'] = bot['y'] + bot['h'] - mid
                    bot['y'] = mid
                moved = True
        if not moved:
            break
    return [{k: round(v, 4) for k, v in q.items()} for q in p]


def content_bbox(panels):
    """Bounding box (x0, y0, x1, y1) of all panels -- the comic content area
    (page margins lie outside it)."""
    return (min(p['x'] for p in panels), min(p['y'] for p in panels),
            max(p['x'] + p['w'] for p in panels), max(p['y'] + p['h'] for p in panels))


def covered_fraction(panels):
    """Fraction of the content bbox covered by panels (coarse raster; overlap-safe)."""
    if len(panels) < 2:
        return 1.0
    bx0, by0, bx1, by1 = content_bbox(panels)
    bw, bh = bx1 - bx0, by1 - by0
    if bw <= 0 or bh <= 0:
        return 1.0
    N = 64
    grid = np.zeros((N, N), dtype=bool)
    for p in panels:
        cx0 = max(0, int((p['x'] - bx0) / bw * N))
        cx1 = min(N, int(round((p['x'] + p['w'] - bx0) / bw * N)))
        cy0 = max(0, int((p['y'] - by0) / bh * N))
        cy1 = min(N, int(round((p['y'] + p['h'] - by0) / bh * N)))
        grid[cy0:cy1, cx0:cx1] = True
    return float(grid.mean())


def has_hole(panels):
    """True if a multi-panel page leaves a sizeable uncovered region inside its
    content bbox (a real hole, not thin gutters)."""
    return len(panels) >= 2 and covered_fraction(panels) < 0.85


def expand_to_page(panels):
    """Grow panels to meet their neighbours / the content-bbox edges so they tile
    the comic area with no gaps (what kumiko does, but the model doesn't), leaving
    page margins out. A missed region gets absorbed into an adjacent, larger panel
    -- reachable, never lost. Facing panels meet at the gutter midpoint -> no new
    overlaps.
    """
    if len(panels) < 2:
        return panels
    eps = 0.01
    bx0, by0, bx1, by1 = content_bbox(panels)
    b = [[p['x'], p['y'], p['x'] + p['w'], p['y'] + p['h']] for p in panels]
    out = []
    for i, (x0, y0, x1, y1) in enumerate(b):
        rights = [o[0] for j, o in enumerate(b) if j != i and o[0] >= x1 - eps and o[3] > y0 and o[1] < y1]
        lefts = [o[2] for j, o in enumerate(b) if j != i and o[2] <= x0 + eps and o[3] > y0 and o[1] < y1]
        downs = [o[1] for j, o in enumerate(b) if j != i and o[1] >= y1 - eps and o[2] > x0 and o[0] < x1]
        ups = [o[3] for j, o in enumerate(b) if j != i and o[3] <= y0 + eps and o[2] > x0 and o[0] < x1]
        nx0 = (x0 + max(lefts)) / 2 if lefts else bx0
        nx1 = (x1 + min(rights)) / 2 if rights else bx1
        ny0 = (y0 + max(ups)) / 2 if ups else by0
        ny1 = (y1 + min(downs)) / 2 if downs else by1
        out.append({'x': round(nx0, 4), 'y': round(ny0, 4),
                    'w': round(nx1 - nx0, 4), 'h': round(ny1 - ny0, 4)})
    return out


def _run_fallback(pages, pages_data, weak, rtl, model_path, conf, log, engine=None):
    """Re-detect weak pages with the fallback engine (YOLO model by default, or
    Magi). Returns count of pages rescued."""
    if engine is None:
        from . import model as engine
    engine.load_model(model_path)
    rescued = 0
    for pn in weak:
        idx = pn - 1
        try:
            boxes, size = engine.detect_panels(
                str(pages[idx]), rtl=rtl, model_path=model_path, conf=conf)
        except Exception as e:
            log(f'  fallback error page {pn}: {e}')
            continue
        panels = engine.normalize(boxes, size)
        # A big/irregular panel's bounding box often overlaps its smaller neighbours
        # (e.g. a splash figure beside a column of insets) -- that's a legit layout,
        # not a bad detection. Clip overlaps apart first (as the coverage guarantee
        # would anyway) BEFORE judging, or we'd wrongly reject good segmentations.
        if len(panels) >= 2 and detect_anomalies(panels):
            panels = deoverlap(panels)
        # Apply the engine only when its result is CLEAN (>=2 panels, no leftover
        # anomalies) AND it covers the same content extent as kumiko (its bounding box
        # isn't smaller -- otherwise it dropped a whole region, e.g. a column, which
        # would be lost). Final tiling fills any internal gaps afterwards.
        kp = pages_data[idx]['panels']
        if len(panels) >= 2 and not detect_anomalies(panels) and not _loses_extent(panels, kp):
            pages_data[idx]['panels'] = panels
            pages_data[idx]['source'] = 'model'
            rescued += 1
    return rescued


def _loses_extent(model_panels, kumiko_panels):
    """True if the model's content area is meaningfully smaller than kumiko's
    (it dropped a region kumiko had). Compares bounding-box areas."""
    if not kumiko_panels:
        return False
    mx0, my0, mx1, my1 = content_bbox(model_panels)
    kx0, ky0, kx1, ky1 = content_bbox(kumiko_panels)
    m = (mx1 - mx0) * (my1 - my0)
    k = (kx1 - kx0) * (ky1 - ky0)
    return k > 0 and m < 0.9 * k


def generate(comic, rtl=None, jobs=None, fallback='auto', model_path=None,
             model_conf=0.25, out_dir=None, limit=None, preview=False, review=False,
             dpi=150, detector='kumiko', magi=False, thorough=False, log=None):
    """Generate panel JSON for one comic (archive or folder of images).

    Returns a stats dict. Writes <name>.json next to the comic, or into out_dir.
    """
    log = log or (lambda *_: None)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    comic = Path(comic).expanduser()
    if limit is not None and limit < 0:
        raise ValueError('--limit must be >= 0')

    if comic.is_dir():
        name = comic.name
        root = comic
        tmp = None
    else:
        name = comic.stem
        tmp = tempfile.mkdtemp(prefix='pannello-')  # system temp (/tmp on Linux)
        extract_archive(comic, tmp, dpi=dpi)
        root = tmp

    try:
        all_pages = list_pages(root)
        if limit is not None:
            pages = all_pages[:max(0, limit)]
        else:
            pages = all_pages
        if not pages and limit is None:
            raise ValueError(f'no page images found in {comic}')

        # Reading direction: explicit flag wins; else ComicInfo.xml; else default
        # ltr (grayscale only hints -- never auto-flips, to avoid flipping B&W
        # Western comics).
        rtl_source = 'flag'
        gray_hint = False
        if rtl is None:
            if pages:
                detected = detect_reading_direction(root)
                if detected is not None:
                    rtl = detected == 'rtl'
                    rtl_source = 'ComicInfo.xml'
                else:
                    rtl = False
                    rtl_source = 'default'
                    gray_hint = looks_grayscale(pages)
            else:
                rtl = False
                rtl_source = 'default'

        # KOReader reads archives in byte-sort order; warn if that differs from
        # pannello's order (panels would misalign) -- only relevant for archives.
        order_mismatch = bool(pages) and (not comic.is_dir()) and koreader_order_differs(pages, root)

        # --thorough makes Magi the SOLE detector (kumiko disabled): Magi runs on
        # every page and its result is authoritative -- a <2-panel result is a
        # genuine splash/full page, not a failure to be overruled by kumiko.
        magi_primary = bool(thorough and magi)

        t0 = time.time()
        pages_data, errors = [], []
        if pages:
            def _dp(done, total):
                print(f'\r  detecting page {done}/{total}',
                      end='\n' if done == total else '', file=sys.stderr, flush=True)
            if detector == 'model' or magi_primary:
                eng = None
                if magi_primary:
                    from . import magi as eng
                    log('  detector: Magi on every page (kumiko disabled)')
                pages_data = detect_pages_model(pages, rtl, model_path, model_conf,
                                                progress=_dp, engine=eng)
            else:
                pages_data, _weak, errors = detect_pages(pages, rtl, jobs, progress=_dp)

        # Unified "low-confidence" classification: any page the detector probably
        # got wrong (under-detection: crash/empty/full_page; or anomaly: sliver/
        # overlap/tiny).
        crashed = {pn for pn, _ in errors}
        issues = {}
        for pd in pages_data:
            r = page_issue(pd['panels'], pd['page'] in crashed)
            if r:
                issues[pd['page']] = r

        # kumiko-primary: try to fix EVERY low-confidence page with the fallback
        # engine (replaces kumiko only when the engine returns a cleaner multi-panel
        # result). 'none' disables it, 'auto' degrades quietly if the [model] extra
        # is missing, 'model' requires it. Skipped when an engine is already the
        # primary detector (--detector model, or --thorough) -- the coverage
        # guarantee handles the rest.
        to_fix = ([] if (fallback == 'none' or detector == 'model' or magi_primary)
                  else sorted(issues))
        if to_fix:
            engine = None
            if magi:
                from . import magi as engine
                log('  fallback engine: Magi (non-commercial model; opt-in)')
            try:
                _run_fallback(pages, pages_data, to_fix, rtl, model_path, model_conf,
                              log, engine=engine)
            except ImportError:
                if magi or fallback == 'model':
                    raise  # user explicitly asked for an engine; surface the install hint
                elif any(r == 'crash' for r in issues.values()):
                    n = sum(1 for r in issues.values() if r == 'crash')
                    log(f'  {n} page(s) kumiko could not parse; install '
                        f'"pannello[model]" to auto-fill them with the model')

        # Any page the model couldn't rescue into a clean multi-panel grid:
        # collapse to a SINGLE full-page panel (cover the whole page -- never
        # leave a strip unreachable, and never ship wrong/partial boundaries).
        for pn, reason in issues.items():
            pd = pages_data[pn - 1]
            if not pd['panels'] or (reason == 'full_page' and pd.get('source') != 'model'):
                pd['panels'] = [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}]
                pd['source'] = 'fullpage'

        # Coverage guarantee: every multi-panel page must tile its content area
        # (panels cover the whole page minus margins -- no holes, no gutter gaps).
        # Expand panels to fill, then clip any overlap that creates.
        for pd in pages_data:
            if len(pd['panels']) >= 2:
                tiled = expand_to_page(pd['panels'])
                if detect_anomalies(tiled):
                    tiled = deoverlap(tiled)
                pd['panels'] = tiled

        # Safety net: the shipped result must be EITHER a clean tiling OR the whole
        # page. Any page still anomalous, or whose panels cover too little of the
        # page (a detector missed a big region), collapses to one full-page panel.
        for pd in pages_data:
            ps = pd['panels']
            if len(ps) >= 2 and (detect_anomalies(ps) or sum(p['w'] * p['h'] for p in ps) < 0.7):
                pd['panels'] = [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}]
                pd['source'] = 'fullpage'
                issues.setdefault(pd['page'], 'coverage')

        # Final report: each low-confidence page, its reason, and how it ended up.
        # NOT written into the panel JSON (that stays the KOReader contract).
        low_confidence = []
        for pn, reason in sorted(issues.items()):
            pd = pages_data[pn - 1]
            low_confidence.append({
                'page': pn, 'reason': reason,
                'fixed': pd.get('source') == 'model',
                'fullpage': pd.get('source') == 'fullpage',
                'panels': len(pd['panels']),
            })

        pages_out = [{'page': pd['page'], 'image': pd['image'], 'panels': pd['panels']}
                     for pd in pages_data]

        result = {
            'reading_direction': 'rtl' if rtl else 'ltr',
            'total_pages': len(pages_data),
            'pages': pages_out,
        }
        dest_dir = Path(out_dir) if out_dir else comic.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f'{name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        preview_dir = preview_sheets = None
        if preview:
            def _pp(done, total):
                print(f'\r  preview: sheet {done}/{total}',
                      end='\n' if done == total else '', file=sys.stderr, flush=True)
            preview_dir, preview_sheets = render_preview(
                pages, pages_data, dest_dir, name, limit, jobs, _pp)

        # --review: a focused contact sheet of just the low-confidence pages,
        # each captioned with its reason (+ "fixed" when the model replaced it).
        review_dir = None
        if review and low_confidence:
            idxs = [x['page'] - 1 for x in low_confidence]
            notes = {x['page'] - 1: x['reason'] + (' fixed' if x['fixed'] else '')
                     for x in low_confidence}
            review_dir, _ = render_preview(pages, pages_data, dest_dir, name,
                                           jobs=jobs, indices=idxs, notes=notes, suffix='review')

        return {
            'comic': name, 'out': out_path, 'pages': len(pages_data),
            'panels': sum(len(p['panels']) for p in pages_data),
            'low_confidence': low_confidence, 'errors': errors,
            'reading_direction': 'rtl' if rtl else 'ltr', 'rtl_source': rtl_source,
            'gray_hint': gray_hint, 'order_mismatch': order_mismatch,
            'preview_dir': preview_dir, 'preview_sheets': preview_sheets,
            'review_dir': review_dir,
            'seconds': time.time() - t0,
        }
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def koreader_order_differs(pages, root):
    """True if KOReader's page order would differ from pannello's reading order.

    KOReader sorts archive entries by raw byte order of their paths; pannello uses
    a natural sort. When they disagree, panel indices misalign in KOReader and the
    book may read out of order -- the fix is --repack.
    """
    rels = [str(p.relative_to(root)) for p in pages]
    return rels != sorted(rels)


def detect_reading_direction(root):
    """Return 'rtl' / 'ltr' from a ComicInfo.xml <Manga> field, or None if unknown."""
    import xml.etree.ElementTree as ET
    for ci in Path(root).rglob('*'):
        if ci.is_file() and ci.name.lower() == 'comicinfo.xml':
            try:
                for el in ET.parse(ci).getroot().iter():
                    if el.tag.split('}')[-1].lower() == 'manga':
                        v = (el.text or '').strip().lower()
                        if 'righttoleft' in v or v == 'yes':
                            return 'rtl'
                        if v in ('no', 'unknown', ''):
                            return 'ltr'
            except Exception:
                pass
            return None
    return None


def looks_grayscale(pages, sample=5):
    """Heuristic: are sampled pages essentially black-and-white (manga-like)?"""
    try:
        from PIL import Image
    except ImportError:
        return False
    if not pages:
        return False
    idxs = sorted(set(min(len(pages) - 1, int((i + 1) * len(pages) / (sample + 1)))
                      for i in range(sample)))
    gray = n = 0
    for i in idxs:
        try:
            im = Image.open(pages[i]).convert('RGB').resize((48, 48))
            px = list(im.getdata())
            chroma = sum(max(p) - min(p) for p in px) / len(px)
            n += 1
            gray += chroma < 12
        except Exception:
            pass
    return n > 0 and gray / n >= 0.8


_PREVIEW_COLS, _PREVIEW_ROWS, _PREVIEW_CW, _PREVIEW_CH = 4, 5, 360, 500


def _render_cell(args):
    """Render one page to a labelled cell thumbnail (runs in a worker thread)."""
    from PIL import Image, ImageDraw
    idx, path, panels, model = args
    im = Image.open(path).convert('RGB')
    W, H = im.size
    d = ImageDraw.Draw(im)
    col = (220, 0, 0) if model else (0, 140, 0)
    for pi, p in enumerate(panels, 1):
        x0, y0 = p['x'] * W, p['y'] * H
        d.rectangle([x0, y0, (p['x'] + p['w']) * W, (p['y'] + p['h']) * H],
                    outline=col, width=max(3, W // 250))
        d.text((x0 + 5, y0 + 3), str(pi), fill=col)
    im.thumbnail((_PREVIEW_CW - 8, _PREVIEW_CH - 24))
    return idx, im, model


def render_preview(pages, pages_data, dest_dir, name, limit=None, jobs=None,
                   progress=None, indices=None, notes=None, suffix='preview'):
    """Write contact-sheet PNGs with numbered panel boxes for visual QA.

    Boxes are numbered in reading order (so RTL is verifiable: panel 1 sits
    top-right for manga). Green = kumiko, red = model-rescued. Pages render in
    parallel across threads (PIL decode/resize release the GIL).

    `indices` renders only those page indices (e.g. the low-confidence ones for
    --review); `notes` is an {index: label} dict appended to each cell caption;
    `suffix` names the output dir (<name>.<suffix>/). Returns (dir, sheets).
    """
    from PIL import Image, ImageDraw
    from concurrent.futures import ThreadPoolExecutor
    jobs = jobs or max(2, (os.cpu_count() or 2) - 2)
    cols, rows, cw, ch = _PREVIEW_COLS, _PREVIEW_ROWS, _PREVIEW_CW, _PREVIEW_CH
    per = cols * rows
    limit_pages = len(pages) if limit is None else max(0, limit)
    idx_list = indices if indices is not None else list(range(min(limit_pages, len(pages))))
    total_sheets = (len(idx_list) + per - 1) // per
    pdir = dest_dir / f'{name}.{suffix}'
    pdir.mkdir(parents=True, exist_ok=True)

    sheets = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for s in range(0, len(idx_list), per):
            batch = idx_list[s:s + per]
            tasks = [(idx, str(pages[idx]), pages_data[idx]['panels'],
                      pages_data[idx].get('source') == 'model') for idx in batch]
            rendered = {idx: (im, model) for idx, im, model in ex.map(_render_cell, tasks)}
            sheet = Image.new('RGB', (cols * cw, rows * ch), 'white')
            sd = ImageDraw.Draw(sheet)
            for k, idx in enumerate(batch):
                im, model = rendered[idx]
                x, y = (k % cols) * cw, (k // cols) * ch
                sheet.paste(im, (x + 4, y + 20))
                cap = f'p{idx + 1}'
                cap += f'  {notes[idx]}' if notes and idx in notes else ('  [model]' if model else '')
                sd.text((x + 5, y + 5), cap, fill='black')
            sheets += 1
            sheet.save(pdir / f'sheet_{sheets:03d}.png')
            if progress:
                progress(sheets, total_sheets)
    return pdir, sheets


def repack(comic, out_dir=None, suffix='', dpi=150, log=None):
    """Normalize a comic into a CBZ with flat, zero-padded page names in reading order.

    KOReader orders archive pages by raw byte order of their paths. Archives with
    chapter subfolders or odd names (e.g. a "- Appendice" folder sorting before
    "00") then read out of order, and panel JSON keyed by page index misaligns.
    Repacking to flat names like 0001.jpg, 0002.jpg, ... makes byte order == reading
    order, so the book reads correctly AND panel indices line up. Also the simplest
    way to turn a PDF into a CBZ (PDF pages are rendered to JPEGs at `dpi`).

    Returns (out_path, page_count). Lossless for image archives (only renamed).
    """
    log = log or (lambda *_: None)
    comic = Path(comic).expanduser()
    tmp = None
    try:
        if comic.is_dir():
            root = comic
        else:
            tmp = tempfile.mkdtemp(prefix='pannello-repack-')
            extract_archive(comic, tmp, dpi=dpi)
            root = tmp
        pages = list_pages(root)
        if not pages:
            raise ValueError(f'no page images found in {comic}')

        dest_dir = Path(out_dir) if out_dir else comic.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f'{comic.stem}{suffix}.cbz'
        if not comic.is_dir() and out.resolve() == comic.resolve():
            out = dest_dir / f'{comic.stem}.pannello.cbz'

        width = max(4, len(str(len(pages))))
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
            for i, p in enumerate(pages, 1):
                z.write(p, arcname=f'{i:0{width}d}{p.suffix.lower()}')
        log(f'repacked {comic.name}: {len(pages)} pages -> {out.name}')
        return out, len(pages)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def find_comics(folder):
    """Return comic archives under a folder (recursive), naturally sorted."""
    folder = Path(folder)
    comics = [p for p in folder.rglob('*')
              if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS
              and '__MACOSX' not in p.parts and not p.name.startswith('._')]
    comics.sort(key=natural_key)
    return comics
