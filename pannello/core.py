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
    # Order matters: 'rar archive' and '7-zip' both contain "zip"-ish text, and
    # 'zip' must also catch "Zip multi-volume archive data" (mislabeled .cbr files).
    if 'rar archive' in out:
        return 'rar'
    if '7-zip' in out:
        return '7z'
    if 'zip' in out and 'gzip' not in out:
        return 'zip'
    if 'pdf document' in out:
        return 'pdf'
    if 'posix tar' in out or 'tar archive' in out:
        return 'tar'
    return {'.cbz': 'zip', '.cbr': 'rar', '.cb7': '7z', '.cbt': 'tar',
            '.pdf': 'pdf'}.get(path.suffix.lower())


def extract_archive(path, dest):
    """Extract a comic archive into dest. Returns the detected type."""
    kind = detect_archive_type(path)
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
        subprocess.run(['pdftoppm', '-jpeg', '-r', '150', str(path),
                        str(Path(dest) / 'page')], check=True)
    else:
        raise ValueError(f'unsupported archive: {path} (detected: {kind})')
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


def _run_fallback(pages, pages_data, weak, rtl, model_path, conf, log):
    """Re-detect weak pages with the model. Returns count of pages rescued."""
    from . import model as panel_model
    panel_model.load_model(model_path)
    rescued = 0
    for pn in weak:
        idx = pn - 1
        try:
            boxes, size = panel_model.detect_panels(
                str(pages[idx]), rtl=rtl, model_path=model_path, conf=conf)
        except Exception as e:
            log(f'  fallback error page {pn}: {e}')
            continue
        panels = panel_model.normalize(boxes, size)
        # Only replace when the model finds a genuine multi-panel split. A single
        # model box must NOT override kumiko's full-page result: on near-blank /
        # splash pages the model often emits one spurious box, and replacing the
        # full page with it would zoom the reader to a meaningless region.
        if len(panels) >= 2:
            pages_data[idx]['panels'] = panels
            pages_data[idx]['source'] = 'model'
            rescued += 1
    return rescued


def generate(comic, rtl=None, jobs=None, fallback='auto', model_path=None,
             model_conf=0.25, out_dir=None, limit=None, preview=False, log=None):
    """Generate panel JSON for one comic (archive or folder of images).

    Returns a stats dict. Writes <name>.json next to the comic, or into out_dir.
    """
    log = log or (lambda *_: None)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    comic = Path(comic).expanduser()

    if comic.is_dir():
        name = comic.name
        root = comic
        tmp = None
    else:
        name = comic.stem
        tmp = tempfile.mkdtemp(prefix='pannello-')  # system temp (/tmp on Linux)
        extract_archive(comic, tmp)
        root = tmp

    try:
        pages = list_pages(root)
        if limit:
            pages = pages[:limit]
        if not pages:
            raise ValueError(f'no page images found in {comic}')

        # Reading direction: explicit flag wins; else ComicInfo.xml; else default
        # ltr (grayscale only hints -- never auto-flips, to avoid flipping B&W
        # Western comics).
        rtl_source = 'flag'
        gray_hint = False
        if rtl is None:
            detected = detect_reading_direction(root)
            if detected is not None:
                rtl = detected == 'rtl'
                rtl_source = 'ComicInfo.xml'
            else:
                rtl = False
                rtl_source = 'default'
                gray_hint = looks_grayscale(pages)

        # KOReader reads archives in byte-sort order; warn if that differs from
        # pannello's order (panels would misalign) -- only relevant for archives.
        order_mismatch = (not comic.is_dir()) and koreader_order_differs(pages, root)

        t0 = time.time()
        pages_data, weak, errors = detect_pages(pages, rtl, jobs)

        # Model fallback on weak pages (kumiko found nothing / one full-page box /
        # crashed). 'none' disables it; 'auto' uses the model if installed and
        # degrades quietly otherwise; 'model' requires it.
        crashed = [pn for pn, _ in errors]
        to_fix = [] if fallback == 'none' else weak
        rescued = 0
        if to_fix:
            try:
                rescued = _run_fallback(pages, pages_data, to_fix, rtl, model_path, model_conf, log)
                weak = sorted(i + 1 for i, p in enumerate(pages_data) if is_weak(p['panels']))
            except ImportError:
                if fallback == 'model':
                    raise  # user explicitly asked for the model; surface the install hint
                elif crashed:
                    log(f'  {len(crashed)} page(s) kumiko could not parse; install '
                        f'"pannello[model]" to auto-fill them with the model')

        result = {
            'reading_direction': 'rtl' if rtl else 'ltr',
            'total_pages': len(pages_data),
            'pages': pages_data,
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

        return {
            'comic': name, 'out': out_path, 'pages': len(pages_data),
            'panels': sum(len(p['panels']) for p in pages_data),
            'weak': len(weak), 'rescued': rescued, 'errors': errors,
            'reading_direction': 'rtl' if rtl else 'ltr', 'rtl_source': rtl_source,
            'gray_hint': gray_hint, 'order_mismatch': order_mismatch,
            'preview_dir': preview_dir, 'preview_sheets': preview_sheets,
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


def render_preview(pages, pages_data, dest_dir, name, limit=None, jobs=None, progress=None):
    """Write contact-sheet PNGs with numbered panel boxes for visual QA.

    Boxes are numbered in reading order (so RTL is verifiable: panel 1 sits
    top-right for manga). Green = kumiko, red = model-rescued. Pages are rendered
    in parallel across threads (PIL decode/resize release the GIL). Returns
    (dir, sheets).
    """
    from PIL import Image, ImageDraw
    from concurrent.futures import ThreadPoolExecutor
    jobs = jobs or max(2, (os.cpu_count() or 2) - 2)
    cols, rows, cw, ch = _PREVIEW_COLS, _PREVIEW_ROWS, _PREVIEW_CW, _PREVIEW_CH
    per = cols * rows
    n = min(limit or len(pages), len(pages))
    total_sheets = (n + per - 1) // per
    pdir = dest_dir / f'{name}.preview'
    pdir.mkdir(parents=True, exist_ok=True)

    sheets = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for s in range(0, n, per):
            batch = list(range(s, min(s + per, n)))
            tasks = [(idx, str(pages[idx]), pages_data[idx]['panels'],
                      pages_data[idx].get('source') == 'model') for idx in batch]
            rendered = {idx: (im, model) for idx, im, model in ex.map(_render_cell, tasks)}
            sheet = Image.new('RGB', (cols * cw, rows * ch), 'white')
            sd = ImageDraw.Draw(sheet)
            for k, idx in enumerate(batch):
                im, model = rendered[idx]
                x, y = (k % cols) * cw, (k // cols) * ch
                sheet.paste(im, (x + 4, y + 20))
                sd.text((x + 5, y + 5), f'p{idx + 1}' + ('  [model]' if model else ''), fill='black')
            sheets += 1
            sheet.save(pdir / f'sheet_{sheets:03d}.png')
            if progress:
                progress(sheets, total_sheets)
    return pdir, sheets


def repack(comic, out_dir=None, suffix='', log=None):
    """Normalize a comic into a CBZ with flat, zero-padded page names in reading order.

    KOReader orders archive pages by raw byte order of their paths. Archives with
    chapter subfolders or odd names (e.g. a "- Appendice" folder sorting before
    "00") then read out of order, and panel JSON keyed by page index misaligns.
    Repacking to flat names like 0001.jpg, 0002.jpg, ... makes byte order == reading
    order, so the book reads correctly AND panel indices line up.

    Returns (out_path, page_count). Lossless: images are copied, only renamed.
    """
    log = log or (lambda *_: None)
    comic = Path(comic).expanduser()
    tmp = None
    try:
        if comic.is_dir():
            root = comic
        else:
            tmp = tempfile.mkdtemp(prefix='pannello-repack-')
            extract_archive(comic, tmp)
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
