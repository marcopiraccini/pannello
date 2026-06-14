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
    if 'zip archive' in out:
        return 'zip'
    if 'rar archive' in out:
        return 'rar'
    if '7-zip' in out:
        return '7z'
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
        if len(panels) >= 2 or (len(panels) == 1 and not is_weak(panels)):
            pages_data[idx]['panels'] = panels
            pages_data[idx]['source'] = 'model'
            rescued += 1
    return rescued


def generate(comic, rtl=False, jobs=None, fallback='none', model_path=None,
             model_conf=0.25, out_dir=None, limit=None, log=None):
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

        t0 = time.time()
        pages_data, weak, errors = detect_pages(pages, rtl, jobs)

        rescued = 0
        if fallback == 'model' and weak:
            rescued = _run_fallback(pages, pages_data, weak, rtl, model_path, model_conf, log)
            weak = sorted(i + 1 for i, p in enumerate(pages_data) if is_weak(p['panels']))

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

        return {
            'comic': name, 'out': out_path, 'pages': len(pages_data),
            'panels': sum(len(p['panels']) for p in pages_data),
            'weak': len(weak), 'rescued': rescued, 'errors': errors,
            'seconds': time.time() - t0,
        }
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
