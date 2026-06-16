'use strict'
"""
pannello.model - optional model-based panel detection fallback.

Uses a general comic-panel YOLO model via ultralytics (CPU by default). Only
imported when --fallback model is used, so the base install needs no torch.

Default model: mosesb/best-comic-panel-detection (YOLOv12, Apache-2.0), which
detects panels on Western and manga pages. Override via env PANNELLO_MODEL_REPO
/ PANNELLO_MODEL_FILE or the --model-path option.

NOTE: ultralytics is AGPL-3.0; installing the [model] extra brings it in.
"""

import os
import sys
import logging
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '4')

# Quiet the "unauthenticated requests to the HF Hub / set HF_TOKEN" notice: we
# only download a public model once, anonymous access is fine. HF_HUB_VERBOSITY
# is read by huggingface_hub at import time (a plain setLevel gets reset then).
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')

DEFAULT_REPO = os.environ.get('PANNELLO_MODEL_REPO', 'mosesb/best-comic-panel-detection')
DEFAULT_FILE = os.environ.get('PANNELLO_MODEL_FILE', 'best.pt')
PANEL_CLASS_NAMES = ('frame', 'panel', 'panels', 'comic panel', 'comic_panel')

_MODEL = None
_PANEL_CLASS_IDS = None


# Convenience presets selectable by name via --model.
PRESETS = {
    'general': (DEFAULT_REPO, DEFAULT_FILE),          # mosesb, Western + manga
    'manga': ('deepghs/manga109_yolo', 'v2023.12.07_l_yv11/model.pt'),
}


def _resolve_weights(model):
    """Resolve --model into a local weights path, downloading from the Hub if needed.

    Accepts: None (default 'general' preset), a preset name ('general'/'manga'),
    a local .pt path, or a Hub spec 'repo' / 'repo:path/to/weights.pt'.
    """
    from huggingface_hub import hf_hub_download
    if model is None or model in PRESETS:
        repo, file = PRESETS.get(model or 'general')
        return hf_hub_download(repo_id=repo, filename=file)
    if os.path.exists(model):
        return model
    if '/' in model:  # Hub repo id, optionally 'repo:weights.pt'
        repo, _, file = model.partition(':')
        return hf_hub_download(repo_id=repo, filename=file or DEFAULT_FILE)
    raise ValueError(f"--model '{model}': not a preset, local path, or hub repo id")


def load_model(model_path=None):
    """Load (and cache) the YOLO model. Downloads weights into the HF cache on first use."""
    global _MODEL, _PANEL_CLASS_IDS
    if _MODEL is not None:
        return _MODEL
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "the model fallback needs the [model] extra: "
            "pip install 'pannello[model]'  (or install torch + ultralytics)") from e

    weights = _resolve_weights(model_path)
    model = YOLO(weights).to('cuda' if torch.cuda.is_available() else 'cpu')

    ids = [cid for cid, name in model.names.items()
           if str(name).lower() in PANEL_CLASS_NAMES]
    if not ids:
        ids = list(model.names.keys())  # single-class model: assume it's the panel
        if len(model.names) > 1:
            print(f'warning: no panel class in {model.names}; keeping all classes',
                  file=sys.stderr)
    _MODEL = model
    _PANEL_CLASS_IDS = set(ids)
    return _MODEL


def _reading_order(boxes, rtl):
    """Sort [x,y,w,h] boxes top-to-bottom, then within a row by reading direction."""
    boxes = sorted(boxes, key=lambda b: b[1])
    rows = []
    for b in boxes:
        placed = False
        for row in rows:
            ry, rh = row[0][1], row[0][3]
            overlap = min(b[1] + b[3], ry + rh) - max(b[1], ry)
            if overlap > 0.5 * min(b[3], rh):
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    rows.sort(key=lambda r: min(box[1] for box in r))
    ordered = []
    for row in rows:
        row.sort(key=lambda b: b[0], reverse=rtl)
        ordered.extend(row)
    return ordered


def detect_panels(image_path, rtl=False, model_path=None, conf=0.25, iou=0.6, imgsz=1024):
    """Return (boxes, (img_w, img_h)) for one image. boxes are pixel [x,y,w,h], reading order."""
    model = load_model(model_path)
    res = model(image_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    img_h, img_w = res.orig_shape
    boxes = []
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c in zip(xyxy, cls):
        if c in _PANEL_CLASS_IDS:
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return _reading_order(boxes, rtl), (img_w, img_h)


def normalize(boxes, size):
    """Pixel [x,y,w,h] boxes -> normalized {x,y,w,h} dicts (0..1, clamped)."""
    w, h = size
    out = []
    for x, y, bw, bh in boxes:
        out.append({
            'x': round(max(0.0, min(1.0, x / w)), 4),
            'y': round(max(0.0, min(1.0, y / h)), 4),
            'w': round(max(0.0, min(1.0, bw / w)), 4),
            'h': round(max(0.0, min(1.0, bh / h)), 4),
        })
    return out
