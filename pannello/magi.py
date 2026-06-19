'use strict'
"""
pannello.magi - optional Magi panel-detection engine (opt-in fallback).

Magi ("The Manga Whisperer", ragavsachdeva/magi) is a transformer that segments
comic pages into panels far more precisely than the YOLO models, including
irregular/splash layouts where kumiko collapses to a full page. It runs on CPU
(grayscales input internally, so it works on colour/Western pages too) but is
slow (~2-7s/page) and ~2GB.

NOT installed or used by default. Enable with the [magi] extra + --magi.

LICENSE NOTE: Magi is released for "personal, research, non-commercial, and
not-for-profit" use only (see its model card). pannello never downloads or uses
it unless you explicitly opt in; doing so is your acceptance of Magi's license.
It is therefore never the default and never a hard dependency.
"""

import os
import logging
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')
# The pinned old transformers (~4.36) on a newer torch / huggingface_hub emits
# noisy FutureWarnings at import and download time. They are harmless; silence them
# (these filters are set before transformers is imported, in load_model).
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', message=r'.*_register_pytree_node.*')
warnings.filterwarnings('ignore', message=r'.*resume_download.*')

REPO = os.environ.get('PANNELLO_MAGI_REPO', 'ragavsachdeva/magi')
_MAGI = None


def load_model(model_path=None):
    """Load (and cache) the Magi model. `model_path` is accepted for interface
    parity with pannello.model but ignored (Magi is a single fixed model)."""
    global _MAGI
    if _MAGI is not None:
        return _MAGI
    try:
        import torch  # noqa: F401
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "Magi needs the [magi] extra: pip install 'pannello[magi]'  "
            "(brings in transformers + torch)") from e
    try:
        _MAGI = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
    except Exception as e:
        raise ImportError(
            f"could not load Magi ({REPO}). It requires an older transformers "
            f"(~4.36); install with: pip install 'pannello[magi]'. Cause: {e}") from e
    return _MAGI


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


def detect_panels(image_path, rtl=False, model_path=None, conf=0.25):
    """Return (boxes, (img_w, img_h)) for one image. boxes are pixel [x,y,w,h],
    in reading order. `model_path`/`conf` accepted for interface parity."""
    import numpy as np
    import torch
    from PIL import Image
    model = load_model()
    im = Image.open(image_path)
    w, h = im.size
    arr = np.array(im.convert('L').convert('RGB'))
    with torch.no_grad():
        res = model.predict_detections_and_associations([arr])[0]
    boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
             for x1, y1, x2, y2 in res['panels']]
    return _reading_order(boxes, rtl), (w, h)


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
