#!/usr/bin/env python3
'use strict'
"""
run_benchmark.py - compare kumiko vs the model on a folder of comic pages.

Runs the official-kumiko detector and the YOLO model on each page, prints a
per-page panel-count table, and (if a ground_truth.json is given) reports
mean absolute count error and exact-match rate for each method. Saves overlay
images (kumiko = red, model = green) so disagreements can be eyeballed.

Run with the venv python (needs torch/ultralytics for the model):

    ../.venv/bin/python run_benchmark.py                       # free P&C set + ground truth
    ../.venv/bin/python run_benchmark.py --pages /path/to/pages --no-gt
    ../.venv/bin/python run_benchmark.py --pages DIR --conf 0.2 --no-overlays

NOTE: only the bundled Pepper&Carrot pages (CC-BY) are redistributable. You can
point --pages at your own comics for a private evaluation, but those results are
not part of the public benchmark.
"""

import os
import sys
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

from pannello import core as mp  # noqa: E402
from pannello import model as pm  # noqa: E402


def overlay(img_path, kumiko_panels, model_panels, dest):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert('RGB')
    W, H = im.size
    d = ImageDraw.Draw(im)
    for p in kumiko_panels:
        d.rectangle([p['x'] * W, p['y'] * H, (p['x'] + p['w']) * W, (p['y'] + p['h']) * H],
                    outline=(230, 0, 0), width=6)
    for p in model_panels:
        d.rectangle([p['x'] * W, p['y'] * H, (p['x'] + p['w']) * W, (p['y'] + p['h']) * H],
                    outline=(0, 200, 0), width=3)
    im.thumbnail((900, 1300))
    im.save(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pages', default=os.path.join(HERE, 'pages'))
    ap.add_argument('--gt', default=os.path.join(HERE, 'ground_truth.json'))
    ap.add_argument('--no-gt', action='store_true')
    ap.add_argument('--rtl', action='store_true')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--overlays', default=os.path.join(HERE, 'overlays'))
    ap.add_argument('--no-overlays', action='store_true')
    args = ap.parse_args()

    gt = {}
    if not args.no_gt and os.path.exists(args.gt):
        gt = json.load(open(args.gt, encoding='utf-8')).get('pages', {})

    files = sorted(glob.glob(os.path.join(args.pages, '*.jpg')) +
                   glob.glob(os.path.join(args.pages, '*.png')))
    if gt:
        files = [f for f in files if os.path.basename(f) in gt] or files
    if not files:
        print(f'no pages in {args.pages} (run fetch_pc.py first)', file=sys.stderr)
        sys.exit(1)

    pm.load_model()
    if not args.no_overlays:
        os.makedirs(args.overlays, exist_ok=True)

    has_gt = bool(gt)
    header = f"{'page':16}"
    header += f"{'true':>6}" if has_gt else ''
    header += f"{'kumiko':>8}{'model':>7}"
    header += f"{'k_err':>7}{'m_err':>7}" if has_gt else ''
    print(header)
    print('-' * len(header))

    k_abs = m_abs = k_exact = m_exact = n = 0
    for f in files:
        name = os.path.basename(f)
        kinfo = mp.kumiko_one(f, args.rtl)
        kpan = mp.normalize_panels(kinfo)
        boxes, size = pm.detect_panels(f, rtl=args.rtl, conf=args.conf)
        mpan = pm.normalize(boxes, size)
        row = f"{name:16}"
        if has_gt:
            t = gt[name]
            ke, me = abs(len(kpan) - t), abs(len(mpan) - t)
            k_abs += ke; m_abs += me
            k_exact += (len(kpan) == t); m_exact += (len(mpan) == t)
            row += f"{t:>6}{len(kpan):>8}{len(mpan):>7}{ke:>7}{me:>7}"
        else:
            row += f"{len(kpan):>8}{len(mpan):>7}"
        print(row)
        n += 1
        if not args.no_overlays:
            overlay(f, kpan, mpan, os.path.join(args.overlays, name + '.overlay.png'))

    if has_gt:
        print('-' * len(header))
        print(f'pages: {n}')
        print(f'kumiko  MAE {k_abs / n:.2f}  exact {k_exact}/{n}')
        print(f'model   MAE {m_abs / n:.2f}  exact {m_exact}/{n}')
    if not args.no_overlays:
        print(f'overlays -> {args.overlays}/ (kumiko=red, model=green)')


if __name__ == '__main__':
    main()
