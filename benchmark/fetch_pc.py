#!/usr/bin/env python3
'use strict'
"""
fetch_pc.py - download the free, reproducible benchmark pages.

Source: "Pepper & Carrot" by David Revoy, licensed CC-BY 4.0.
        https://www.peppercarrot.com  (attribution required, redistribution allowed)

Downloads the comic pages listed in ground_truth.json into ./pages/ so the
benchmark is fully reproducible from public URLs. Pages are the official
low-res renders.
"""

import os
import json
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# episode dir on the server, keyed by the E## code used in page filenames.
EPISODES = {
    'E10': 'ep10_Summer-Special',
    'E31': 'ep31_The-Fight',
    'E33': 'ep33_Spell-of-War',
    'E01': 'ep01_Potion-of-Flight',
}
URL = ('https://www.peppercarrot.com/0_sources/{ep}/low-res/'
       'en_Pepper-and-Carrot_by-David-Revoy_{page}')


def main():
    gt = json.load(open(os.path.join(HERE, 'ground_truth.json'), encoding='utf-8'))
    pages = gt['pages']
    out = os.path.join(HERE, 'pages')
    os.makedirs(out, exist_ok=True)
    ok = 0
    for fname in sorted(pages):
        code = fname[:3]  # E10
        ep = EPISODES.get(code)
        if not ep:
            print(f'skip {fname}: unknown episode {code}')
            continue
        url = URL.format(ep=ep, page=fname)
        dest = os.path.join(out, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 5000:
            ok += 1
            continue
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'panels-benchmark/1.0'})
            data = urllib.request.urlopen(req, timeout=30).read()
            if len(data) > 5000:
                open(dest, 'wb').write(data)
                ok += 1
                print(f'ok  {fname}')
            else:
                print(f'FAIL {fname}: too small')
        except Exception as e:
            print(f'FAIL {fname}: {e}')
    print(f'\n{ok}/{len(pages)} pages in {out}')


if __name__ == '__main__':
    main()
