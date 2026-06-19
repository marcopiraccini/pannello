import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pannello import core


def _make_comic_dir(root, name='comic', filenames=('page001.png',)):
    comic = Path(root) / name
    comic.mkdir()
    for fname in filenames:
        # The tests patch panel detection, so the file only needs the right suffix.
        (comic / fname).write_bytes(b'not an image')
    return comic


class GenerateTests(unittest.TestCase):
    def test_crash_pages_collapse_to_full_page(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            comic = _make_comic_dir(root, 'book')
            out_dir = root / 'out'

            pages_data = [{'page': 1, 'image': 'page001.png', 'panels': []}]
            with patch.object(core, 'detect_pages', return_value=(pages_data, [1], [(1, 'boom')])):
                stats = core.generate(comic, rtl=False, fallback='none', out_dir=out_dir)

            out_path = out_dir / 'book.json'
            payload = json.loads(out_path.read_text(encoding='utf-8'))

            self.assertEqual(stats['pages'], 1)
            self.assertEqual(payload['total_pages'], 1)
            self.assertEqual(payload['pages'][0]['panels'], [
                {'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0},
            ])
            self.assertNotIn('source', payload['pages'][0])
            self.assertEqual(stats['low_confidence'][0]['fullpage'], True)
            self.assertEqual(stats['low_confidence'][0]['panels'], 1)

    def test_limit_zero_skips_detection_and_preview(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            comic = _make_comic_dir(root, 'book')
            out_dir = root / 'out'

            with patch.object(core, 'detect_pages', side_effect=AssertionError('should not run')):
                stats = core.generate(
                    comic,
                    rtl=False,
                    fallback='none',
                    out_dir=out_dir,
                    limit=0,
                    preview=True,
                )

            out_path = out_dir / 'book.json'
            payload = json.loads(out_path.read_text(encoding='utf-8'))
            preview_dir = out_dir / 'book.preview'

            self.assertEqual(stats['pages'], 0)
            self.assertEqual(payload['total_pages'], 0)
            self.assertEqual(payload['pages'], [])
            self.assertEqual(stats['preview_sheets'], 0)
            self.assertTrue(preview_dir.exists())
            self.assertEqual(list(preview_dir.glob('sheet_*.png')), [])

    def test_negative_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            comic = _make_comic_dir(root, 'book')

            with self.assertRaises(ValueError):
                core.generate(comic, rtl=False, fallback='none', limit=-1)

    def test_fallback_accepts_overlapping_engine_output_via_deoverlap(self):
        # A big panel's bbox overlapping its neighbours is a legit layout: the
        # fallback must clip it apart and ACCEPT it, not reject it as anomalous.
        class OverlapEngine:
            @staticmethod
            def load_model(model_path=None):
                pass

            @staticmethod
            def detect_panels(path, rtl=False, model_path=None, conf=0.25):
                return [[0, 0, 60, 100], [40, 0, 60, 100]], (100, 100)  # overlap x 0.4-0.6

            @staticmethod
            def normalize(boxes, size):
                w, h = size
                return [{'x': x / w, 'y': y / h, 'w': bw / w, 'h': bh / h}
                        for x, y, bw, bh in boxes]

        pages = [Path('p1.png')]
        pages_data = [{'page': 1, 'image': 'p1.png',
                       'panels': [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}]}]
        rescued = core._run_fallback(pages, pages_data, [1], False, None, 0.25,
                                     lambda *a: None, engine=OverlapEngine)

        self.assertEqual(rescued, 1)
        self.assertEqual(pages_data[0]['source'], 'model')
        self.assertGreaterEqual(len(pages_data[0]['panels']), 2)
        self.assertFalse(core.detect_anomalies(pages_data[0]['panels']))

    def test_fallback_rejects_engine_output_that_loses_extent(self):
        # Engine covers only the left half -> it dropped a region kumiko had -> reject.
        class HalfEngine:
            @staticmethod
            def load_model(model_path=None):
                pass

            @staticmethod
            def detect_panels(path, rtl=False, model_path=None, conf=0.25):
                return [[0, 0, 25, 100], [25, 0, 25, 100]], (100, 100)  # only x 0-0.5

            @staticmethod
            def normalize(boxes, size):
                w, h = size
                return [{'x': x / w, 'y': y / h, 'w': bw / w, 'h': bh / h}
                        for x, y, bw, bh in boxes]

        pages = [Path('p1.png')]
        kept = [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}]
        pages_data = [{'page': 1, 'image': 'p1.png', 'panels': kept}]
        rescued = core._run_fallback(pages, pages_data, [1], False, None, 0.25,
                                     lambda *a: None, engine=HalfEngine)

        self.assertEqual(rescued, 0)
        self.assertNotIn('source', pages_data[0])
        self.assertEqual(pages_data[0]['panels'], kept)

    def test_magi_primary_kumiko_rescues_holey_page(self):
        # Magi left a mid-page hole; kumiko has a clean 2-panel grid -> kumiko wins.
        magi_holey = [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 0.3},
                      {'x': 0.0, 'y': 0.7, 'w': 1.0, 'h': 0.3}]
        pages = [Path('p1.png')]
        pages_data = [{'page': 1, 'image': 'p1.png', 'panels': magi_holey}]
        info = {'size': [100, 100], 'panels': [[0, 0, 100, 50], [0, 50, 100, 50]]}

        with patch.object(core, 'kumiko_one', return_value=info):
            n = core._kumiko_fallback_for_magi(pages, pages_data, False)

        self.assertEqual(n, 1)
        self.assertEqual(pages_data[0]['source'], 'kumiko')
        self.assertFalse(core.has_hole(pages_data[0]['panels']))

    def test_magi_primary_keeps_clean_magi_without_calling_kumiko(self):
        clean = [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 0.5},
                 {'x': 0.0, 'y': 0.5, 'w': 1.0, 'h': 0.5}]
        pages = [Path('p1.png')]
        pages_data = [{'page': 1, 'image': 'p1.png', 'panels': clean}]

        with patch.object(core, 'kumiko_one', side_effect=AssertionError('kumiko not needed')):
            n = core._kumiko_fallback_for_magi(pages, pages_data, False)

        self.assertEqual(n, 0)
        self.assertNotIn('source', pages_data[0])

    def test_magi_primary_trusts_single_panel_splash(self):
        # A Magi 1-panel result is a real splash; kumiko must not be consulted.
        pages = [Path('p1.png')]
        pages_data = [{'page': 1, 'image': 'p1.png',
                       'panels': [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}]}]

        with patch.object(core, 'kumiko_one', side_effect=AssertionError('splash, no kumiko')):
            n = core._kumiko_fallback_for_magi(pages, pages_data, False)

        self.assertEqual(n, 0)
        self.assertNotIn('source', pages_data[0])

    def test_preview_only_missing_json_raises(self):
        with tempfile.TemporaryDirectory() as td:
            comic = _make_comic_dir(Path(td), 'book', filenames=('page001.png',))
            with self.assertRaises(FileNotFoundError):
                core.preview_from_json(comic)

    def test_preview_from_json_uses_json_panels_and_runs_no_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            comic = _make_comic_dir(root, 'book', filenames=('p1.png', 'p2.png'))
            panels = [{'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 0.5},
                      {'x': 0.0, 'y': 0.5, 'w': 1.0, 'h': 0.5}]
            (root / 'book.json').write_text(json.dumps({
                'reading_direction': 'ltr', 'total_pages': 2,
                'pages': [{'page': 1, 'image': 'p1.png', 'panels': panels},
                          {'page': 2, 'image': 'p2.png', 'panels': []}]}), encoding='utf-8')

            captured = {}

            def fake_render(pages, pages_data, dest_dir, name, **kw):
                captured['pages'] = pages
                captured['data'] = pages_data
                return (dest_dir / f'{name}.preview', 1)

            with patch.object(core, 'render_preview', side_effect=fake_render), \
                 patch.object(core, 'detect_pages', side_effect=AssertionError('no detection')), \
                 patch.object(core, 'kumiko_one', side_effect=AssertionError('no kumiko')):
                st = core.preview_from_json(comic)

            self.assertEqual([d['panels'] for d in captured['data']], [panels, []])
            self.assertEqual([p.name for p in captured['pages']], ['p1.png', 'p2.png'])
            self.assertEqual(st['pages'], 2)

    def test_extract_archive_missing_tool_raises_helpful_error(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            src = dest / 'comic.cbr'
            src.write_bytes(b'not used')

            with patch.object(core, 'detect_archive_type', return_value='rar'), \
                 patch.object(core.subprocess, 'run', side_effect=FileNotFoundError('missing')):
                with self.assertRaises(RuntimeError) as cm:
                    core.extract_archive(src, dest)

            self.assertIn('unrar', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
