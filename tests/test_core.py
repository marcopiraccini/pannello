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
