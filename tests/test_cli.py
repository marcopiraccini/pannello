import io
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr

from pannello import cli


class CliTests(unittest.TestCase):
    def test_rtl_and_ltr_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            comic = Path(td) / 'comic.cbz'
            comic.write_bytes(b'not used')

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    cli.main(['--rtl', '--ltr', str(comic)])

            self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
