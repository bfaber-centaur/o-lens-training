import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "exporter", Path(__file__).resolve().parents[1] / "scripts/export_l26_hf.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ExportTests(unittest.TestCase):
    def test_copy_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, dest = root/"source", root/"out/adapter"
            source.write_bytes(b"weights")
            m.copy_verified(source, dest)
            out = dest.parent
            (out/"SHA256SUMS").write_text(f"{m.sha(dest)}  adapter\n")
            m.verify_payload(out)
            dest.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                m.verify_payload(out)

    def test_symlink_and_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"source").write_bytes(b"x")
            (root/"link").symlink_to(root/"source")
            with self.assertRaises(ValueError):
                m.copy_verified(root/"link", root/"dest")
            (root/"SHA256SUMS").write_text("bad  ../outside\n")
            with self.assertRaises(ValueError):
                m.verify_payload(root)

if __name__ == "__main__":
    unittest.main()
