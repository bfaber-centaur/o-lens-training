import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts/preflight_l30.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class PreflightTests(unittest.TestCase):
    def fixture(self, root):
        cp = root / "lab/ar/l30_200k_v3/checkpoints/step_001412"
        cp.mkdir(parents=True)
        w = root / "lab/ar/l30_200k_v3/whitener_l30.safetensors"
        w.write_bytes(b"whitener")
        cfg = dict(model_id="Qwen/Qwen3.5-4B", target_layer=30, d_model=2560,
                   split_seed=42, val_frac=.1, pair_shards=["pairs_l30_00000.safetensors"],
                   rollout_shards=["rollouts_00000.jsonl"], whitener_sha256=m.sha(w))
        ref = root / "repo/legacy/artifacts/ar/l30_200k_v3/run_config.json"
        ref.parent.mkdir(parents=True)
        ref.write_text(json.dumps(cfg))
        (cp / "run_config.json").write_text(json.dumps(cfg))
        for name in ("COMPLETE", "training_state.pt", "lora/adapter_config.json",
                     "lora/adapter_model.safetensors"):
            p = cp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fixture")
        data = root / "lab/data/olens_200k_l22_l26_l30"
        for directory, names in (("pairs", cfg["pair_shards"]), ("rollouts", cfg["rollout_shards"])):
            (data / directory).mkdir(parents=True)
            for name in names:
                (data / directory / name).write_bytes(b"fixture")
        return cp, w

    def run_preflight(self, root):
        with patch.object(m, "__file__", str(root / "repo/scripts/preflight_l30.py")), patch(
            "sys.argv", ["preflight", "--lab", str(root / "lab"),
                         "--output", str(root / "receipt.json")]):
            m.main()

    def test_receipt_and_missing_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp, _ = self.fixture(root)
            self.run_preflight(root)
            receipt = json.loads((root / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "sources_readable")
            self.assertEqual(len(receipt["sha256"]), 8)
            (root / "receipt.json").unlink()
            (cp / "training_state.pt").unlink()
            with self.assertRaises(RuntimeError):
                self.run_preflight(root)
            self.assertFalse((root / "receipt.json").exists())

    def test_whitener_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, w = self.fixture(root)
            w.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                self.run_preflight(root)
            self.assertFalse((root / "receipt.json").exists())

if __name__ == "__main__":
    unittest.main()
