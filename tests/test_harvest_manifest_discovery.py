import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from legacy.harvest_manifests import scan_manifests, cleanup_unmarked_files


class ManifestDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('manifests', 'pairs', 'rollouts'):
            (self.root / name).mkdir()
        self.md = self.root / 'manifests'

    def shard(self, sid, committed=True):
        (self.root/'pairs'/f'pairs_l26_{sid:05d}.safetensors').write_bytes(b'preserve')
        (self.root/'rollouts'/f'rollouts_{sid:05d}.jsonl').write_text('preserve')
        if committed:
            (self.md/f'shard_{sid:05d}.json').write_text(json.dumps({'shard_id':sid}))

    def hidden_manifests(self):
        original = Path.glob
        md = self.md
        return patch.object(Path, 'glob', lambda path, pattern: iter(()) if path == md else original(path, pattern))

    def test_data_names_find_unlisted_manifests(self):
        self.shard(4); self.shard(7)
        with self.hidden_manifests():
            self.assertEqual(set(scan_manifests(self.md)), {4,7})

    def test_progress_finds_shards_when_all_listings_empty(self):
        for i in range(3): self.shard(i)
        (self.root/'progress.json').write_text(json.dumps({'complete_shards':3}))
        with patch.object(Path, 'glob', return_value=iter(())):
            self.assertEqual(set(scan_manifests(self.md)), {0,1,2})

    def test_cleanup_preserves_unlisted_commits(self):
        self.shard(4); self.shard(7); self.shard(8, committed=False)
        with self.hidden_manifests():
            cleanup_unmarked_files(self.root/'pairs',self.root/'rollouts',self.md,[26])
        for i in (4,7):
            self.assertEqual((self.root/'pairs'/f'pairs_l26_{i:05d}.safetensors').read_bytes(),b'preserve')
        self.assertFalse((self.root/'pairs'/'pairs_l26_00008.safetensors').exists())

    def test_progress_missing_manifest_refuses_cleanup(self):
        self.shard(0, committed=False)
        (self.root/'progress.json').write_text(json.dumps({'complete_shards':1}))
        with self.assertRaisesRegex(RuntimeError,'Refusing cleanup'):
            cleanup_unmarked_files(self.root/'pairs',self.root/'rollouts',self.md,[26])
        self.assertTrue((self.root/'pairs'/'pairs_l26_00000.safetensors').exists())

    def test_corrupt_or_wrong_identity_not_treated_as_uncommitted(self):
        self.shard(0)
        path=self.md/'shard_00000.json'
        path.write_text('{broken')
        with self.assertRaises(json.JSONDecodeError): scan_manifests(self.md)
        path.write_text(json.dumps({'shard_id':1}))
        with self.assertRaisesRegex(ValueError,'identity'): scan_manifests(self.md)
