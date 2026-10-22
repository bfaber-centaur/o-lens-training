"""CPU synthetic checks for fresh AO pool diversity and prefix exclusion."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file

SPEC=importlib.util.spec_from_file_location("pool_builder",Path(__file__).resolve().parents[1]/"oracle"/"build_fresh_ao_pool.py")
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

class FakeTokenizer:
    all_special_ids=[0]
    def decode(self,seq):return " ".join(map(str,seq))

class PoolTests(unittest.TestCase):
    def pair(self,path,prefix):
        ids=torch.tensor(prefix,dtype=torch.int32)
        offsets=torch.tensor([0,len(ids)],dtype=torch.int64)
        save_file({"span_token_ids":ids,"span_offsets":offsets},str(path))

    def test_exclude_ar_prefix_and_nonoverlap(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"pairs_l26_0000.safetensors"
            self.pair(f,list(range(1,65)))
            # Several distinct 64-token windows, one an AR training prefix.
            roll=Path(td)/"rollouts_0000.jsonl"
            rows=[
                {"rollout_id":1,"output_ids":list(range(1,65))+list(range(101,165))},
                {"rollout_id":2,"output_ids":list(range(201,265))+list(range(301,365))},
                {"rollout_id":3,"output_ids":list(range(401,465))+list(range(501,565))},
            ]
            import json
            roll.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
            pool,stats=mod.create_pool([roll],[f],FakeTokenizer(),seed=42,val_fraction=.5)
            self.assertGreater(stats["ar_prefix_excluded"],0)
            self.assertGreater(sum(map(len,pool.values())),0)
            self.assertEqual(sum(map(len,pool.values()))+stats["ar_prefix_excluded"]+stats["duplicate_crop"]+stats["duplicate_window"]+stats["special"]+stats["placeholder"],stats["windows"])

    def test_fail_closed_when_no_output_ids(self):
        with tempfile.TemporaryDirectory() as td:
            import json
            pair=Path(td)/"pairs_l26_0000.safetensors"
            self.pair(pair,list(range(1,65)))
            roll=Path(td)/"rollouts_0000.jsonl"
            roll.write_text(json.dumps({"rollout_id":1,"prompt":"hi"})+"\n")
            with self.assertRaisesRegex(ValueError,r"missing (assistant_ids/)?output_ids"):
                mod.create_pool([roll],[pair],FakeTokenizer())
if __name__=="__main__": unittest.main()

