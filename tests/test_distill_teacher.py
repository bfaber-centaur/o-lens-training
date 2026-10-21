"""CPU-only checks for the GT distill item/teacher sampling contract."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from oracle.prepare_distill_items import choose_items, partition, degenerate
from oracle.sample_distill_teacher import load_contract


class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "Here is a meaningful natural language continuation."


class DistillTests(unittest.TestCase):
    def test_deterministic_rollout_disjoint_selection(self):
        n=80
        corpus=SimpleNamespace(
            val_indices=torch.arange(n),
            shard_ids=torch.zeros(n,dtype=torch.long),
            rollout_ids=torch.arange(n),
            span_offsets=torch.arange(n+1)*12,
            span_token_ids=torch.arange(n*12,dtype=torch.long),
            activations=torch.ones((n,8),dtype=torch.float32))
        a,_=choose_items(corpus,DummyTokenizer(),seed=42,n_train=5,n_val=3,val_frac=.3)
        b,_=choose_items(corpus,DummyTokenizer(),seed=42,n_train=5,n_val=3,val_frac=.3)
        self.assertEqual(a,b)
        left={(s,r) for _,s,r,_ in a["train"]}
        right={(s,r) for _,s,r,_ in a["val"]}
        self.assertTrue(left.isdisjoint(right))
        self.assertEqual((len(left),len(right)),(5,3))
        for split in a:
            for _,shard,rollout,_ in a[split]:
                self.assertEqual(split,partition(shard,rollout,42,.3))

    def test_degenerate_filter(self):
        self.assertTrue(degenerate("   "))
        self.assertTrue(degenerate("one two two two two"))
        self.assertFalse(degenerate("These are four perfectly ordinary concepts in context."))

    def test_sampling_manifest_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            teacher=root/"teacher"
            items=root/"items"
            (teacher/"lora").mkdir(parents=True)
            items.mkdir()
            (teacher/"COMPLETE").write_text("ok\n")
            (items/"COMPLETE").write_text("ok\n")
            (teacher/"lora"/"adapter_model.safetensors").write_bytes(b"fake-adapter")
            (items/"items.safetensors").write_bytes(b"fake-items")
            tm={"stage":"l26_ao_sft_fresh_pool_v1","model":"Qwen/Qwen3.5-4B",
                "layer":26,"alpha":8000.0,"transform":"unit",
                "prompt":"Read the following representation and reproduce the text:\n",
                "prompt_slot":"last token of rendered prompt"}
            im={"model":tm["model"],"layer":26,"teacher":str(teacher),
                "teacher_alpha":8000.0,"prompt":tm["prompt"],
                "train_rows":10,"val_rows":4,
                "items_sha256":hashlib.sha256(b"fake-items").hexdigest()}
            (teacher/"meta.json").write_text(json.dumps(tm))
            (items/"meta.json").write_text(json.dumps(im))
            _,_,manifest=load_contract(teacher,items,"train",0,2,64,100,8,0)
            self.assertEqual(manifest["samples_per_item"],64)
            self.assertEqual(manifest["alpha"],8000.0)
            im["teacher_alpha"]=16000
            (items/"meta.json").write_text(json.dumps(im))
            with self.assertRaisesRegex(ValueError,"injection contract"):
                load_contract(teacher,items,"train",0,2,64,100,8,0)


if __name__=="__main__":
    unittest.main()
