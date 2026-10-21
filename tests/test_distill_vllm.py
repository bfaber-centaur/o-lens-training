"""CPU-only contract tests for the bulk vLLM teacher adapter.

No vLLM installation or GPU required; the actual engine is imported lazily.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from oracle.sample_distill_teacher_vllm import (
    shard_bounds, inject_prompt, sample_chunk, contract, materialize_requests
)


class FakeParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeEngine:
    def generate(self, prompts, sampling_params, use_tqdm):
        assert not use_tqdm
        assert len(prompts)==len(sampling_params)
        return [SimpleNamespace(outputs=[
            SimpleNamespace(index=j, token_ids=[9,j+20], text=f"sample-{j}")
            for j in range(params.n)][::-1]) for params in sampling_params]


class VLLMBulkTests(unittest.TestCase):
    def test_sharding_exact_cover(self):
        for n in (1,2,3,4,7):
            spans = [shard_bounds(11,21,n,i) for i in range(n)]
            joined=[idx for lo,hi in spans for idx in range(lo,hi)]
            self.assertEqual(joined,list(range(11,32)))
        with self.assertRaises(ValueError):
            shard_bounds(0,2,3,0)

    def test_inject_replaces_only_final_embedding(self):
        matrix=torch.arange(120,dtype=torch.float32).reshape(30,4)
        activation=torch.tensor([3.,4.,0.,0.])
        ids=[5,7,8]
        out=inject_prompt(matrix,ids,activation,10.0)
        self.assertTrue(torch.equal(out[:-1],matrix[torch.tensor(ids[:-1])]))
        self.assertTrue(torch.allclose(out[-1],torch.tensor([6.,8.,0.,0.])))
        self.assertAlmostEqual(float(out[-1].norm()),10.)
        self.assertTrue(torch.equal(matrix[8],torch.arange(32.,36.)))
        with self.assertRaisesRegex(ValueError,"near-zero"):
            inject_prompt(matrix,ids,torch.zeros(4),10.)

    def test_chunk_output_contract_and_shard_independent_seeds(self):
        prompts=[{"prompt_embeds":torch.ones(3,4)} for _ in range(2)]
        samples=sample_chunk(FakeEngine(),FakeParams,prompts,[7,19],4,100,42)
        self.assertEqual(len(samples),2)
        self.assertEqual(len(samples[0]),4)
        self.assertEqual(samples[0][0]["ids"],[9,20])
        self.assertEqual(samples[1][3]["text"],"sample-3")

    def test_materialize_requests_is_per_row(self):
        bank={"train_vec":torch.tensor([[3.,4.],[0.,8.]]),
              "train_pair_rows":torch.tensor([1,2])}
        mat=torch.eye(3,2)
        req=materialize_requests([0,1],bank,"train",mat,[0,1],10)
        self.assertEqual(len(req),2)
        self.assertFalse(torch.equal(req[0]["prompt_embeds"][-1],req[1]["prompt_embeds"][-1]))

    def test_manifest_compatibility_and_mismatched_alpha_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            teacher=root/"teacher"
            items=root/"items"
            merged=root/"merged"
            (teacher/"lora").mkdir(parents=True)
            items.mkdir()
            merged.mkdir()
            (teacher/"COMPLETE").write_text("ok")
            (items/"COMPLETE").write_text("ok")
            adapter=b"adapterbytes"
            (teacher/"lora"/"adapter_model.safetensors").write_bytes(adapter)
            (items/"items.safetensors").write_bytes(b"itemsbytes")
            tm={"stage":"l26_ao_sft_fresh_pool_v1","model":"Qwen/Qwen3.5-4B",
                "revision":"revision","layer":26,"alpha":8000.,
                "prompt":"Read an activation!","prompt_slot":"last token of rendered prompt",
                "transform":"unit"}
            im={"schema":"gt_distill_items_v1","model":tm["model"],"layer":26,
                "teacher":str(teacher),"teacher_alpha":8000.,
                "prompt":tm["prompt"],"train_rows":40,"val_rows":10,
                "items_sha256":hashlib.sha256(b"itemsbytes").hexdigest()}
            mm={"schema":"l26_vllm_merged_teacher_v1",
                "source_checkpoint":str(teacher),
                "source_adapter_sha256":hashlib.sha256(adapter).hexdigest(),
                "model":tm["model"],"layer":26,"transform":"unit",
                "prompt":tm["prompt"],"prompt_slot":tm["prompt_slot"],"alpha":8000.}
            (teacher/"meta.json").write_text(json.dumps(tm))
            (items/"meta.json").write_text(json.dumps(im))
            (merged/"olens_merge_manifest.json").write_text(json.dumps(mm))
            _,_,_,manifest=contract(teacher,items,merged,"train",0,40,4,2,64,100,2,5,128)
            self.assertEqual((manifest["start"],manifest["count"]),(20,10))
            self.assertEqual(manifest["format"],"gt_distill_teacher_samples_v1")
            self.assertEqual(manifest["samples_per_item"],64)
            mm["alpha"]=16000.
            (merged/"olens_merge_manifest.json").write_text(json.dumps(mm))
            with self.assertRaisesRegex(ValueError,"contract mismatch"):
                contract(teacher,items,merged,"train",0,40,4,2,64,100,2,5,128)


if __name__=="__main__":
    unittest.main()
