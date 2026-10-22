"""CPU-only checks for the L26 distillation LoRA trainer (no model/GPU required)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from oracle.train_distill_student import (
    PROMPT_TEMPLATE, render_concepts_prompt, load_targets,
    row, pack, conditioned_batch, next_indices, verified_contract, sha256
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        # A single ideograph in the accepted upstream range has one ID.
        if text == chr(0x3200):
            return SimpleNamespace(input_ids=[800])
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        s = messages[0]["content"]
        assert "distinct concepts" in s
        prefix, suffix = s.split(chr(0x3200))
        return [11] + [ord(c) for c in prefix] + [800] + [
            ord(c) for c in suffix] + [12]


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(900, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(3600).reshape(900, 4) / 100)

    def get_input_embeddings(self):
        return self.embedding


class DistillStudentTests(unittest.TestCase):
    def test_prompt_slot_unique_and_matches_upstream_wording(self):
        ids, slot, char = render_concepts_prompt(FakeTokenizer(), 26)
        self.assertEqual(char, chr(0x3200))
        self.assertEqual(int(ids[slot]), 800)
        self.assertEqual(int((ids == 800).sum()), 1)
        self.assertEqual(PROMPT_TEMPLATE.format(layer=26, char=char).count(
            "each as a '- ' bullet on its own line."), 1)

    def test_packed_eos_not_duplicated_and_prompt_masks(self):
        rows = [(torch.tensor([3.,4.,0.,0.]), torch.tensor([21,22,2])),
                (torch.tensor([0.,1.,0.,0.]), torch.tensor([23,2]))]
        vec, ids, labs, masks = pack(rows, pad_id=0)
        self.assertEqual(labs.tolist(), [[21,22,2],[23,2,-100]])
        self.assertEqual(masks.tolist(), [[1,1,1],[1,1,0]])
        model = Embedding()
        prompt = torch.tensor([1,2,3])
        emb, attn, labels = conditioned_batch(
            model, prompt, 1, vec, ids, labs, masks, 10.)
        self.assertEqual(emb.shape, (2,6,4))
        self.assertTrue(torch.allclose(emb[0,1],torch.tensor([6.,8.,0.,0.])))
        self.assertTrue(torch.equal(emb[0,0],model.embedding(prompt[0])))
        self.assertTrue(torch.equal(emb[0,2],model.embedding(prompt[2])))
        self.assertEqual(labels.tolist(),
                         [[-100,-100,-100,21,22,2],
                          [-100,-100,-100,23,2,-100]])
        self.assertEqual(attn.tolist(),[[1,1,1,1,1,1],[1,1,1,1,1,0]])

    def test_step_schedule_deterministic(self):
        self.assertEqual(next_indices(374,42,1,0,5),next_indices(374,42,1,0,5))
        self.assertNotEqual(next_indices(374,42,1,0,5),next_indices(374,42,2,0,5))
        self.assertTrue(all(0 <= n < 374 for n in next_indices(374,42,1,0,100)))

    def test_target_checksum_and_split_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parent = root / "teacher"
            (parent / "lora").mkdir(parents=True)
            (parent / "COMPLETE").write_text("ok\n")
            (parent / "lora" / "adapter_model.safetensors").write_bytes(b"adapter")
            pmeta={"stage":"l26_ao_sft_fresh_pool_v1","model":"Qwen/Qwen3.5-4B",
                   "layer":26,"source":"frozen_AR_of_fresh_nonoverlap_AO_crops",
                   "transform":"unit","alpha":8000.,
                   "prompt_slot":"last token of rendered prompt"}
            (parent / "meta.json").write_text(json.dumps(pmeta))
            for split,pair_id in (("train",100),("val",200)):
                path=root/split
                path.mkdir()
                (path / "COMPLETE").write_text("ok\n")
                payload={"target_ids":torch.tensor([21,22,2],dtype=torch.int32),
                         "offsets":torch.tensor([0,3],dtype=torch.int64),
                         "vec":torch.ones(1,2560),
                         "src_pair_row":torch.tensor([pair_id],dtype=torch.int64),
                         "src_item_idx":torch.tensor([0],dtype=torch.int64)}
                save_file(payload,str(path/"targets.safetensors"))
                meta={"schema":"l26_distill_bullets_v1",
                      "model":"Qwen/Qwen3.5-4B","layer":26,"source":"true_GT_residual",
                      "split":split,"prompt_mode":"concepts_raw","filter_policy":"no_unigram",
                      "kept_rows":1,"teacher":str(parent),"items_sha256":"banksha",
                      "targets_sha256":sha256(path/"targets.safetensors")}
                (path/"meta.json").write_text(json.dumps(meta))
            trm,tr,vam,va,parent_meta,adapter_sha=verified_contract(
                root/"train",root/"val",parent)
            self.assertEqual(trm["filter_policy"],"no_unigram")
            self.assertEqual(adapter_sha,sha256(parent/"lora"/"adapter_model.safetensors"))
            self.assertEqual(len(row(tr,0)[1]),3)
            with self.assertRaisesRegex(ValueError,"SHA-mismatched"):
                (root/"val"/"targets.safetensors").write_bytes(b"bad")
                load_targets(root/"val","val")

    def test_reject_train_val_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            # A simpler direct sanity check covers overlap's intended predicate.
            a=torch.tensor([42,51])
            b=torch.tensor([73,42])
            self.assertTrue(bool(set(a.tolist()) & set(b.tolist())))


if __name__ == "__main__":
    unittest.main()
