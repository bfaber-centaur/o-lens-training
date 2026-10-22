"""CPU-only controls for the 12-activation L26 lens workout."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from oracle.lens_workout import (
    select_rows, shuffle_plan, bullet_stats, trim_eos, injected_embeddings,
    decode, check_lineage, sha256, markdown_report, html_report
)


class TinyModel:
    def __init__(self):
        self.embedding=torch.nn.Embedding(100,4)
        torch.nn.init.zeros_(self.embedding.weight)
        self.saved_embeds=None

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.saved_embeds=kwargs["inputs_embeds"]
        return torch.tensor([[42,2]])


class TinyTokenizer:
    eos_token_id=2
    pad_token_id=0

    def decode(self, ids, skip_special_tokens=True):
        return "decoded " + " ".join(str(i) for i in ids)


class WorkoutTests(unittest.TestCase):
    def test_stratification_deterministic_and_order_invariant(self):
        items=[{"item_idx":i,"omp_fve":i/100,
                "omp_sel":[{"text":"some text"}]*4} for i in range(30)]
        sel={"schema":"l26_nnomp_smoke_v1","split":"val","layer":26,"rows":items}
        bank={"src_item_idx":torch.tensor([i for i in range(30)])}
        a=select_rows(sel,bank,per_bin=4,seed=42)
        b=select_rows({**sel,"rows":items[::-1]},bank,per_bin=4,seed=42)
        self.assertEqual(a,b)
        self.assertEqual(len(a),12)
        self.assertEqual([x["stratum"] for x in a],
                         ["low"]*4+["mid"]*4+["high"]*4)
        self.assertTrue(all(a[i]["item_idx"] < 10 for i in range(4)))
        self.assertTrue(all(10<=a[i]["item_idx"]<20 for i in range(4,8)))
        self.assertTrue(all(a[i]["item_idx"]>=20 for i in range(8,12)))

    def test_reject_insufficient_bin_or_corrupt_fve(self):
        items=[{"item_idx":i,"omp_fve":i/100,
                "omp_sel":[{"text":"ok"}]*4} for i in range(11)]
        sel={"schema":"l26_nnomp_smoke_v1","split":"val","layer":26,"rows":items}
        bank={"src_item_idx":torch.arange(11)}
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            select_rows(sel,bank,4)
        items[0]["omp_fve"]=float("nan")
        with self.assertRaisesRegex(ValueError,"Missing finite"):
            select_rows(sel,bank,1)

    def test_derangement_never_self_maps(self):
        for n in (2,3,12):
            order=shuffle_plan(n,42)
            self.assertEqual(sorted(order),list(range(n)))
            self.assertTrue(all(i!=other for i,other in enumerate(order)))

    def test_injection_only_changes_slot_and_is_normalized(self):
        model=TinyModel()
        ids=torch.tensor([1,3,5])
        vec=torch.tensor([3.,4.,0.,0.])
        pe=injected_embeddings(model,ids,1,vec,10.)
        self.assertTrue(torch.allclose(pe[0,1],torch.tensor([6.,8.,0.,0.])))
        self.assertTrue(torch.equal(pe[0,0],model.embedding(ids[:1])[0]))
        self.assertTrue(torch.equal(pe[0,2],model.embedding(ids[2:3])[0]))
        with self.assertRaisesRegex(ValueError,"zero residual"):
            injected_embeddings(model,ids,1,torch.zeros(4),10.)

    def test_generation_output_eos_and_structural_metrics(self):
        model=TinyModel()
        result=decode(model,TinyTokenizer(),torch.tensor([1,3]),1,
                      torch.tensor([1.,0.,0.,0.]),10.,20)
        self.assertEqual(result["ids"],[42])
        self.assertTrue(result["eos"])
        self.assertFalse(result["hit_cap"])
        self.assertEqual(result["structure"]["bullet_count"],0)
        self.assertEqual(trim_eos([8,2,9],2),([8],True))
        self.assertEqual(bullet_stats("- alpha\n- beta\n- gamma\n- delta")["four_bullets"],True)

    def test_html_escapes_untrusted_generated_text(self):
        report_row={"item_idx":1,"stratum":"low","fve":0.25,
                    "src_pair_row":22,"shuffled_from_item_idx":8,
                    "target":"<script>bad()</script>","outputs":{}}
        for arm in ("parent_shared","student_shared","student_shuffled"):
            report_row["outputs"][arm]={"text":"<img src=x onerror=alert(1)>",
                "structure":{"bullet_count":0},"tokens":5,"eos":True}
        rendered=html_report([report_row])
        self.assertNotIn("<script>bad()",rendered)
        self.assertIn("&lt;script&gt;bad()",rendered)
        self.assertNotIn("<img src=x",rendered)
        self.assertIn("&lt;img src=x",rendered)
        manifest={"alpha":8000.,"max_new_tokens":160}
        self.assertIn("NNOMP bullet target",markdown_report(manifest,[report_row]))

    def test_lineage_rejects_tampered_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            selection=root/"selection"
            selection.mkdir()
            (selection/"COMPLETE").write_text("ok")
            data={"schema":"l26_nnomp_smoke_v1","split":"val",
                  "layer":26,"source":"true_GT_residual",
                  "items_sha256":"orig","rows":[]}
            (selection/"report.json").write_text(json.dumps(data))
            val=root/"val"
            val.mkdir()
            (val/"COMPLETE").write_text("ok")
            save_file({"target_ids":torch.tensor([5,2],dtype=torch.int32),
                       "offsets":torch.tensor([0,2],dtype=torch.int64),
                       "vec":torch.ones(1,2560),
                       "src_pair_row":torch.tensor([8],dtype=torch.int64),
                       "src_item_idx":torch.tensor([0],dtype=torch.int64)},
                      str(val/"targets.safetensors"))
            vm={"schema":"l26_distill_bullets_v1",
                "source":"true_GT_residual","prompt_mode":"concepts_raw",
                "split":"val","layer":26,"filter_policy":"no_unigram",
                "model":"Qwen/Qwen3.5-4B","kept_rows":1,
                "targets_sha256":sha256(val/"targets.safetensors"),
                "selection_sha256":sha256(selection/"report.json"),
                "items_sha256":"orig"}
            (val/"meta.json").write_text(json.dumps(vm))
            parent=root/"parent"
            (parent/"lora").mkdir(parents=True)
            (parent/"COMPLETE").write_text("ok")
            (parent/"lora"/"adapter_model.safetensors").write_bytes(b"parent")
            (parent/"meta.json").write_text(json.dumps({
                "stage":"l26_ao_sft_fresh_pool_v1","model":"Qwen/Qwen3.5-4B",
                "layer":26,"alpha":8000,"transform":"unit",
                "prompt_slot":"last token of rendered prompt"}))
            student=root/"student"/"checkpoints"/"step_000050"
            (student/"lora").mkdir(parents=True)
            (student/"COMPLETE").write_text("ok")
            (student/"lora"/"adapter_model.safetensors").write_bytes(b"student")
            sm={"stage":"l26_distill_student_sft_v1",
                "source":"true_GT_residual","model":"Qwen/Qwen3.5-4B",
                "layer":26,"transform":"unit","alpha":8000,
                "prompt_mode":"concepts_raw","init_from":str(parent),
                "init_adapter_sha256":sha256(parent/"lora"/"adapter_model.safetensors"),
                "val":str(val),"val_targets_sha256":vm["targets_sha256"],
                "val_filter_policy":"no_unigram"}
            (student.parent.parent/"meta.json").write_text(json.dumps(sm))
            check_lineage(selection,val,parent,student)
            (selection/"report.json").write_text(json.dumps({**data,"rows":[{}]}))
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                check_lineage(selection,val,parent,student)


if __name__=="__main__":
    unittest.main()
