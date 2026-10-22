"""CPU controls for L26 distillation's Agam-style within-item NNOMP."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from oracle.select_distill_nnomp import (candidates, nnls, nnomp,
                                         selected_bounds, atomic_json, strict_filter_counts)


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        # Deterministic varied strings: repeating "token" triggers the real
        # F1 character-4-gram loop filter and invalidates this dedup test.
        return " ".join(hashlib.blake2b(str(t).encode(), digest_size=8).hexdigest() for t in ids)


class NNOMPTests(unittest.TestCase):
    def test_selection_bounds_within_teacher_manifest(self):
        manifest={"start":20,"count":100}
        self.assertEqual(selected_bounds(manifest),(20,120))
        self.assertEqual(selected_bounds(manifest,32,40),(32,72))
        self.assertEqual(selected_bounds(manifest,117,None),(117,120))
        for start,count in ((19,1),(118,3),(20,0),(121,None)):
            with self.subTest(start=start,count=count):
                with self.assertRaises(ValueError):
                    selected_bounds(manifest,start,count)

    def test_atomic_checkpoint_write_and_partial_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"item_000123.json"
            atomic_json(file,{"item_idx":123,"n_candidates":30})
            self.assertEqual(json.loads(file.read_text())["item_idx"],123)
            atomic_json(file,{"item_idx":123,"n_candidates":31})
            self.assertEqual(json.loads(file.read_text())["n_candidates"],31)
            self.assertFalse((Path(directory)/"item_000123.json.partial").exists())
            (Path(directory)/"item_000123.json.partial").write_text("interrupted")
            with self.assertRaisesRegex(RuntimeError,"interrupted write"):
                atomic_json(file,{"item_idx":123,"n_candidates":32})
            self.assertEqual(json.loads(file.read_text())["n_candidates"],31)

    def test_strict_audit_uses_original_assembly_rule(self):
        accepted=["orchid violet","blue orchard","lunar marble","quiet harbor"]
        repeated=["a mouse saw a bird beside a tree","blue orchard",
                  "lunar marble","quiet harbor"]
        rows=[{"n_candidates":30,"omp_sel":[{"text":text} for text in accepted]},
              {"n_candidates":30,"omp_sel":[{"text":text} for text in repeated]},
              {"n_candidates":4,"omp_sel":[]}]
        self.assertEqual(strict_filter_counts(rows),
                         {"accepted":1,"n_cand<25":1,"ngram_loop":1})

    def test_nonnegative_orthogonal_components(self):
        dictionary=torch.eye(4,dtype=torch.float32)
        x=torch.tensor([2.,1.,0.,0.],dtype=torch.float32)
        sel,coef,fve=nnomp(x,dictionary,k=4)
        self.assertEqual(sel,[0,1])
        self.assertAlmostEqual(coef[0],2,places=3)
        self.assertAlmostEqual(coef[1],1,places=3)
        self.assertAlmostEqual(fve,1,places=4)

    def test_negative_direction_not_selected(self):
        d=torch.tensor([[-1.,0.],[0.,1.]],dtype=torch.float32)
        sel,coef,fve=nnomp(torch.tensor([1.,0.]),d,k=2)
        self.assertEqual(sel,[])
        self.assertEqual(coef,[])
        self.assertEqual(fve,0.)

    def test_nnls_nonnegative(self):
        directions=torch.eye(2)
        w,fve=nnls(directions,torch.tensor([2.,-1.]))
        self.assertTrue(bool((w>=0).all()))
        self.assertAlmostEqual(float(w[1]),0,places=4)

    def test_candidate_dedup_by_token_prefix(self):
        ids=list(range(1,100))
        samples=[{"ids":ids},{"ids":ids}]
        c=candidates(samples,FakeTokenizer())
        self.assertEqual(len(c),6)
        self.assertEqual([r["len"] for r in c],[2,4,8,16,32,64])
        self.assertTrue(all(row["src"]==0 for row in c))

    def test_repetitive_synthetic_text_is_filtered(self):
        class RepetitiveTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(f"token{t}" for t in ids)

        ids = list(range(1, 100))
        c = candidates([{"ids": ids}], RepetitiveTokenizer())
        self.assertEqual([r["len"] for r in c], [2])

    def test_nonfinite_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            nnomp(torch.tensor([float("nan"),0.]),torch.eye(2))


if __name__=="__main__":
    unittest.main()
