"""CPU checks for Agam-style distillation bullet target assembly."""
import unittest
from types import SimpleNamespace

import torch

from oracle.assemble_distill_bullets import (
    reject_reason, assemble, ngram_loop, keys,
)


class DummyTokenizer:
    eos_token_id = 2
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[100+ord(ch)%37 for ch in text])


class BulletAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.good = [
            "Graph structure conveys directional relationships.",
            "The satellite approaches orbital apogee.",
            "Metabolic pathways are regulated by enzymes.",
            "Musical intervals form harmonic patterns.",
        ]
    def test_accepts_diverse_four(self):
        self.assertEqual(reject_reason(self.good,100),"")
    def test_rejects_near_duplication(self):
        p=list(self.good)
        p[1]="Graph structure conveys directional relationships through a network."
        self.assertEqual(reject_reason(p,100),"jaccard>=0.45")
    def test_rejects_prefix_redundancy(self):
        p=list(self.good)
        p[1]=p[0]+" Additional words follow."
        self.assertEqual(reject_reason(p,100),"prefix_redundant")
    def test_rejects_loop_or_bullet_marker(self):
        p=list(self.good)
        p[1]="word word word word"
        self.assertEqual(reject_reason(p,100),"ngram_loop")
        p[1]="- A bullet inside a bullet."
        self.assertEqual(reject_reason(p,100),"bullet_collision")
    def test_no_unigram_policy_ignores_function_word_frequency(self):
        p=list(self.good)
        p[0]="A traveler saw a mountain beyond a forest."
        self.assertEqual(reject_reason(p,100),"ngram_loop")
        self.assertEqual(reject_reason(p,100,filter_policy="no_unigram"),"")
        self.assertTrue(ngram_loop(p[0]))
        self.assertFalse(ngram_loop(p[0],min_n=2))

    def test_no_unigram_still_rejects_actual_multiword_loop(self):
        p=list(self.good)
        p[0]="red green red green red green"
        self.assertEqual(reject_reason(p,100,filter_policy="no_unigram"),
                         "ngram_loop")

    def test_no_unigram_preserves_other_quality_filters(self):
        p=list(self.good)
        p[0]="A traveler saw a mountain beyond a forest."
        self.assertEqual(reject_reason(p[:3],100,filter_policy="no_unigram"),
                         "n_sel!=4")
        p[1]="- An embedded bullet."
        self.assertEqual(reject_reason(p,100,filter_policy="no_unigram"),
                         "bullet_collision")
        with self.assertRaisesRegex(ValueError,"Unknown assembly filter policy"):
            reject_reason(p,100,filter_policy="unknown")

    def test_alternative_assembly_audits_upstream_counterfactual(self):
        picks=list(self.good)
        picks[0]="A traveler saw a mountain beyond a forest."
        selection={"split":"train","rows":[{
            "item_idx":0,"n_candidates":100,
            "omp_sel":[{"len":16+i*8,"text":t} for i,t in enumerate(picks)]
        }]}
        bank={"train_vec":torch.ones(1,4),
              "train_pair_rows":torch.tensor([777])}
        strict,strict_audit=assemble(selection,bank,DummyTokenizer())
        relaxed,relaxed_audit=assemble(selection,bank,DummyTokenizer(),
                                      filter_policy="no_unigram")
        self.assertEqual(len(strict),0)
        self.assertEqual(strict_audit[0]["reason"],"ngram_loop")
        self.assertEqual(len(relaxed),1)
        self.assertEqual(relaxed_audit[0]["reason"],"accepted")
        self.assertEqual(relaxed_audit[0]["upstream_reason"],"ngram_loop")
        self.assertTrue(torch.equal(relaxed[0]["vec"],bank["train_vec"][0]))
        self.assertEqual(relaxed[0]["src_pair_row"],777)

    def test_requires_four_and_sufficient_candidates(self):
        self.assertEqual(reject_reason(self.good,24),"n_cand<25")
        self.assertEqual(reject_reason(self.good[:3],100),"n_sel!=4")
    def test_roundtrip_true_activation_and_shortest_first(self):
        selection={"split":"train","rows":[{
            "item_idx":0,"n_candidates":120,
            "omp_sel":[{"len":64,"text":self.good[0]},
                       {"len":32,"text":self.good[1]},
                       {"len":16,"text":self.good[2]},
                       {"len":8,"text":self.good[3]}]}]}
        bank={"train_vec":torch.tensor([[3.,4.,5.]]),
              "train_pair_rows":torch.tensor([901])}
        rows,audit=assemble(selection,bank,DummyTokenizer())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["src_pair_row"],901)
        self.assertTrue(torch.equal(rows[0]["vec"],bank["train_vec"][0]))
        self.assertTrue(rows[0]["text"].startswith("- Musical intervals"))
        self.assertEqual(rows[0]["target_ids"][-1],2)
        self.assertEqual(audit[0]["reason"],"accepted")
    def test_out_of_range_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"out-of-range"):
            assemble({"split":"val","rows":[{"item_idx":5,"n_candidates":0}]},
                     {"val_vec":torch.ones(1,4)},DummyTokenizer())


if __name__=="__main__":
    unittest.main()
