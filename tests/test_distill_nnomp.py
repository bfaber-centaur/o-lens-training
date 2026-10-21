"""CPU controls for L26 distillation's Agam-style within-item NNOMP."""
import unittest

import torch

from oracle.select_distill_nnomp import candidates, nnls, nnomp


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"token{t}" for t in ids)


class NNOMPTests(unittest.TestCase):
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

    def test_nonfinite_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            nnomp(torch.tensor([float("nan"),0.]),torch.eye(2))


if __name__=="__main__":
    unittest.main()
