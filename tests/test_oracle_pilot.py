"""CPU-level contract regression tests for the bounded oracle pilot."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[1] / "oracle" / "train_gt_sft.py"
spec = importlib.util.spec_from_file_location("oracle_pilot", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 4)

    def get_input_embeddings(self):
        return self.emb


class OracleContractTests(unittest.TestCase):
    def test_split_deterministic_and_rollout_pure(self):
        self.assertEqual(mod.oracle_partition(1, 7, 42, 0.2), mod.oracle_partition(1, 7, 42, 0.2))
        self.assertIn(mod.oracle_partition(1, 7, 42, 0.2), {"train", "val"})

    def test_injection_replaces_last_prompt_embedding_only(self):
        model = FakeModel()
        prompt = torch.tensor([2, 3, 4])
        v = torch.tensor([[3., 4., 0., 0.]])
        ids = torch.tensor([[5, 6]])
        labels = ids.clone()
        mask = torch.ones_like(ids)
        emb, attn, y = mod.conditioned_batch(model, prompt, v, ids, labels, mask, 10.)
        self.assertEqual(tuple(emb.shape), (1, 5, 4))
        self.assertTrue(torch.allclose(emb[0, :2], model.emb(prompt[:2])))
        self.assertTrue(torch.allclose(emb[0, 2].float(), torch.tensor([6., 8., 0., 0.])))
        self.assertTrue(torch.equal(y[0], torch.tensor([-100, -100, -100, 5, 6])))
        self.assertTrue(torch.equal(attn, torch.ones(1, 5, dtype=torch.long)))

    def test_variable_length_masked_labels(self):
        rows = [(torch.ones(4), torch.tensor([1, 2, 3])),
                (torch.ones(4) * 2, torch.tensor([4]))]
        _, ids, labels, mask = mod.collate(rows, 0, 9)
        self.assertEqual(ids.tolist(), [[1, 2, 3, 9], [4, 9, 0, 0]])
        self.assertEqual(labels.tolist(), [[1, 2, 3, 9], [4, 9, -100, -100]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
