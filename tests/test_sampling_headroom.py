import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "oracle/l26_sampling_headroom.py"
spec = importlib.util.spec_from_file_location("headroom", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class HeadroomTests(unittest.TestCase):
    def test_nested_means_and_best_excludes_greedy(self):
        rows = []
        for item, scale in enumerate((1, 2)):
            samples = {arm: dict(fve=scale*j/100, complementarity=j/1000,
                                valid=j != 0, hit_cap=j == 7)
                       for j, arm in enumerate(m.ARMS)}
            samples["student_shared"] = dict(fve=.9, complementarity=.1)
            rows.append(dict(item_idx=item, stratum="low", scores=samples))
        result = m.aggregate(rows)
        self.assertAlmostEqual(result["means"]["sample_mean_fve"], .0525)
        self.assertAlmostEqual(result["means"]["best_of_8_fve"], .105)
        self.assertAlmostEqual(result["means"]["valid_fraction"], 7/8)
        self.assertEqual(result["items"][0]["best_arm"], m.ARMS[-1])

    def test_ties_and_empty(self):
        scores = {arm: dict(fve=0., complementarity=0., valid=False, hit_cap=False)
                  for arm in m.ARMS}
        scores["student_shared"] = dict(fve=0., complementarity=0.)
        self.assertEqual(m.aggregate([dict(item_idx=1, stratum="low", scores=scores)])
                         ["items"][0]["best_arm"], m.ARMS[0])
        with self.assertRaises(ValueError):
            m.aggregate([])

if __name__ == "__main__":
    unittest.main()
