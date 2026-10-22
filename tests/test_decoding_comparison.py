"""CPU-only paired decoding controls, stop handling, seeds and aggregation."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from oracle.compare_decoding import (PROFILES, CONDITIONS, fifth_marker, StopFour,
    stable_seed, schedule, decode, row_hash, load_row, summary_rows)
from oracle.score_generated_readouts import atomic_json


class DecodeTests(unittest.TestCase):
    def test_schedule_complete_deterministic(self):
        self.assertEqual(schedule(52),schedule(52))
        self.assertEqual(set(schedule(52)),{(p,c) for p in PROFILES for c in CONDITIONS})
        self.assertEqual(len(schedule(52)),14)
        self.assertNotEqual(schedule(52),schedule(51))
        self.assertEqual(stable_seed(52,'sample070_s11'),stable_seed(52,'sample070_s11'))
        self.assertNotEqual(stable_seed(52,'sample070_s11'),stable_seed(51,'sample070_s11'))

    def test_stop_waits_for_fifth_marker_and_preserves_multiline(self):
        four = '- one\ncontinued\n- two\n- three\n- four'
        self.assertFalse(fifth_marker(four))
        self.assertTrue(fifth_marker(four+'\n-'))
        self.assertFalse(fifth_marker(four+'\nnot - a bullet'))
        self.assertTrue(fifth_marker(four+'\n  - fifth'))

    def test_generation_flags_eos_and_cap(self):
        class Tok:
            eos_token_id=9
            pad_token_id=0
            def decode(self,ids,**kw): return ' '.join(map(str,ids))
        class Model:
            def generate(self,**kwargs):
                self.kw = kwargs
                return torch.tensor([[1,2,9]])
        model = Model()
        with patch('oracle.compare_decoding.injected_embeddings',return_value=torch.zeros(1,5,2)):
            result = decode(model,Tok(),torch.ones(5),2,torch.ones(2),8000,'rep110',123)
        self.assertEqual(result['ids'],[1,2])
        self.assertTrue(result['eos'])
        self.assertEqual(result['generated_tokens_including_eos'],3)
        self.assertFalse(result['hit_cap'])
        self.assertEqual(model.kw['repetition_penalty'],1.1)
        self.assertFalse(model.kw['do_sample'])
        with patch('oracle.compare_decoding.injected_embeddings',return_value=torch.zeros(1,5,2)):
            decode(model,Tok(),torch.ones(5),2,torch.ones(2),8000,'sample070_s11',123)
        self.assertEqual(model.kw['top_k'],0)
        self.assertEqual(model.kw['top_p'],.9)
        self.assertEqual(model.kw['temperature'],.7)
        self.assertTrue(model.kw['do_sample'])

    def test_stop_criterion_batch_and_short_cap(self):
        class Tok:
            eos_token_id=9
            pad_token_id=0
            def decode(self,ids,**kw):
                return '\n'.join('- word' for _ in ids)
        stop=StopFour(Tok())
        self.assertFalse(stop(torch.ones(1,4,dtype=torch.long),None).item())
        self.assertTrue(stop(torch.ones(1,5,dtype=torch.long),None).item())
        class Model:
            def generate(self,**kw):
                self.kw=kw
                return torch.ones(1,kw['max_new_tokens'],dtype=torch.long)
        model=Model()
        with patch('oracle.compare_decoding.injected_embeddings',return_value=torch.zeros(1,5,2)):
            result=decode(model,Tok(),torch.ones(5),2,torch.ones(2),8000,'greedy64',123)
        self.assertEqual(result['tokens'],64)
        self.assertTrue(result['hit_cap'])
        self.assertFalse(result['eos'])
        self.assertEqual(result['stop_reason'],'token_cap')

    def test_row_resume_preserves_original_and_detects_tamper(self):
        original={'item_idx':1,'target':'- two words','outputs':{'student_shared':{'ids':[1,2]}}}
        r=copy.deepcopy(original)
        r['outputs']['decode_test_correct']={'ids':[3,4]}
        r['comparison_contract']='fp'
        r['comparison_row_sha256']=row_hash(r)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'item.json'
            atomic_json(p,r)
            load_row(p,'fp',original,list(r['outputs']))
            with self.assertRaisesRegex(ValueError,'contract'):
                load_row(p,'bad',original,list(r['outputs']))
            r['outputs']['student_shared']['ids']=[8]
            atomic_json(p,r)
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_row(p,'fp',original,list(r['outputs']))

    def test_sampling_averaged_within_item_not_best_selected(self):
        rows,generated=[],{}
        for idx in range(2):
            scores,outputs={},{}
            for name in PROFILES:
                for condition in CONDITIONS:
                    arm=f'decode_{name}_{condition}'
                    fve={'sample070_s11':0.,'sample070_s22':.3,'sample070_s33':.6}.get(name,.1)
                    scores[arm]={'fve':fve,'first_bullet_fve':0.,'valid':True,
                                 'parser':{'unique_text_count':1,'scored_count':4}}
                    outputs[arm]={'generated_tokens_including_eos':100,'hit_cap':True}
            rows.append({'item_idx':idx,'stratum':'low','scores':scores})
            generated[idx]={'outputs':outputs}
        result=summary_rows(rows,generated)
        s=next(r for r in result if r['stratum']=='all' and r['profile']=='sample070' and r['condition']=='correct')
        self.assertEqual(s['n_items'],2)
        self.assertEqual(s['samples_per_item'],3)
        self.assertAlmostEqual(s['fve'],.3)
        self.assertAlmostEqual(s['delta_greedy'],.2)
        self.assertEqual(len(s['per_item']),2)


if __name__=='__main__': unittest.main()
