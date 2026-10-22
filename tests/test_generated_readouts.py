"""CPU-only scoring controls: no model downloads."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from oracle.score_generated_readouts import (
    parse_bullets as parse, score_vectors, geometry, atomic_json, saved_row, write_row,
    run, verify_texts, validate_inputs,
)


def parse_bullets(text):
    return parse(text,lambda b: list(''.join(b.split())))


class ScoringTests(unittest.TestCase):
    def score(self, dirs, target):
        return score_vectors(torch.tensor(dirs,dtype=torch.float32),
            torch.tensor(target,dtype=torch.float32),torch.zeros(len(target)),torch.eye(len(target)))

    def test_multiline_filter_cap_duplicate(self):
        p = parse_bullets('preamble\n  - alpha\n  continuation\n\nnext\n- x\n- \n- beta\n- beta\n- gamma\n- delta')
        self.assertEqual(p['bullets'],['alpha\ncontinuation\n\nnext','beta','beta','gamma'])
        self.assertEqual(p['parsed_count'],7)
        self.assertEqual(p['dropped_short'],2)
        self.assertEqual(p['beyond_cap'],1)
        self.assertEqual(p['unique_text_count'],3)

    def test_no_prose_fallback_and_cjk(self):
        self.assertEqual(parse_bullets('a plain paragraph')['status'],'no_valid_bullets')
        self.assertEqual(parse_bullets('- 电荷\n- 静电')['scored_count'],2)
        self.assertEqual(parse_bullets('1. alpha\n* beta')['scored_count'],0)

    def test_minimum_uses_ar_tokens_not_characters(self):
        parsed = parse('- longword\n- two words',lambda b: b.split())
        self.assertEqual(parsed['bullets'],['two words'])
        self.assertEqual(parsed['dropped_short'],1)

    def test_orthogonal_known_prefix_and_nonnegative(self):
        s = self.score([[1,0],[0,1]],[2,1])
        self.assertAlmostEqual(s['prefix_fve'][0],.8,places=6)
        self.assertAlmostEqual(s['fve'],1,places=6)
        self.assertAlmostEqual(s['incremental_fve'][1],.2,places=6)
        self.assertEqual(s['direction_rank'],2)
        s = self.score([[1,0],[0,1]],[2,-1])
        self.assertAlmostEqual(s['fve'],.8,places=6)
        self.assertEqual(s['coefficients'][1],0)

    def test_duplicate_and_zero_directions(self):
        s = self.score([[1,0],[1,0],[0,0]],[1,1])
        self.assertAlmostEqual(s['fve'],.5,places=6)
        self.assertAlmostEqual(s['incremental_fve'][1],0,places=6)
        self.assertEqual(s['direction_rank'],1)
        self.assertEqual(s['near_duplicate_pairs_cos_ge_0999'],1)
        self.assertEqual(s['valid_direction_count'],2)
        z = self.score([[0,0]],[1,1])
        self.assertFalse(z['valid'])
        self.assertEqual(z['fve'],0)

    def test_no_bullets_invalid_zero_score(self):
        s = score_vectors(torch.empty(0,2),torch.ones(2),torch.zeros(2),torch.eye(2))
        self.assertEqual(s['fve'],0)
        self.assertFalse(s['valid'])
        self.assertEqual(s['prefix_fve'],[])

    def test_center_whiten_without_target_normalization(self):
        d,t,v = geometry(torch.tensor([[3.,1.]]),torch.tensor([2.,3.]),
                         torch.ones(2),torch.diag(torch.tensor([2.,3.])))
        torch.testing.assert_close(d,torch.tensor([[1.,0.]]))
        torch.testing.assert_close(t,torch.tensor([2.,6.]))
        self.assertTrue(v.item())

    def test_nonfinite_shape_and_zero_target_fail(self):
        for target in ([0.,0.],[float('nan'),1.]):
            with self.assertRaises(ValueError): self.score([[1,0]],target)
        with self.assertRaises(ValueError): self.score([[1,2,3]],[1,1])

    def test_nnls_against_independent_active_set_reference(self):
        dirs = torch.tensor([[1.,0.,0.],[.3,1.,0.],[0.,.2,1.]])
        target = torch.tensor([.7,1.4,-.2])
        s = score_vectors(dirs,target,torch.zeros(3),torch.eye(3))
        dirs = torch.nn.functional.normalize(dirs,dim=1)
        best = float(target.square().sum())
        for mask in range(1,8):
            active = dirs[[j for j in range(3) if mask & (1<<j)]]
            w = torch.linalg.lstsq(active.T,target).solution
            if (w>=0).all(): best = min(best,float((target-w@active).square().sum()))
        self.assertAlmostEqual(s['fve'],1-best/float(target.square().sum()),places=5)

    def test_saved_row_checksum_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'item.json'
            write_row(p,'abc',{'item_idx':7,'value':.5})
            self.assertEqual(saved_row(p,'abc',7)['value'],.5)
            with self.assertRaisesRegex(ValueError,'different run'): saved_row(p,'def',7)
            obj = json.loads(p.read_text())
            obj['result']['value'] = .9
            atomic_json(p,obj)
            with self.assertRaisesRegex(ValueError,'checksum'): saved_row(p,'abc',7)

    def test_stale_temp_does_not_block_atomic_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'x.json'
            (Path(td)/'.x.json.stale.tmp').write_text('partial')
            atomic_json(p,{'ok':True})
            self.assertEqual(json.loads(p.read_text()),{'ok':True})

    def test_verify_text_rejects_mismatched_generation(self):
        class Tok:
            def decode(self,ids,**kw): return str(ids)
        bank = {'offsets':torch.tensor([0,1]),'target_ids':torch.tensor([1])}
        rows = [{'target_row':0,'target':'[1]','outputs':{'x':{'ids':[2],'text':'wrong'}}}]
        with self.assertRaisesRegex(ValueError,'Generated text'): verify_texts(rows,bank,Tok())

    def test_wrong_source_sha_rejected_before_ar_loading(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'COMPLETE').write_text('ok')
            atomic_json(root/'manifest.json',{'schema':'l26_lens_workout_v1','layer':26,
                'source':'true_GT_residual','selection':td,'val':td,'parent':td,'student':td,
                'selection_sha256':'wrong'})
            (root/'report.json').write_text('{}')
            with patch('lens_workout.check_lineage',return_value=({},None,None,None,None,'p','s')):
                with self.assertRaisesRegex(ValueError,'source SHA'): validate_inputs(root,root)

    def test_smoke_resume_and_original_target_control(self):
        class Tok:
            pad_token_id = 0
            def __call__(self,text,**kw):
                return argparse.Namespace(input_ids=[1,1] if text=='east' else [2,2])
        class AR:
            peft_model = argparse.Namespace(get_base_model=lambda: argparse.Namespace(config=argparse.Namespace(_commit_hash='fake')))
            def eval(self): return self
            def requires_grad_(self,x): pass
        def embed(ar, rows, tag, pad, batch):
            return torch.tensor([[1.,0.] if x['ids']==[1,1] else [0.,1.] for x in rows])
        rows = [{'item_idx':i,'target_row':i,'src_pair_row':i+10,'stratum':'low',
                 'shuffled_from_item_idx':1-i,'target':'- east',
                 'outputs':{k:{'text':'- north' if k.endswith('_shuffled') else '- east',
                               'eos':False,'hit_cap':True,'tokens':3}
                            for k in ('parent_shared','student_shared','student_shuffled','decode_rep110_shuffled')}} for i in range(2)]
        bank = {'src_item_idx':torch.tensor([0,1]),'vec':torch.eye(2)}
        sel = {'rows':[{'item_idx':i,'omp_fve':.8} for i in range(2)]}
        args = argparse.Namespace(max_new_items=1,batch_size=4,ar_checkpoint=Path('/fake'))
        with tempfile.TemporaryDirectory() as td, \
             patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_name',return_value='CPU mock'), \
             patch('train_ar.build_ar',return_value=AR()) as build, \
             patch('oracle.score_generated_readouts.embed_candidates',side_effect=embed):
            out = Path(td)
            def invoke():
                run(args,{},bank,sel,rows,torch.zeros(2),torch.eye(2),Tok(),{'model_id':'fake'},out,'fp')
            invoke()
            report = json.loads((out/'report.json').read_text())
            shuffled = report['rows'][0]['scores']['student_shuffled']
            self.assertEqual(shuffled['fve'],0)
            self.assertEqual(shuffled['donor_diagnostic']['fve'],1)
            new_shuffled = report['rows'][0]['scores']['decode_rep110_shuffled']
            self.assertEqual(new_shuffled['fve'],0)
            self.assertEqual(new_shuffled['donor_diagnostic']['fve'],1)
            self.assertFalse((out/'COMPLETE').exists())
            first = (out/'item_000000.json').read_bytes()
            args.max_new_items = None
            invoke()
            self.assertTrue((out/'COMPLETE').exists())
            self.assertEqual(first,(out/'item_000000.json').read_bytes())
            self.assertEqual(build.call_count,2)
            invoke()
            self.assertEqual(build.call_count,2)


if __name__=='__main__':
    unittest.main()
