import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from oracle.distill_expansion import digest, verify_expansion, verify_targets
from oracle.combine_distill_targets import combine


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'base'
        self.exp = self.root / 'exp'
        self.old = self.raw({'train':[1,2], 'val':[3]})
        self.new = self.raw({'train':[1,2,4,5], 'val':[3], 'test':[6]})
        self.bsha = self.save_bank(self.base, self.old)
        self.lineage = {'base_items':str(self.base), 'base_sha256':self.bsha}
        self.esha = self.save_bank(self.exp, self.new, self.lineage)

    def raw(self, splits):
        out = {}
        for s, rows in splits.items():
            out.update({s+'_vec':torch.tensor(rows).float()[:,None].repeat(1,2560),
                s+'_ids':torch.tensor(rows)[:,None], s+'_lengths':torch.ones(len(rows),dtype=torch.int32),
                s+'_pair_rows':torch.tensor(rows), s+'_shards':torch.zeros(len(rows),dtype=torch.int32),
                s+'_rollouts':torch.tensor(rows)})
        return out

    def save_bank(self, path, data, lineage=None):
        path.mkdir(exist_ok=True)
        save_file(data,str(path/'items.safetensors'))
        sha=digest(path/'items.safetensors')
        (path/'meta.json').write_text(json.dumps({'items_sha256':sha,'expansion':lineage}))
        (path/'COMPLETE').touch()
        return sha

    def targets(self, path, raw, idx, sha):
        path.mkdir()
        data={'vec':raw['train_vec'][idx], 'src_pair_row':raw['train_pair_rows'][idx],
              'src_item_idx':torch.tensor(idx), 'target_ids':torch.tensor([11,12]*len(idx)),
              'offsets':torch.arange(len(idx)+1)*2}
        save_file(data,str(path/'targets.safetensors'))
        meta={'schema':'l26_distill_bullets_v1','source':'true_GT_residual','prompt_mode':'concepts_raw',
              'split':'train','layer':26,'model':'test','teacher':'parent','ar_checkpoint':'ar',
              'filter_policy':'no_unigram','min_picks':4,'kept_rows':len(idx),'input_rows':len(idx),
              'items_sha256':sha,'targets_sha256':digest(path/'targets.safetensors')}
        (path/'meta.json').write_text(json.dumps(meta)); (path/'COMPLETE').touch()
        return data

    def test_fixed_validation_and_prefix(self):
        verify_expansion(self.exp,self.esha,self.bsha)
        self.new['val_vec'][0,0] += 1
        sha=self.save_bank(self.exp,self.new,self.lineage)
        with self.assertRaisesRegex(ValueError,'changed'):
            verify_expansion(self.exp,sha,self.bsha)

    def test_rollout_leak_even_different_pair(self):
        self.new['train_rollouts'][-1]=3
        sha=self.save_bank(self.exp,self.new,self.lineage)
        with self.assertRaisesRegex(ValueError,'leakage'):
            verify_expansion(self.exp,sha,self.bsha)

    def test_hash_and_wrong_lineage(self):
        with self.assertRaises(ValueError): verify_expansion(self.exp,'wrong',self.bsha)
        with self.assertRaises(ValueError): verify_expansion(self.exp,self.esha,'wrong')
        (self.base/'items.safetensors').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): verify_expansion(self.exp,self.esha,self.bsha)

    def test_combine_preserves_packed_targets(self):
        old=self.root/'old'; fresh=self.root/'fresh'; out=self.root/'out'
        self.targets(old,self.old,[0],self.bsha)
        self.targets(fresh,self.new,[2,3],self.esha)
        meta=combine(old,fresh,self.exp,out)
        self.assertEqual(meta['kept_rows'],3)
        from oracle.train_distill_student import load_targets
        _, data=load_targets(out,'train')
        self.assertEqual(data['src_item_idx'].tolist(),[0,2,3])
        self.assertEqual(data['offsets'].tolist(),[0,2,4,6])
        verify_targets(meta,data,self.new,'train')
        data['vec'][0,0]+=1
        with self.assertRaises(ValueError): verify_targets(meta,data,self.new,'train')

    def test_reject_fresh_prefix(self):
        old=self.root/'old'; fresh=self.root/'fresh'
        self.targets(old,self.old,[0],self.bsha)
        self.targets(fresh,self.new,[1,2],self.esha)
        with self.assertRaisesRegex(ValueError,'prefix'):
            combine(old,fresh,self.exp,self.root/'out')

    def test_trainer_accepts_verified_expansion_and_rejects_changed_val(self):
        from oracle.train_distill_student import verified_contract
        old=self.root/'old'; fresh=self.root/'fresh'; out=self.root/'out'
        self.targets(old,self.old,[0],self.bsha)
        self.targets(fresh,self.new,[2],self.esha)
        combine(old,fresh,self.exp,out)
        val=self.root/'val'
        raw={'train_vec':self.old['val_vec'], 'train_pair_rows':self.old['val_pair_rows']}
        self.targets(val,raw,[0],self.bsha)
        parent=self.root/'parent'; (parent/'lora').mkdir(parents=True)
        (parent/'COMPLETE').touch()
        (parent/'lora'/'adapter_model.safetensors').write_bytes(b'fake')
        (parent/'meta.json').write_text(json.dumps({'stage':'l26_ao_sft_fresh_pool_v1',
          'layer':26,'model':'test','source':'frozen_AR_of_fresh_nonoverlap_AO_crops',
          'transform':'unit','alpha':8000,'prompt_slot':'last token of rendered prompt'}))
        for path in (val,out):
            meta=json.loads((path/'meta.json').read_text()); meta['teacher']=str(parent)
            if path==val: meta['split']='val'
            (path/'meta.json').write_text(json.dumps(meta))
        verified_contract(out,val,parent)
        from safetensors.torch import load_file
        data=load_file(str(val/'targets.safetensors')); data['vec'][0,0]+=1
        save_file(data,str(val/'targets.safetensors'))
        meta=json.loads((val/'meta.json').read_text())
        meta['targets_sha256']=digest(val/'targets.safetensors')
        (val/'meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'source mapping'):
            verified_contract(out,val,parent)

    def test_selector_excludes_whole_rollout_and_is_deterministic(self):
        from types import SimpleNamespace
        from oracle.prepare_distill_items import choose_items
        corpus=SimpleNamespace(val_indices=torch.arange(100),
            shard_ids=torch.zeros(100,dtype=torch.long),rollout_ids=torch.arange(100)//2,
            span_offsets=torch.arange(101)*2,span_token_ids=torch.ones(200,dtype=torch.long),
            activations=torch.ones(100,2))
        tok=SimpleNamespace(decode=lambda *a,**k:'a useful diverse passage')
        excluded={(0,0),(0,1)}
        a,_=choose_items(corpus,tok,42,10,3,exclude=excluded)
        b,_=choose_items(corpus,tok,42,10,3,exclude=excluded)
        self.assertEqual(a,b)
        self.assertFalse({(x[1],x[2]) for rows in a.values() for x in rows}&excluded)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            choose_items(corpus,tok,42,100,3,exclude=excluded)
