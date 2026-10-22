import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from legacy.harvest_exclusions import audit_fresh, rollout_inventory, prompt_digest
from legacy.harvest_ar import prompt_stream, run_contract
from oracle.prepare_distill_items import choose_items  # installs legacy import path
from oracle.fresh_distill_corpus import load_fresh


class HarvestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def rollouts(self,name,texts):
        directory=self.root/name; directory.mkdir(parents=True)
        (directory/'rollouts_00000.jsonl').write_text(''.join(json.dumps({'prompt':s})+'\n' for s in texts))
        return directory

    def test_audit_overlap_and_internal_duplicates(self):
        old=self.rollouts('old',['a prompt'])
        fresh=self.rollouts('fresh',['another prompt'])
        self.assertEqual(audit_fresh(old,fresh)['fresh_prompts'],1)
        (fresh/'rollouts_00000.jsonl').write_text(json.dumps({'prompt':'a  prompt'})+'\n')
        with self.assertRaisesRegex(ValueError,'original'): audit_fresh(old,fresh)
        (fresh/'rollouts_00000.jsonl').write_text((json.dumps({'prompt':'other'})+'\n')*2)
        with self.assertRaisesRegex(ValueError,'internally'): audit_fresh(old,fresh)

    def test_missing_prompt_fails(self):
        d=self.rollouts('old',[''])
        with self.assertRaisesRegex(ValueError,'Missing'): rollout_inventory(d)
        with self.assertRaisesRegex(ValueError,'No rollout'): rollout_inventory(self.root/'absent')

    def test_sampling_skips_old_and_normalized_duplicates(self):
        class Dataset(list):
            def shuffle(self,**kwargs): return self
        ds=Dataset({'conversation':[{'role':'user','content':s}]} for s in ['old text','new text','new  text','last text'])
        args=SimpleNamespace(dataset='fake',dataset_split='train',seed=43,shuffle_buffer=100,
            min_prompt_tokens=1,max_prompt_tokens=20,excluded_prompt_digests={prompt_digest('old text')},
            exclusion_hashes={'source':'sha'})
        tok=lambda *a,**k:SimpleNamespace(input_ids=[1,2])
        with patch.dict('sys.modules',{'datasets':SimpleNamespace(load_dataset=lambda *a,**k:ds)}):
            self.assertEqual(list(prompt_stream(args,tok,set())),['new text','last text'])

    def test_contract_binds_exclusions(self):
        fields=dict(model='model',layers=[26],dataset='data',dataset_split='train',shuffle_buffer=1,
            min_prompt_tokens=1,max_prompt_tokens=10,min_response_tokens=8,max_new_tokens=32,
            temperature=1.,top_p=1.,max_spans=1,span_min=1,span_max=32,seed=43)
        args=SimpleNamespace(**fields)
        original=run_contract(args)
        args.exclusion_hashes={'old':'sha'}; args.revision='pinned'
        self.assertNotEqual(original,run_contract(args))
        self.assertEqual(run_contract(args)['exclude_rollout_sha256'],{'old':'sha'})

    def setup_fresh(self):
        olddir=self.rollouts('old',['old'])
        root=self.root/'fresh'; self.rollouts('fresh/rollouts',['fresh'])
        (root/'pairs').mkdir(); (root/'pairs'/'pairs_l26_00000.safetensors').touch()
        _, hashes,_=rollout_inventory(olddir)
        (root/'run_config.json').write_text(json.dumps({'contract':{'model_id':'model','layers':[26],
            'max_spans':1,'exclude_rollout_sha256':hashes}}))
        (root/'environment.json').write_text(json.dumps({'model_revision':'revision'}))
        orig=SimpleNamespace(d_model=2,shard_ids=torch.tensor([9]),activations=torch.ones(100,2),pair_paths=[])
        fresh=SimpleNamespace(d_model=2,shard_ids=torch.tensor([0,1]),activations=torch.ones(2,2),
            is_val=torch.tensor([True,False]),is_train=torch.tensor([False,True]))
        ns=SimpleNamespace(rollout_dir=olddir,pair_dir=self.root/'pairs')
        return root,orig,fresh,ns

    def test_namespace_and_whole_fresh_pool(self):
        root,orig,fresh,ns=self.setup_fresh()
        with patch('oracle.fresh_distill_corpus.validate_completed_shards',return_value=({0:{}},{})), patch('oracle.fresh_distill_corpus.load_corpus',return_value=fresh):
            corpus,audit=load_fresh(root,orig,ns,'model',26,'revision')
        self.assertEqual(corpus.shard_ids.tolist(),[10,11])
        self.assertTrue(corpus.is_val.all())
        self.assertEqual(audit['pair_row_offset'],100)
        self.assertEqual(audit['prompt_overlap'],0)

    def test_reject_revision_or_exclusion_drift(self):
        root,orig,fresh,ns=self.setup_fresh()
        with self.assertRaisesRegex(ValueError,'revision'): load_fresh(root,orig,ns,'model',26,'wrong')
        with (ns.rollout_dir/'rollouts_00000.jsonl').open('a') as f: f.write(json.dumps({'prompt':'additional'})+'\n')
        with patch('oracle.fresh_distill_corpus.validate_completed_shards',return_value=({0:{}},{})):
            with self.assertRaisesRegex(ValueError,'exclusions'): load_fresh(root,orig,ns,'model',26)

    def test_uncommitted_shard_rejected(self):
        root,orig,fresh,ns=self.setup_fresh()
        (root/'rollouts'/'rollouts_00001.jsonl').touch()
        with patch('oracle.fresh_distill_corpus.validate_completed_shards',return_value=({0:{}},{})):
            with self.assertRaisesRegex(ValueError,'Uncommitted'): load_fresh(root,orig,ns,'model',26)

    def test_dry_run_never_touches_cuda_or_output(self):
        import io
        from contextlib import redirect_stdout
        from legacy.harvest_ar import main
        old=self.rollouts('old',['old prompt'])
        output=self.root/'unwritten'
        argv=['harvest_ar.py','--layers','26','--output',str(output),
              '--exclude-rollouts',str(old),'--dry-run']
        with patch('sys.argv',argv), patch('torch.cuda.set_device',side_effect=AssertionError('CUDA touched')), redirect_stdout(io.StringIO()) as capture:
            main()
        self.assertEqual(json.loads(capture.getvalue())['excluded_prompts'],1)
        self.assertFalse(output.exists())

    def test_prepare_fresh_bank_preserves_old_and_offsets_new_rows(self):
        import io
        from contextlib import redirect_stdout
        from safetensors.torch import save_file, load_file
        from oracle import prepare_distill_items as prepare
        from oracle.distill_expansion import digest, verify_expansion
        teacher=self.root/'teacher'; teacher.mkdir(); (teacher/'COMPLETE').touch()
        tm={'stage':'l26_ao_sft_fresh_pool_v1','layer':26,'model':'Qwen/Qwen3.5-4B',
            'prompt':prepare.PROMPT,'transform':'unit','alpha':8000,
            'source':'frozen_AR_of_fresh_nonoverlap_AO_crops'}
        (teacher/'meta.json').write_text(json.dumps(tm))
        old={}
        for split,i in [('train',0),('val',1)]:
            ids=torch.zeros(1,64,dtype=torch.int32); ids[0,:2]=1
            old.update({split+'_vec':torch.ones(1,2560),split+'_ids':ids,
                split+'_lengths':torch.tensor([2]),split+'_pair_rows':torch.tensor([i]),
                split+'_shards':torch.tensor([0]),split+'_rollouts':torch.tensor([i])})
        base=self.root/'base'; base.mkdir(); save_file(old,str(base/'items.safetensors'))
        bsha=digest(base/'items.safetensors')
        (base/'meta.json').write_text(json.dumps({'schema':'gt_distill_items_v1',
            'seed':42,'model':tm['model'],'layer':26,'teacher':str(teacher),'items_sha256':bsha}))
        (base/'COMPLETE').touch()
        original=SimpleNamespace(d_model=2560,is_val=torch.ones(2,dtype=torch.bool),
            span_offsets=torch.tensor([0,2,4]),span_token_ids=torch.ones(4,dtype=torch.int32),
            shard_ids=torch.zeros(2,dtype=torch.int32),rollout_ids=torch.tensor([0,1]),
            activations=torch.ones(2,2560))
        fresh=SimpleNamespace(val_indices=torch.arange(50),shard_ids=torch.ones(50,dtype=torch.int32),
            rollout_ids=torch.arange(50),span_offsets=torch.arange(51)*2,
            span_token_ids=torch.ones(100,dtype=torch.int32),activations=torch.ones(50,2560)*2,
            pair_paths=[],rollout_paths=[])
        audit={'pair_row_offset':2,'shard_offset':1}
        out=self.root/'out'
        argv=['prepare','--teacher',str(teacher),'--base-items',str(base),'--fresh-data',str(self.root/'fresh'),
              '--output',str(out),'--n-train','4','--reserve-test','1']
        with patch('sys.argv',argv), patch.object(prepare,'load_corpus',return_value=original), patch.object(prepare.AutoTokenizer,'from_pretrained',return_value=SimpleNamespace(decode=lambda *a,**k:'a useful diverse passage')), patch.dict('sys.modules',{'fresh_distill_corpus':SimpleNamespace(load_fresh=lambda *a:(fresh,audit))}), redirect_stdout(io.StringIO()):
            prepare.main()
        meta=json.loads((out/'meta.json').read_text())
        _, bank, _=verify_expansion(out,meta['items_sha256'],bsha)
        self.assertEqual(len(bank['train_vec']),4)
        self.assertTrue((bank['train_pair_rows'][1:]>=2).all())
        self.assertTrue(torch.equal(bank['val_vec'],old['val_vec']))
        self.assertTrue(torch.equal(bank['train_vec'][1:],torch.ones(3,2560)*2))
