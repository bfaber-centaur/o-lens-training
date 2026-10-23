import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage_fresh_harvest import stage, copy_verified


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root/'source'; self.src.mkdir()
        for d in ('pairs','rollouts','manifests'): (self.src/d).mkdir()
        self.write('progress.json',dict(complete_shards=1,pairs=1,target_pairs=1,rollouts=1,
                   generated_tokens=2,span_length_hist={'2':1}))
        self.write('run_config.json',{'contract':{'layers':[26]}})
        self.write('environment.json',{'model_revision':'test'})
        self.write('manifests/shard_00000.json',dict(shard_id=0,layers=[26],pairs=1,rollouts=1,
                   generated_tokens=2,span_length_hist={'2':1},
                   pair_files={'26':'pairs_l26_00000.safetensors'},rollout_file='rollouts_00000.jsonl'))
        self.write('rollouts/rollouts_00000.jsonl',{'assistant_ids':[1,2]})
        header=json.dumps({'activations':{'dtype':'F32','shape':[1,2560],'data_offsets':[0,10240]}}).encode()
        (self.src/'pairs/pairs_l26_00000.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes(10240))

    def write(self,name,obj): (self.src/name).write_text(json.dumps(obj)+'\n')

    def test_stage_never_lists_source(self):
        out=self.root/'local'
        with patch.object(Path,'glob',side_effect=AssertionError('glob forbidden')):
            report=stage(self.src,out)
        self.assertEqual(report['totals']['pairs'],1)
        self.assertTrue((out/'STAGED_COMPLETE').is_file())
        for name in report['files_sha256']:
            self.assertEqual((self.src/name).read_bytes(),(out/name).read_bytes())

    def test_missing_manifest_stops_without_success_marker(self):
        (self.src/'manifests/shard_00000.json').unlink()
        out=self.root/'local'
        with self.assertRaises(FileNotFoundError): stage(self.src,out,attempts=1)
        self.assertFalse((out/'STAGED_COMPLETE').exists())
        self.assertTrue((self.src/'pairs/pairs_l26_00000.safetensors').is_file())

    def test_count_mismatch_stops(self):
        self.write('rollouts/rollouts_00000.jsonl',{'assistant_ids':[1]})
        with self.assertRaisesRegex(ValueError,'count mismatch'): stage(self.src,self.root/'local')

    def test_retry_transient_missing_direct_read(self):
        original=Path.open
        source=self.src/'environment.json'
        failed=[]
        def opened(path,*args,**kwargs):
            if path==source and not failed:
                failed.append(True); raise FileNotFoundError('transient')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',opened):
            copy_verified(source,self.root/'copied',attempts=2,delay=0)
        self.assertEqual(source.read_bytes(),(self.root/'copied').read_bytes())

    def test_mismatch_not_retried_as_missing(self):
        with patch('scripts.stage_fresh_harvest.digest',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                copy_verified(self.src/'environment.json',self.root/'copied',delay=0)
        self.assertFalse((self.root/'copied').exists())

    def test_existing_destination_refused(self):
        out=self.root/'local'; out.mkdir(); (out/'keep').write_text('keep')
        with self.assertRaises(FileExistsError): stage(self.src,out)
        self.assertEqual((out/'keep').read_text(),'keep')
