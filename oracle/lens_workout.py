#!/usr/bin/env python3
"""Qualitative L26 lens workout: 12 fixed val activations, parent/student/shifted.

Four examples per low/middle/high FVE tercile, sampled deterministically among
accepted validation bullets. Both adapters share the student concepts_raw
prompt and identical true activations; a deranged-activation student control
tests sensitivity. Optional parent-native prompt as a separate control.
"""
import argparse
import hashlib
import html
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from train_distill_student import (
    PROMPT_TEMPLATE, load_targets, render_concepts_prompt, row, sha256
)

BINS = ("low", "mid", "high")
BASE_ARMS = ("parent_shared", "student_shared", "student_shuffled")


def select_rows(selection, bank, per_bin=4, seed=42):
    if per_bin < 1 or selection.get("schema") != "l26_nnomp_smoke_v1":
        raise ValueError("Invalid selection configuration")
    if selection.get("split") != "val" or selection.get("layer") != 26:
        raise ValueError("Only L26 validation is supported")
    eligible = {int(x) for x in bank["src_item_idx"].tolist()}
    if len(eligible) != len(bank["src_item_idx"]):
        raise ValueError("Duplicate accepted validation item")
    source = {}
    for item in selection["rows"]:
        i = int(item["item_idx"])
        if i in source:
            raise ValueError("Duplicate NNOMP item")
        source[i] = item
    if not eligible.issubset(source):
        raise ValueError("Assembly item absent from NNOMP selections")
    values = []
    for i in eligible:
        item = source[i]
        value = item.get("omp_fve")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing finite FVE for item {i}")
        if len(item.get("omp_sel", [])) != 4:
            raise ValueError(f"Accepted item {i} was not selected with four picks")
        values.append((float(value), i))
    values.sort()
    n = len(values)
    cuts = (0, n//3, (2*n)//3, n)
    out = []
    for b, name in enumerate(BINS):
        group = values[cuts[b]:cuts[b+1]]
        if len(group) < per_bin:
            raise ValueError(f"Insufficient accepted items in {name} bin")
        sampled = random.Random(f"workout:{seed}:{name}").sample(group, per_bin)
        for fve, i in sorted(sampled):
            out.append({"item_idx":i,"stratum":name,"fve":fve})
    return out


def shuffle_plan(n, seed):
    if n < 2:
        raise ValueError("Need at least two samples")
    delta = random.Random(f"shuffle:{seed}").randrange(1, n)
    return [(i+delta)%n for i in range(n)]


def bullet_stats(text):
    bullets = [x.strip()[2:].strip() for x in text.splitlines()
               if x.lstrip().startswith("- ")]
    return {"bullet_count":len(bullets),"unique_bullets":len(set(bullets)),
            "four_bullets":len(bullets)==4,"characters":len(text)}


def trim_eos(ids, eos):
    out = []
    for i in ids:
        if int(i) == eos:
            return out, True
        out.append(int(i))
    return out, False


def injected_embeddings(model, prompt, slot, vector, alpha):
    embed = model.get_input_embeddings()
    if not 0 <= slot < len(prompt):
        raise ValueError("Bad injection slot")
    if vector.ndim != 1 or vector.numel() != embed.weight.shape[1]:
        raise ValueError("Activation width mismatch")
    if not torch.isfinite(vector).all() or vector.float().norm() < 1e-8:
        raise ValueError("Nonfinite/zero residual")
    raw = embed(prompt[None, :]).clone()
    raw[:, slot] = (F.normalize(vector.float(), dim=0)*alpha).to(raw.dtype)
    return raw


@torch.inference_mode()
def decode(model, tok, prompt, slot, vector, alpha, max_new_tokens):
    pe = injected_embeddings(model,prompt,slot,vector,alpha)
    attn = torch.ones(pe.shape[:2],device=pe.device,dtype=torch.long)
    seq = model.generate(inputs_embeds=pe,attention_mask=attn,do_sample=False,
                         max_new_tokens=max_new_tokens,
                         eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.pad_token_id,use_cache=True)
    if seq.shape[0] != 1 or seq.ndim != 2:
        raise RuntimeError("Unexpected generation shape")
    ids = seq[0].tolist()
    if not 1 <= len(ids) <= max_new_tokens:
        raise RuntimeError("HF generate output includes prompt or is empty")
    ids, ended = trim_eos(ids,tok.eos_token_id)
    text = tok.decode(ids,skip_special_tokens=True)
    return {"text":text,"ids":ids,"tokens":len(ids),"eos":ended,
            "hit_cap":len(seq[0])==max_new_tokens,
            "structure":bullet_stats(text)}


def atomic_text(path, body):
    part = path.with_name(path.name+".partial")
    if part.exists():
        raise RuntimeError(f"Inspect interrupted write: {part}")
    part.write_text(body)
    os.replace(part,path)


def atomic_json(path, content):
    atomic_text(path,json.dumps(content,ensure_ascii=False,indent=2)+"\n")


def check_lineage(selection_path,val_path,parent_path,student_path):
    vm, bank = load_targets(val_path,"val")
    report_path = selection_path/"report.json"
    if not (selection_path/"COMPLETE").is_file():
        raise ValueError("Validation NNOMP selection incomplete")
    selection = json.loads(report_path.read_text())
    if (vm["selection_sha256"] != sha256(report_path)
        or selection["items_sha256"] != vm["items_sha256"]
        or selection.get("source") != "true_GT_residual"):
        raise ValueError("Selection/assembly checksum mismatch")
    if not (parent_path/"COMPLETE").is_file():
        raise ValueError("Parent checkpoint incomplete")
    pm = json.loads((parent_path/"meta.json").read_text())
    padapter = parent_path/"lora"/"adapter_model.safetensors"
    if (pm.get("stage") != "l26_ao_sft_fresh_pool_v1"
        or pm.get("model") != vm["model"] or pm.get("layer") != 26
        or pm.get("transform") != "unit" or float(pm.get("alpha",-1)) != 8000
        or pm.get("prompt_slot") != "last token of rendered prompt"
        or not padapter.is_file()):
        raise ValueError("Parent SFT injection lineage mismatch")
    run_path = student_path.parent.parent
    if not (student_path/"COMPLETE").is_file():
        raise ValueError("Student checkpoint incomplete")
    sadapter = student_path/"lora"/"adapter_model.safetensors"
    sm = json.loads((run_path/"meta.json").read_text())
    if (not sadapter.is_file()
        or sm.get("stage") != "l26_distill_student_sft_v1"
        or sm.get("source") != "true_GT_residual"
        or sm.get("prompt_mode") != "concepts_raw"
        or sm.get("model") != vm["model"] or sm.get("layer") != 26
        or sm.get("transform") != "unit" or float(sm.get("alpha",-1)) != 8000
        or Path(sm["init_from"]).resolve() != parent_path.resolve()
        or sm["init_adapter_sha256"] != sha256(padapter)
        or Path(sm["val"]).resolve() != val_path.resolve()
        or sm["val_targets_sha256"] != vm["targets_sha256"]
        or sm.get("val_filter_policy") != vm["filter_policy"]):
        raise ValueError("Student or fixed-validation lineage mismatch")
    return vm, bank, selection, pm, sm, sha256(padapter), sha256(sadapter)


def markdown_report(manifest, rows):
    content = ["# L26 Lens Workout", "",
        "Qualitative validation pilot, not an independent final evaluation. "
        "NNOMP joint FVE is geometric reconstruction, not interpretation quality.",
        "",f"**Injection:** true L26 residual / norm × {manifest['alpha']:g}; "
        f"greedy {manifest['max_new_tokens']}-token cap.", "",
        "| FVE bin | Item | FVE | Parent bullets | Student bullets | Shuffled bullets |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        vals=[r["outputs"][k]["structure"]["bullet_count"] for k in BASE_ARMS]
        content.append(f"| {r['stratum']} | {r['item_idx']} | {r['fve']:.3f} | "
                       + " | ".join(str(x) for x in vals)+" |")
    for r in rows:
        content.extend(["",f"## {r['stratum'].upper()} FVE — item {r['item_idx']} "
                        f"({r['fve']:.3f})", "",
                        f"Pair row: {r['src_pair_row']}; shuffled control from "
                        f"item {r['shuffled_from_item_idx']}.", "",
                        "### Frozen NNOMP bullet target", "", "~~~text",r["target"],"~~~"])
        for arm, label in (("parent_shared","Parent: shared prompt"),
                           ("student_shared","Student: shared prompt"),
                           ("student_shuffled","Student: shuffled activation"),
                           ("parent_native","Parent: native parent prompt")):
            if arm in r["outputs"]:
                d=r["outputs"][arm]
                content.extend(["",f"### {label}", "",
                                f"{d['tokens']} tokens, EOS={d['eos']}, "
                                f"{d['structure']['bullet_count']} bullet lines", "",
                                "~~~text",d["text"],"~~~"])
    return "\n".join(content)+"\n"


def html_report(rows):
    esc = lambda x: html.escape(str(x),quote=True)
    navigation = " · ".join('<a href="#i'+str(r["item_idx"])+'">'+
                            esc(r["stratum"]+" "+str(r["item_idx"]))+'</a>' for r in rows)
    sections=[]
    for r in rows:
        cols=[("NNOMP target",{"text":r["target"],"structure":bullet_stats(r["target"])}),
              ("Parent — same prompt",r["outputs"]["parent_shared"]),
              ("Student — same prompt",r["outputs"]["student_shared"]),
              ("Student — shuffled h",r["outputs"]["student_shuffled"])]
        if "parent_native" in r["outputs"]:
            cols.append(("Parent — native prompt",r["outputs"]["parent_native"]))
        cards="".join('<article><header><b>'+esc(label)+'</b><small>'+
             str(data["structure"]["bullet_count"])+' bullets</small></header><pre>'+
             esc(data["text"])+'</pre></article>' for label,data in cols)
        sections.append('<section id="i'+str(r["item_idx"])+'"><h2>'+
                        esc(r["stratum"].upper())+" FVE · item "+str(r["item_idx"])+
                        '</h2><p>Joint FVE '+esc(f'{r["fve"]:.3f}')+
                        ' · source row '+str(r["src_pair_row"])+
                        ' · shuffled from item '+str(r["shuffled_from_item_idx"])+
                        '</p><div class="grid">'+cards+'</div></section>')
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L26 Lens Workout</title><style>
:root{color-scheme:dark;background:#10191e;color:#e1eef1;font:15px/1.5 system-ui,sans-serif}
body{max-width:1700px;margin:auto;padding:2rem 1.5rem}h1{font-size:2.1rem}
h2{font-size:1.3rem;margin-bottom:.2rem}p{color:#b0c6cf}
a{color:#91ddce}nav{line-height:2.2}section{margin:2.8rem 0;border-top:1px solid #395561}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}
article{background:#1a2a32;border:1px solid #33515b;border-radius:12px;min-width:0;overflow:hidden}
header{padding:10px 12px;border-bottom:1px solid #33515b;display:flex;gap:8px;
justify-content:space-between;flex-wrap:wrap}
small{color:#8caeb8}pre{font:13px/1.55 ui-monospace,monospace;white-space:pre-wrap;
overflow-wrap:anywhere;margin:0;padding:12px}
</style></head><body><h1>L26 Lens Workout</h1>
<p>12 fixed validation activations; 4 from each accepted FVE tercile. Parent and
student share the same activation and concepts_raw prompt; shuffled student is
a control. FVE measures NNOMP reconstruction, not explanation quality.
Exploratory validation, not an independent test.</p><nav>""" + navigation + "</nav>" + "\n".join(sections) + "</body></html>\n"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection",type=Path,required=True)
    ap.add_argument("--val",type=Path,required=True)
    ap.add_argument("--parent",type=Path,required=True)
    ap.add_argument("--student",type=Path,required=True,
                    help="Full path to completed checkpoint, e.g. step_000050")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--per-bin",type=int,default=4)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--max-new-tokens",type=int,default=160)
    ap.add_argument("--include-parent-native",action="store_true")
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    if not 1<=a.max_new_tokens<=1024:
        ap.error("Invalid max-new-tokens")
    vm,bank,sel,pm,sm,psha,ssha=check_lineage(
        a.selection,a.val,a.parent,a.student)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.parent/"tokenizer")
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    if tok.eos_token_id is None:
        ap.error("Tokenizer has no EOS")
    shared,slot,char=render_concepts_prompt(tok,26)
    shared_hash=hashlib.sha256(shared.numpy().tobytes()).hexdigest()
    if (sm["rendered_prompt_ids_sha256"]!=shared_hash
        or sm["prompt_slot"]!=slot or sm["prompt_char"]!=char
        or sm["prompt"]!=PROMPT_TEMPLATE.format(layer=26,char=char)):
        ap.error("Prompt/tokenizer contract differs from trained student")
    native=torch.tensor(tok(pm["prompt"],add_special_tokens=False).input_ids,dtype=torch.long)
    if native.numel()<1:
        ap.error("Parent native prompt empty")
    choices=select_rows(sel,bank,a.per_bin,a.seed)
    loc={int(item):j for j,item in enumerate(bank["src_item_idx"].tolist())}
    permutation=shuffle_plan(len(choices),a.seed)
    for j,c in enumerate(choices):
        c["target_row"]=loc[c["item_idx"]]
        c["src_pair_row"]=int(bank["src_pair_row"][c["target_row"]])
        c["shuffled_from_item_idx"]=choices[permutation[j]]["item_idx"]
    arms=list(BASE_ARMS)+(["parent_native"] if a.include_parent_native else [])
    manifest={"schema":"l26_lens_workout_v1","model":pm["model"],
        "source":"true_GT_residual","layer":26,"alpha":float(pm["alpha"]),
        "seed":a.seed,"per_bin":a.per_bin,"max_new_tokens":a.max_new_tokens,
        "decoding":"greedy","arms":arms,"stratification":"FVE terciles of accepted val rows",
        "selection":str(a.selection.resolve()),
        "selection_sha256":sha256(a.selection/"report.json"),
        "val":str(a.val.resolve()),"val_targets_sha256":vm["targets_sha256"],
        "parent":str(a.parent.resolve()),"parent_adapter_sha256":psha,
        "student":str(a.student.resolve()),"student_adapter_sha256":ssha,
        "student_run_meta_sha256":sha256(a.student.parent.parent/"meta.json"),
        "prompt_sha256":shared_hash,"prompt_slot":slot,
        "selected":choices,"evaluation_status":"validation pilot, not independent final test"}
    print("WORKOUT",json.dumps({"items":[[c["stratum"],c["item_idx"],round(c["fve"],3)]
                                             for c in choices],"arms":arms},indent=2),flush=True)
    if a.dry_run:
        for c in choices[:2]:
            _,ids=row(bank,c["target_row"])
            print("TARGET",c["item_idx"],repr(tok.decode(ids.tolist(),skip_special_tokens=True)[:250]))
        print("DRY RUN: no GPU, no output",flush=True)
        return
    if not torch.cuda.is_available():
        ap.error("CUDA required")
    if a.output.exists():
        mf=a.output/"manifest.json"
        if not mf.is_file() or json.loads(mf.read_text())!=manifest:
            ap.error("Existing output does not match complete run contract")
    else:
        a.output.mkdir(parents=True)
        atomic_json(a.output/"manifest.json",manifest)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base=AutoModelForCausalLM.from_pretrained(
        pm["model"],revision=pm.get("revision"),dtype=torch.bfloat16).cuda()
    base.config.use_cache=True
    model=PeftModel.from_pretrained(
        base,a.parent/"lora",adapter_name="parent",is_trainable=False)
    model.load_adapter(a.student/"lora",adapter_name="student",is_trainable=False)
    model.eval()
    if not {"parent","student"}.issubset(model.peft_config):
        raise RuntimeError("Missing parent or student adapter")
    shared=shared.cuda()
    native=native.cuda()
    results=[]
    for c in choices:
        target_vec,target_ids=row(bank,c["target_row"])
        target=tok.decode(target_ids.tolist(),skip_special_tokens=True)
        path=a.output/f"item_{c['item_idx']:06d}.json"
        if path.exists():
            result=json.loads(path.read_text())
            if (result.get("item_idx")!=c["item_idx"]
                or result.get("src_pair_row")!=c["src_pair_row"]
                or result.get("target")!=target
                or set(result.get("outputs",{}))!=set(arms)):
                raise RuntimeError(f"Cannot resume inconsistent item {path}")
            print("SKIP",c["item_idx"],flush=True)
        else:
            shuffled_vec,_=row(bank,loc[c["shuffled_from_item_idx"]])
            outputs={}
            for arm in arms:
                model.set_adapter("parent" if arm.startswith("parent") else "student")
                is_native=(arm=="parent_native")
                prompt=(native if is_native else shared)
                injection_slot=(len(native)-1 if is_native else slot)
                vec=shuffled_vec if arm=="student_shuffled" else target_vec
                outputs[arm]=decode(model,tok,prompt,injection_slot,vec.cuda(),
                                    float(pm["alpha"]),a.max_new_tokens)
            result={**c,"target":target,"outputs":outputs}
            atomic_json(path,result)
            print(f"ITEM {c['item_idx']} {c['stratum']} FVE={c['fve']:.3f} "
                  f"parent={outputs['parent_shared']['structure']['bullet_count']} "
                  f"student={outputs['student_shared']['structure']['bullet_count']} bullets",
                  flush=True)
        results.append(result)
    metrics={arm:{"four_bullet_outputs":sum(
            r["outputs"][arm]["structure"]["four_bullets"] for r in results),
            "mean_bullet_count":sum(
            r["outputs"][arm]["structure"]["bullet_count"] for r in results)/len(results),
            "hit_cap":sum(r["outputs"][arm]["hit_cap"] for r in results)}
            for arm in arms}
    atomic_json(a.output/"report.json",{"manifest":manifest,"rows":results,
                                         "format_diagnostics":metrics})
    atomic_text(a.output/"report.md",markdown_report(manifest,results))
    atomic_text(a.output/"report.html",html_report(results))
    (a.output/"COMPLETE").write_text("ok\n")
    print("LENS WORKOUT COMPLETE",a.output,flush=True)
    print("FORMAT DIAGNOSTICS",json.dumps(metrics),flush=True)


if __name__=="__main__":
    main()
