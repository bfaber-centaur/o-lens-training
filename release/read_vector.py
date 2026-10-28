#!/usr/bin/env python3
"""Read one saved residual vector with a released O-Lens adapter."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True, help="Downloaded HF repo")
    p.add_argument("--layer", default="l26")
    p.add_argument("--variant", default="expanded", choices=["expanded", "control"])
    p.add_argument("--vector", type=Path, required=True, help="Safetensors with raw activation [2560]")
    p.add_argument("--key", default="activation")
    p.add_argument("--result", type=Path)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, get_peft_model_state_dict
    directory = a.release / a.layer / a.variant
    m = json.loads((directory / "contract.json").read_text())
    tok = AutoTokenizer.from_pretrained(directory / "tokenizer")
    ids = torch.tensor(m["prompt_ids"], dtype=torch.long, device="cuda")
    assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == m["rendered_prompt_ids_sha256"]
    vector = load_file(str(a.vector))[a.key].float()
    assert vector.shape == (2560,) and torch.isfinite(vector).all() and vector.norm() > 1e-8
    base = AutoModelForCausalLM.from_pretrained(
        m["model"], revision=m.get("revision"), dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False).eval()
    saved = load_file(str(directory / "adapter/adapter_model.safetensors"))
    live = get_peft_model_state_dict(model)
    assert saved.keys() == live.keys(), "Adapter keys did not map exactly"
    assert all(torch.equal(v.float(), live[k].detach().cpu().float())
               for k, v in saved.items()), "Adapter weights differ after loading"
    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(ids[None]).clone()
        embeddings[:, m["prompt_slot"]] = (
            F.normalize(vector.cuda(), dim=0) * m["alpha"]).to(embeddings.dtype)
        output = model.generate(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(ids[None].shape, dtype=torch.long, device="cuda"),
            max_new_tokens=64, do_sample=False, num_beams=1, repetition_penalty=1.0,
            no_repeat_ngram_size=0, use_cache=True,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    text = tok.decode(output[0], skip_special_tokens=True)
    if a.result:
        a.result.write_text(json.dumps({"text": text, "ids": output[0].tolist()}) + "\n")
    print(text)

if __name__ == "__main__":
    main()
