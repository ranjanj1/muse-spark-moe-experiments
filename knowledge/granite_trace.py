"""Full-depth routing trace of a small real MoE: IBM Granite 3.1 1B-A400M (32 experts, 8 per token, 24 layers, Apache-2.0).

Same 8-per-token choice as Muse Spark, small enough to run whole on this Mac. Records the 8 experts each token goes
to at every layer, using the router's own top-k (forward hook on each layer's router).
Usage: .venv/bin/python -I knowledge/granite_trace.py <text-file>... -> knowledge/granite-<name>.json
"""
import json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
routers = [L.block_sparse_moe.router for L in model.model.layers]
picked = []
for r in routers:
    r.register_forward_hook(lambda m, i, o: picked.append(o))

for path in map(Path, sys.argv[1:]):
    text = path.read_text()
    ids = tok(text, return_tensors="pt").input_ids
    picked.clear()
    with torch.no_grad():
        model(input_ids=ids)
    layers = []
    for L, out in enumerate(picked):
        idx = next(o for o in out if torch.is_tensor(o) and o.dtype in (torch.int64, torch.int32))
        layers.append({"layer": L + 1, "experts": idx.view(ids.shape[1], -1).tolist()})
    res = {"model": MODEL, "text": text, "tokens": [tok.decode([i]) for i in ids[0]], "layers": layers}
    out = path.parent.parent / f"granite-{path.stem}.json"
    out.write_text(json.dumps(res))
    print("wrote", out, len(res["tokens"]), "tokens", file=sys.stderr)
