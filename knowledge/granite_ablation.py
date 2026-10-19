"""What if a word gets fewer experts, or random ones? Real test on IBM Granite 3.1 1B-A400M (32 experts, 8 per token).

For each setting, run the five test texts and measure (a) average surprise (cross-entropy, in nats; lower is better)
on the real next token, and (b) how often the top predicted next token matches the normal model (8 experts, router).
Settings: router top-k for k in 8, 4, 2, 1; 8 random experts per token; all 32 experts (weighted by the router).
"""
import json, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
k = Path(__file__).resolve().parent
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
routers = [L.block_sparse_moe.router for L in model.model.layers]
texts = [(k / "texts" / f).read_text() for f in ["prose.txt", "code.py", "maths.txt", "poem.txt", "spanish.txt"]]
gen = torch.Generator().manual_seed(0)


def make_forward(mode, kk):
    def fwd(self, h):
        logits = F.linear(h, self.weight).float()
        if mode == "random":
            idx = torch.stack([torch.randperm(logits.shape[1], generator=gen)[:kk] for _ in range(logits.shape[0])])
            top = logits.gather(1, idx)
        else:
            top, idx = logits.topk(kk, dim=-1)
        return logits, torch.softmax(top, -1).type_as(h), idx
    return fwd


def run(mode, kk):
    for r in routers:
        r.forward = make_forward(mode, kk).__get__(r)
    loss, preds = [], []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids
            out = model(input_ids=ids).logits[0, :-1]
            loss.append(F.cross_entropy(out, ids[0, 1:], reduction="none"))
            preds.append(out.argmax(-1))
    return torch.cat(loss).mean().item(), torch.cat(preds)


settings = [("router", 8), ("router", 4), ("router", 2), ("router", 1), ("random", 8), ("router", 32)]
base_loss, base_pred = run("router", 8)
res = []
for mode, kk in settings:
    l, p = run(mode, kk)
    res.append({"setting": f"{mode} top-{kk}" if mode == "router" else f"random {kk}", "experts": kk,
                "surprise_nats": round(l, 3), "same_top_token_as_normal": round((p == base_pred).float().mean().item(), 3)})
    print(res[-1], file=sys.stderr)
(k.parent / "data/granite-ablation.json").write_text(json.dumps({"model": MODEL, "tokens": len(base_pred), "results": res}, indent=1))
