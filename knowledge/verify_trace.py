"""Check knowledge/route_trace.py against the official transformers MuseSparkTextModel on a small random model.

Same random weights in both; compares the experts chosen and their weights at every layer.
Usage: .venv/bin/python -I knowledge/verify_trace.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import MuseSparkTextConfig, MuseSparkTextModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_trace as T  # noqa: E402

torch.manual_seed(0)
layers = 6
cfg = MuseSparkTextConfig(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=8,
    num_key_value_heads=2, head_dim=16, sliding_window=5, moe_intermediate_size=32, moe_hidden_size=32,
    num_local_experts=16, num_experts_per_tok=4, pad_token_id=0, bos_token_id=1, eos_token_id=2,
)
cfg._attn_implementation = "eager"
model = MuseSparkTextModel(cfg).float().eval()
with torch.no_grad():  # norms, gates and biases start at zero; give every parameter a value so each one matters
    for name, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * (0.3 if p.dim() == 1 else 0.2))
    for L in model.layers:
        L.mlp.gate.e_score_correction_bias.copy_(torch.randn(16) * 0.05)

sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
T.tensor = lambda name, rows=None: sd[name] if rows is None else sd[name][rows]
T.P = ""
T.cfg = cfg.to_dict()
T.D, T.H, T.KV, T.HD = 64, 8, 2, 16
T.E, T.TOPK = 16, 4
T.EPS, T.POST_EPS = cfg.rms_norm_eps, cfg.post_norm_eps

ids = torch.randint(3, 300, (1, 12))
ids[0, 0] = cfg.bos_token_id  # route_trace adds BOS like the real tokenizer config
got = []
hooks = [L.mlp.gate.register_forward_hook(lambda m, i, o: got.append((o[2][:, :].numpy(), o[1].numpy())))
         for L in model.layers]
with torch.no_grad():
    model(input_ids=ids)


class Tok:  # stand-in tokenizer so trace.run can take ids directly
    def encode(self, _):
        return type("E", (), {"ids": ids[0].tolist()})()

    def decode(self, i):
        return str(i[0])


T.Tokenizer = type("X", (), {"from_file": staticmethod(lambda _: Tok())})
mine = T.run("", layers)["layers"]
ok = True
for L in range(layers):
    ref_idx, ref_w = got[L]
    for t in range(ids.shape[1]):
        a = sorted(zip(ref_idx[t].tolist(), ref_w[t].round(3).tolist()))
        b = sorted(zip(mine[L]["experts"][t], np.round(mine[L]["weights"][t], 3).tolist()))
        if [x for x, _ in a] != [x for x, _ in b] or not np.allclose([w for _, w in a], [w for _, w in b], atol=2e-3):
            ok = False
            print(f"layer {L + 1} token {t}: ref {a} mine {b}")
print("MATCH: same experts and weights at every layer and token" if ok else "MISMATCH")
