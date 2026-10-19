"""How much does the trained balancing bias change Muse Spark's choices? (layer-1 traces + real bias)

For every token: top 8 by raw router score alone vs top 8 by score + bias (what the model really uses).
Scores in the traces are rounded to 4 decimals, so near-ties are reported separately.
"""
import json
from pathlib import Path

import numpy as np

k = Path(__file__).resolve().parent
bias = np.array(json.load(open(k / "expert-bias.json"))
                ["model.language_model.layers.0.mlp.gate.e_score_correction_bias"]["values"])
rows, example = [], None
for n in ["prose", "code", "maths", "poem", "spanish"]:
    t = json.load(open(k / f"trace-l1-{n}.json"))
    L = t["layers"][0]
    for i, (tok, s, used) in enumerate(zip(t["tokens"], L["scores"], L["experts"])):
        if i == 0:
            continue
        s = np.array(s)
        plain = set(np.argsort(-s)[:8].tolist())
        changed = len(set(used) - plain)
        rows.append(changed)
        if n == "maths" and tok == " derivative":
            example = {"token": tok, "by_score_only": sorted(plain), "with_bias": sorted(used),
                       "swapped_out": sorted(plain - set(used)), "swapped_in": sorted(set(used) - plain),
                       "scores": {int(e): round(float(s[e]), 4) for e in plain | set(used)},
                       "bias": {int(e): round(float(bias[e]), 4) for e in plain | set(used)}}
rows = np.array(rows)
out = {"tokens": len(rows), "share_with_any_change": float((rows > 0).mean()),
       "mean_experts_swapped": float(rows.mean()), "histogram": np.bincount(rows).tolist(), "example": example}
print(json.dumps(out, indent=1))
(k.parent / "data/bias-effect.json").write_text(json.dumps(out, indent=1))
