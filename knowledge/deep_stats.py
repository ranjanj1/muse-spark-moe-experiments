"""What the 62-floor trace shows (knowledge/trace-deep-derivative.json) -> data/deep.json.

- the two " x" tokens: experts shared per floor (random would be 8*8/256 = 0.25)
- each word: how often a floor reuses an expert from the floor below (experts are different networks on every floor,
  so this is only a numbering coincidence check: expected 8*8/256)
- sentence-level: distinct experts used per floor (10 words x 8 picks)
- 4-GPU expert parallelism, assuming experts split in order (0-63 on GPU 1, ...): GPUs each word touches per floor
- whether floor-1 "maths" expert 130 appears for this sentence
"""
import json
from pathlib import Path

import numpy as np

k = Path(__file__).resolve().parent
tr = json.load(open(k / "trace-deep-derivative.json"))
tok, layers = tr["tokens"], tr["layers"]
words = list(range(1, len(tok)))                                   # skip BOS
xs = [i for i, t in enumerate(tok) if t == " x"]
shared_x = [len(set(L["experts"][xs[0]]) & set(L["experts"][xs[1]])) for L in layers]
distinct = [len({e for w in words for e in L["experts"][w]}) for L in layers]
gpus = [[len({e // 64 for e in L["experts"][w]}) for w in words] for L in layers]
gpu_mean = float(np.mean(gpus))
gpu_hist = np.bincount(np.array(gpus).ravel(), minlength=5)[1:].tolist()
e130 = [sum(130 in L["experts"][w] for w in words) for L in layers]
# expected GPUs touched if 8 experts were random out of 256 split in 4 groups of 64
rng = np.random.default_rng(0)
rand = np.mean([len({e // 64 for e in rng.choice(256, 8, replace=False)}) for _ in range(20000)])
# per-word experts at floor 1 vs other words: how similar are words to each other, floor by floor
pair = [np.mean([len(set(L["experts"][a]) & set(L["experts"][b])) for a in words for b in words if a < b]) for L in layers]
out = {"tokens": tok, "x_positions": xs, "shared_between_xs": shared_x, "random_shared": 8 * 8 / 256,
       "mean_shared_xs_floors_1_10": float(np.mean(shared_x[:10])), "mean_shared_xs_floors_53_62": float(np.mean(shared_x[-10:])),
       "distinct_experts_per_floor": distinct, "mean_shared_any_two_words": [round(float(p), 2) for p in pair],
       "gpus_touched_mean": gpu_mean, "gpus_touched_hist_1to4": gpu_hist, "gpus_touched_random": float(rand),
       "expert130_count_per_floor": e130}
(k.parent / "data/deep.json").write_text(json.dumps(out, indent=1))
print("tokens", tok)
print("x positions", xs)
print("shared between the two x's per floor:", shared_x)
print(f"  floors 1-10 mean {out['mean_shared_xs_floors_1_10']:.2f}, floors 53-62 mean {out['mean_shared_xs_floors_53_62']:.2f}, random 0.25")
print("mean experts shared by any two words, floors 1..62:", out["mean_shared_any_two_words"])
print("distinct experts per floor (max 80):", distinct)
print(f"GPUs touched per word per floor: mean {gpu_mean:.2f} (random {rand:.2f}), counts of 1/2/3/4 GPUs: {gpu_hist}")
print("expert 130 uses per floor:", e130[:5], "...")
