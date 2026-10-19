"""Statistics over the routing traces (Muse Spark layer 1, Granite all 24 layers) -> data/routing.json.

- usage similarity: cosine similarity of how often each expert is used, between two texts (1 = same mix)
- neighbour overlap: how many of a token's 8 experts the next token shares (random would be 8*8/E)
"""
import itertools, json
from pathlib import Path

import numpy as np

k = Path(__file__).resolve().parent
names = ["prose", "code", "maths", "poem", "spanish"]


def stats(files, E):
    traces = {n: json.load(open(files[n])) for n in names}
    out = []
    for L in range(len(traces["prose"]["layers"])):
        use, overlap = {}, []
        for n, t in traces.items():
            ex = np.array(t["layers"][L]["experts"][1:])            # skip the first token (BOS / start)
            use[n] = np.bincount(ex.ravel(), minlength=E) / len(ex)
            overlap += [len(set(a) & set(b)) for a, b in zip(ex[:-1], ex[1:])]
        sims = [float(use[a] @ use[b] / np.linalg.norm(use[a]) / np.linalg.norm(use[b]))
                for a, b in itertools.combinations(names, 2)]
        top = {n: int(np.argmax(use[n])) for n in names}
        out.append({"layer": L + 1, "mean_similarity_between_texts": round(float(np.mean(sims)), 3),
                    "neighbour_overlap": round(float(np.mean(overlap)), 2), "random_overlap": 8 * 8 / E,
                    "top_expert": top, "top_share": {n: round(float(use[n].max()), 2) for n in names}})
    return out


res = {"muse_spark_layer1": stats({n: k / f"trace-l1-{n}.json" for n in names}, 256),
       "granite": stats({n: k / f"granite-{n}.json" for n in names}, 32)}
(k.parent / "data/routing.json").write_text(json.dumps(res, indent=1))
for name, rows in res.items():
    print(name)
    for r in rows:
        print(f"  L{r['layer']:2} similarity {r['mean_similarity_between_texts']:.2f}  neighbours share "
              f"{r['neighbour_overlap']:.2f} of 8 (random {r['random_overlap']:.2f})  top {r['top_expert']}")
