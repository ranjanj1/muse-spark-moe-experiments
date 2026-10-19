"""Range-read small tensors from Meta's gated Muse Spark Open safetensors (needs HF_TOKEN in the environment).

Reads only each shard's header plus the requested tensors' bytes; never downloads a whole shard.
Usage: python3 -I knowledge/fetch_tensors.py <regex> <out.json>
"""
import json, os, re, struct, sys, urllib.request
from pathlib import Path

REPO = "https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/resolve/main/"
root = Path(__file__).resolve().parent.parent
token = os.environ["HF_TOKEN"]

def get(url, start, end):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()

def decode(raw, dtype):
    if dtype == "F32":
        return list(struct.unpack(f"<{len(raw)//4}f", raw))
    if dtype == "BF16":
        return [struct.unpack("<f", b"\0\0" + raw[i:i+2])[0] for i in range(0, len(raw), 2)]
    raise ValueError(dtype)

pattern, out = re.compile(sys.argv[1]), Path(sys.argv[2])
wmap = json.load(open(root / "data/raw/model.safetensors.index.json"))["weight_map"]
want = sorted(k for k in wmap if pattern.search(k))
headers, result = {}, {}
for name in want:
    shard = wmap[name]
    if shard not in headers:
        n = struct.unpack("<Q", get(REPO + shard, 0, 7))[0]
        headers[shard] = (8 + n, json.loads(get(REPO + shard, 8, 7 + n)))
    base, h = headers[shard]
    meta = h[name]
    s, e = meta["data_offsets"]
    result[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "shard": shard,
                    "values": decode(get(REPO + shard, base + s, base + e - 1), meta["dtype"])}
    print(name, meta["dtype"], meta["shape"], file=sys.stderr)
out.write_text(json.dumps(result))
