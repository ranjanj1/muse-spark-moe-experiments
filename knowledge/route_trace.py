"""Trace real Muse Spark Open routing: which 8 of 256 experts each token is sent to, layer by layer.

A numpy re-implementation of the text decoder in transformers' modeling_muse_spark.py (data/raw/), fed with Meta's
real BF16 weights, range-read from the gated Hugging Face files (needs HF_TOKEN). Non-expert weights of a layer are
cached in knowledge/cache/ (gitignored, ~575 MB per layer); expert weights are streamed (~100 MB each) and not kept.

Usage: .venv/bin/python -I knowledge/route_trace.py "<text>" <layers> <out.json>
  layers = how many layers to run from the bottom (1 = route at layer 1 only; each extra layer needs the experts
  chosen in the layer below: about 100 MB per distinct expert).
"""
import json, os, struct, sys, time, urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent.parent
raw = root / "data/raw"
cache = root / "knowledge/cache"
cache.mkdir(exist_ok=True)
REPO = "https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/resolve/main/"
cfg = json.load(open(raw / "config.json"))["text_config"]
wmap = json.load(open(raw / "model.safetensors.index.json"))["weight_map"]
P = "model.language_model."
D, H, KV, HD = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
E, TOPK = cfg["num_local_experts"], cfg["num_experts_per_tok"]
EPS, POST_EPS = cfg["rms_norm_eps"], cfg["post_norm_eps"]

_headers = {}


def _get(url, start, end, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}",
                                                       "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # network hiccups on long reads
            if i == tries - 1:
                raise
            print(f"  retry {i + 1}: {type(e).__name__}", file=sys.stderr)
            time.sleep(3 * (i + 1))


def _meta(name):
    shard = wmap[name]
    if shard not in _headers:
        n = struct.unpack("<Q", _get(REPO + shard, 0, 7))[0]
        _headers[shard] = (8 + n, json.loads(_get(REPO + shard, 8, 7 + n)))
    base, h = _headers[shard]
    return shard, base, h[name]


def _decode(buf, dtype, shape):
    if dtype == "F32":
        return np.frombuffer(buf, "<f4").reshape(shape).copy()
    if dtype == "BF16":
        return (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    raise ValueError(dtype)


def tensor(name, rows=None):
    """Whole tensor (cached), or selected leading-index rows of it (streamed, e.g. embedding rows or one expert)."""
    shard, base, m = _meta(name)
    s, e = m["data_offsets"]
    shape = m["shape"]
    if rows is None:
        f = cache / (name + ".npy")
        if f.exists():
            return np.load(f)
        a = _decode(_get(REPO + shard, base + s, base + e - 1), m["dtype"], shape)
        if not os.environ.get("TRACE_NOCACHE"):  # a full-depth run would need ~35 GB of cache
            np.save(f, a)
        return a
    row_bytes = (e - s) // shape[0]
    out = [_decode(_get(REPO + shard, base + s + r * row_bytes, base + s + (r + 1) * row_bytes - 1), m["dtype"],
                   shape[1:]) for r in rows]
    return np.stack(out)


def rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def centered_norm(x, w, eps=EPS):        # MuseSparkTextCenteredRMSNorm
    return rms(x, eps) * (1.0 + w)


def input_scaled_norm(x, w, eps=POST_EPS):  # MuseSparkTextInputScaledRMSNorm: scale first, then normalise
    return rms(x * (1.0 + w), eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def residual_gate(res, y, g, temp=cfg["residual_gate_temperature"]):
    w = g / temp
    beta = sigmoid(w)
    alpha = np.sqrt(np.maximum(sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha * res + beta * y


def rope(x, theta):                      # x: [T, heads, HD]
    T = x.shape[0]
    inv = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    f = np.arange(T, dtype=np.float32)[:, None] * inv[None]
    emb = np.concatenate([f, f], -1)
    cos, sin = np.cos(emb)[:, None], np.sin(emb)[:, None]
    half = np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)
    return x * cos + half * sin


def attention(x, L):
    p = f"{P}layers.{L}.self_attn."
    T = x.shape[0]
    q = (x @ tensor(p + "q_proj.weight").T).reshape(T, H, HD)
    k = (x @ tensor(p + "k_proj.weight").T).reshape(T, KV, HD)
    v = (x @ tensor(p + "v_proj.weight").T).reshape(T, KV, HD)
    q = rms(q, EPS) * cfg["qk_scale_factor"]
    k = rms(k, EPS)
    theta = cfg["layer_rope_theta"][L]
    if theta:
        q, k = rope(q, theta), rope(k, theta)
    rep = H // KV
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)   # repeat_kv: kv head j serves q heads j*rep..
    att = np.einsum("thd,shd->hts", q, k) * HD ** -0.5
    window = cfg["sliding_window"] if cfg["layer_types"][L] == "sliding_attention" else None
    i, j = np.arange(T)[:, None], np.arange(T)[None]
    mask = (j > i) | ((i - j) >= window if window else False)
    att = np.where(mask[None], -np.inf, att)
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    o = np.einsum("hts,shd->thd", att, v)
    o = rms(o, EPS).reshape(T, H * HD)
    o = o * sigmoid(x @ tensor(p + "gate_proj.weight").T)
    return o @ tensor(p + "o_proj.weight").T


def route(x, L):
    p = f"{P}layers.{L}.mlp.gate."
    logits = (x @ tensor(p + "weight").T) * cfg["output_multiplier"]
    s = sigmoid(logits)
    bias = tensor(p + "e_score_correction_bias")
    choice = s + bias
    top = np.argsort(-choice, -1)[:, :TOPK]
    w = np.take_along_axis(s, top, -1)
    w = w / (w.sum(-1, keepdims=True) + 1e-15) * cfg["routed_scaling_factor"]
    return s, choice, top, w


def moe(x, top, w, L):
    p = f"{P}layers.{L}.mlp."
    z = centered_norm(x @ tensor(p + "pre_expert_proj.weight").T, tensor(p + "pre_expert_norm.weight"))
    post_w = tensor(p + "experts.post_expert_norm.weight")
    out = np.zeros_like(z)
    experts = sorted(set(top.ravel().tolist()))
    for n, e in enumerate(experts):
        t0 = time.time()
        gu = tensor(p + "experts.gate_up_proj", rows=[e])[0]
        down = tensor(p + "experts.down_proj", rows=[e])[0]
        tok, slot = np.where(top == e)
        h = z[tok] @ gu.T
        gate, up = h[:, : h.shape[1] // 2], h[:, h.shape[1] // 2:]
        a = gate / (1.0 + np.exp(-gate)) * up
        y = input_scaled_norm(a @ down.T, post_w) * w[tok, slot][:, None]
        np.add.at(out, tok, y)
        print(f"  {time.strftime('%H:%M:%S')} layer {L + 1}: expert {e} ({n + 1}/{len(experts)}, {len(tok)} tokens, {time.time() - t0:.1f}s)",
              file=sys.stderr, flush=True)
    return out @ tensor(p + "post_expert_proj.weight").T


def run(text, layers, out=None):
    tk = Tokenizer.from_file(str(raw / "tokenizer.json"))
    ids = tk.encode(text).ids
    if not ids or ids[0] != cfg["bos_token_id"]:   # tokenizer_config: add_bos_token = true
        ids = [cfg["bos_token_id"]] + ids
    words = [tk.decode([i]) for i in ids]
    h = tensor(P + "embed_tokens.weight", rows=ids)
    h = rms(h, EPS)                      # MuseSparkTextNormedEmbedding
    trace = []
    for L in range(layers):
        p = f"{P}layers.{L}."
        res = h
        y = attention(centered_norm(h, tensor(p + "input_layernorm.weight")), L)
        h = residual_gate(res, rms(y, POST_EPS), tensor(p + "post_attention_residual_gate.gate"))
        res = h
        x = centered_norm(h, tensor(p + "pre_feedforward_layernorm.weight"))
        s, choice, top, w = route(x, L)
        trace.append({"layer": L + 1, "experts": top.tolist(), "weights": np.round(w, 4).tolist(),
                      "scores": np.round(s, 4).tolist()})
        print(f"{time.strftime('%H:%M:%S')} layer {L + 1} routed", file=sys.stderr, flush=True)
        if out:  # save after every layer so a long run can be used (or watched) while it goes
            out.write_text(json.dumps({"text": text, "ids": ids, "tokens": words, "layers": trace}))
        if L + 1 < layers:
            y = moe(x, top, w, L)
            h = residual_gate(res, input_scaled_norm(y, tensor(p + "post_feedforward_layernorm.weight")),
                              tensor(p + "post_feedforward_residual_gate.gate"))
    return {"text": text, "ids": ids, "tokens": words, "layers": trace,
            "note": "numpy re-implementation of transformers modeling_muse_spark.py with Meta's BF16 weights"}


if __name__ == "__main__":
    text, layers, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.write_text(json.dumps(run(text, layers, out)))
    print("wrote", out, file=sys.stderr)
