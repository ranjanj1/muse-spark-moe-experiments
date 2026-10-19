"""Derived numbers for the Muse Spark Open film, from Meta's own sources only.

Inputs (data/raw/): titan-flavors.py + configuration_muse_spark.py (architecture), hf-api.json (BF16 parameter
count), tree-*.json (checkpoint file sizes), and the NVFP4 repo's safetensors dtype counts (below, from the HF API).
Writes data/analysis.json.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
raw = root / "data" / "raw"

# Architecture: TorchTitan flavor "816B-A42B" and transformers MuseSparkConfig (both Meta / HF, 2026-10-15)
dim, layers, experts, top_k = 8192, 62, 256, 8
expert_hidden = 4096          # SwiGLU expert width; experts read a 4096-wide compressed input
compressed = 4096
vocab = 202_048
heads, kv_heads, head_dim = 128, 16, 64

per_expert = 3 * compressed * expert_hidden          # gate, up, down
experts_total = per_expert * experts * layers
proj_per_layer = 2 * dim * compressed                # pre/post expert projections
router_per_layer = dim * experts
attn_per_layer = dim * heads * head_dim * 2 + dim * kv_heads * head_dim * 2  # q, o, k, v
embed = 2 * vocab * dim                              # untied in/out

checkpoint_params = json.load(open(raw / "hf-api.json"))["safetensors"]["total"]
llm_claimed = 816e9
llm_non_expert = llm_claimed - experts_total         # everything outside the experts (attention, norms, gates, embeds)
active = llm_non_expert + per_expert * top_k * layers

# NVFP4 repo (HF API safetensors dtype counts, 2026-10-19): U8 packs two 4-bit values; F8_E4M3 = one scale per 16
nvfp4 = {"BF16": 48812377088, "F32": 130131456, "U8": 386547056640, "F8_E4M3": 48318382080}
fp4_values = nvfp4["U8"] * 2
quantized_layers = fp4_values / (per_expert * experts)

def size(name):
    t = json.load(open(raw / f"tree-{name}.json"))
    return sum(f.get("size", 0) for f in t if f["path"].endswith(".safetensors"))

out = {
    "per_expert_params": per_expert,
    "experts_per_layer": experts, "layers": layers, "experts_total_count": experts * layers,
    "experts_used_per_token_per_layer": top_k, "experts_used_per_token": top_k * layers,
    "expert_params_total": experts_total,
    "expert_share_of_816B": experts_total / llm_claimed,
    "non_expert_params_816B_minus_experts": llm_non_expert,
    "estimated_attention_params": attn_per_layer * layers,
    "estimated_projection_router_params": (proj_per_layer + router_per_layer) * layers,
    "embedding_params": embed,
    "active_params_estimate": active,
    "expert_fraction_used": top_k / experts,
    "active_fraction_of_816B": 42e9 / llm_claimed,
    "checkpoint_params_incl_encoders": checkpoint_params,
    "bf16_checkpoint_bytes": size("Muse-Spark-Open-816B-A42B"),
    "nvfp4_checkpoint_bytes": size("Muse-Spark-Open-816B-A42B-NVFP4"),
    "nvfp4_fp4_values": fp4_values, "nvfp4_quantized_expert_layers": quantized_layers,
    "nvfp4_bits_per_quantized_weight": (nvfp4["U8"] * 8 + nvfp4["F8_E4M3"] * 8) / fp4_values,
    "rule_of_thumb_4bit_GB": checkpoint_params * 4 / 8 / 1e9,
    "two_mac_studios_GB": 2 * 512,
    "m5_ultra_bandwidth_GBps": 1200,
    "active_expert_bytes_per_token_nvfp4_GB": per_expert * top_k * layers * 4.5 / 8 / 1e9,
}
# What one token reads in the NVFP4 build: its 8 experts per layer at 4.5 bits, plus every non-expert LLM weight at
# BF16 (the input embedding is a one-row lookup, so it is left out). Rough: ignores the conversation's KV cache.
tok_bytes = per_expert * top_k * layers * 4.5 / 8 + (llm_non_expert - vocab * dim) * 2
out["bytes_read_per_token_nvfp4_GB"] = tok_bytes / 1e9
out["share_of_nvfp4_file_read_per_token"] = tok_bytes / out["nvfp4_checkpoint_bytes"]
out["dense_ceiling_tok_s_one_chip_533GB"] = 1200e9 / out["nvfp4_checkpoint_bytes"]
(root / "data" / "analysis.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:45} {v:,.4f}" if isinstance(v, float) else f"{k:45} {v:,}")
