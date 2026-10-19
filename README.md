# Opening up Muse Spark Open: mixture-of-experts experiments

Code, run logs and results behind the It's Not Magic video *"816 billion parameters, 42 billion at a time. How?"*
(YouTube [@itsnotmagicai](https://www.youtube.com/@itsnotmagicai)). Built with Muse Spark Open.

Meta's [Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) has 816B parameters, of which
about 42B are used per token: 256 experts in each of 62 layers, 8 chosen per token by a router. These scripts read
small parts of Meta's real weights (range requests, nothing like the full 1.6 TB) and trace which experts real text
is sent to.

## What we found
- **The count** (`tools/analyze.py`, `data/analysis.json`): one expert = 3 × 4096 × 4096 = 50,331,648 weights;
  15,872 experts = 798.9B = 98% of the model; per token 496 experts (25.0B) + ~17.1B shared = 42.1B, matching the
  model card. Meta's NVFP4 file quantizes exactly the experts of 60 of 62 layers (4.5 bits per weight).
- **Our re-implementation matches the official code** (`knowledge/verify_trace.py`): same experts and weights as
  transformers' `MuseSparkTextModel` on a random 6-layer model.
- **Layer 1 routing on real weights** (`knowledge/route_trace.py`, five texts in `knowledge/texts/`): maths text →
  expert 130 for 85% of tokens (but not the digits); Spanish → 76; code identifiers → 50; neighbouring tokens share
  experts 4.5× more than chance.
- **The balancing bias** (`knowledge/bias_effect.py`): averages exactly 0 per layer; changes ≥1 of the 8 picks for
  93% of tokens (2.6 of 8 on average).
- **Full-depth trace** of one sentence through all 62 layers: `knowledge/trace-deep-derivative.json`.
- **IBM Granite 3.1 1B-A400M** (32 experts, 8 per token): routing by depth (`knowledge/granite_trace.py`) and what
  happens with fewer, random or all experts (`knowledge/granite_ablation.py`): 8 random experts ≈ as bad as 2 chosen.
- **Ten open MoE models compared** from their own model cards and configs: `data/moe-models.json`.

Small samples (five short texts); read these as observations, not general laws.

## Reproduce
Needs approved access to the gated Meta repo and `HF_TOKEN` in your environment. Download `config.json`,
`model.safetensors.index.json` and `tokenizer.json` from the model repo into `data/raw/`.

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install numpy tokenizers torch transformers pypdf
.venv/bin/python -I knowledge/verify_trace.py                       # check against the official code
experiments/run.sh my-run .venv/bin/python -I knowledge/route_trace.py "Your text" 1 knowledge/trace-mine.json
```
`experiments/run.sh` saves every output line with a timestamp in `experiments/logs/`; the logs of our runs are there.
A full 62-layer trace of a 10-token sentence streams ~350 GB (nothing is stored) and took about 6 hours.

## Licences
Our code: MIT (`LICENSE`). `knowledge/expert-bias.json` holds values copied from Muse Spark Open's weights (the
`e_score_correction_bias` of every layer); it and anything else derived from the model are covered by Meta's Muse Model
License and usage policy, included as `MUSE-MODEL-LICENSE.md`. Built with Muse Spark Open. IBM Granite: Apache-2.0.
