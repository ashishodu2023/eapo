# EAPO — Energy-Aware Prompt Optimization

Optimize prompts for **accuracy**, **energy**, and **token efficiency** (Tokens-per-Joule, TPJ) on local or API LLMs. EAPO measures GPU power (via NVML), runs **Bayesian optimization** over a structured prompt space, and produces CSV logs and Pareto plots.

> ✅ Phi-3 and many open models supported.  
> ✅ 4-bit / 8-bit quantization (bitsandbytes).  
> ✅ Works without NVML (energy becomes N/A; tokens/latency still logged).

---

## Table of Contents

- [Features](#features)  
- [Project Structure](#project-structure)  
- [Quickstart](#quickstart)  
- [Configuration](#configuration)  
- [How It Works](#how-it-works)  
- [Results & Visualization](#results--visualization)  
- [Models & Quantization Tips](#models--quantization-tips)  
- [Troubleshooting](#troubleshooting)  
- [Extending EAPO](#extending-eapo)  
- [Cite This Work](#cite-this-work)  
- [License](#license)

---

## Features

- **Energy-aware optimization**: balances task performance, token usage, and **energy** (J).
- **Prompt search space**: instruction style, reasoning cues, output format, brevity controls.
- **Optuna** Bayesian optimization with CSV logging of all trials.
- **Plotting**: Pareto frontier (accuracy vs energy), TPJ views.
- **Quantization**: 4-bit/8-bit (bitsandbytes) for low-VRAM GPUs.
- **Pluggable evaluation**: simple ROUGE-L proxy included; swap for `evaluate` easily.

---

## Project Structure

```
eapo/
├─ README.md
├─ requirements.txt
├─ config.yaml
├─ run.py                           # entry point: runs the search
├─ data/
│  └─ xsum_sample.jsonl             # tiny sample dataset
├─ results/                         # created at runtime (CSV, plots)
├─ scripts/
│  └─ plot_pareto.py                # visualization
└─ eapo/
   ├─ __init__.py
   ├─ utils.py                      # config, seeding
   ├─ power.py                      # NVML sampler → energy (J)
   ├─ models.py                     # tokenizer/model loader (+ quantization)
   ├─ prompts.py                    # prompt space & renderer
   ├─ evaluator.py                  # generation + metrics + accounting
   ├─ optimizer.py                  # Optuna loop
   └─ logger.py                     # CSV export
```

---

## Quickstart

> **Requirements**: Python 3.10+; NVIDIA GPU recommended for energy logging (NVML).  
> If you use a gated model on Hugging Face, request access and `huggingface-cli login`.

```bash
# 1) Create venv
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Configure a model (edit config.yaml)
#    Default is an open model; for Phi-3 use eager attention.

# 4) Run search
python run.py

# 5) Plot Pareto frontier
python scripts/plot_pareto.py --csv results/trials.csv --out results/pareto_accuracy_vs_energy.png
```

You’ll see Optuna trial logs and a **best trial** summary; results are written to `results/trials.csv`.

---

## Configuration

Edit **`config.yaml`**:

```yaml
model_id: "microsoft/Phi-3-mini-4k-instruct"     # any HF model you can access
quantization: "4bit"                             # "4bit" | "8bit" | "none"
device_map: "auto"                               # leave "auto" for quantized models

# Attention kernel
# - Phi-3 requires "eager"
# - Use "flash_attention_2" only if you've installed flash-attn and your GPU supports it
attn_implementation: "eager"

# bitsandbytes compute dtype: "auto" picks bf16 if supported, else fp16
bnb_compute_dtype: "auto"

max_new_tokens: 128
do_sample: false
temperature: 0.0
top_p: 1.0

dataset_path: "data/xsum_sample.jsonl"           # JSONL with {"doc": "...", "ref": "..."}
num_trials: 20                                   
energy_weight: 1.0
accuracy_weight: 1.0
tpj_weight: 1.0
eval_batch: 6
nvml_interval_ms: 50
seed: 42
```

**Dataset format (`.jsonl`)**:
```json
{"doc": "source text ...", "ref": "reference summary ..."}
```

---

## How It Works

1. **Prompt space** (`eapo/prompts.py`)  
   Composable parameters:
   - `style`: concise | role | stepwise  
   - `reasoning`: none | brief | bounded  
   - `format`: free | bullets | json  
   - `brevity`: none | 1sent | 3sent | word50

2. **Generation + Measurement** (`eapo/evaluator.py`)  
   For each example:
   - Build prompt → tokenize → **generate** (keep `use_cache=False` to avoid cache bugs on some versions).
   - Log **energy** by integrating GPU power samples (NVML).
   - Compute a task metric (default **ROUGE-L** proxy), tokens, latency.

3. **Optimization** (`eapo/optimizer.py`)  
   Optuna proposes prompt configs; objective:
   score = accuracy_weight * Acc + tpj_weight * TPJ - energy_weight * Energy (kJ)  
   The best config and metrics are saved; you can also extract the **Pareto frontier**.

---

## Results & Visualization

- **CSV**: `results/trials.csv` (params, prompt config, metrics per trial).
- **Pareto plot**:
  ```bash
  python scripts/plot_pareto.py     --csv results/trials.csv     --out results/pareto_accuracy_vs_energy.png
  ```
- **TPJ vs Accuracy**:
  ```bash
  python scripts/plot_pareto.py     --csv results/trials.csv     --out results/pareto_accuracy_vs_tpj.png     --x metric_tpj --y metric_rougeL
  ```

  ```python
# Accuracy (ROUGE-L) vs Energy — same plot as before, smaller labels
python scripts/plot_pareto.py \
  --csv results/trials.csv \
  --out results/pareto_accuracy_vs_energy.png \
  --label cfg_brevity \
  --annotate_topk 0 \
  --jitter_frac 0.0

# TPJ vs Accuracy — same as before
python scripts/plot_pareto.py \
  --csv results/trials.csv \
  --out results/pareto_accuracy_vs_tpj.png \
  --x metric_tpj \
  --y metric_rougeL \
  --label cfg_style \
  --annotate_topk 0 \
  --jitter_frac 0.0


# Latency vs Accuracy — same as before
python scripts/plot_pareto.py \
  --csv results/trials.csv \
  --out results/pareto_accuracy_vs_latency.png \
  --x metric_latency_s \
  --y metric_rougeL \
  --label cfg_style \
  --annotate_topk 0 \
  --jitter_frac 0.0

 ``` 

**Metrics logged**
- `rougeL` — ROUGE-L proxy (token-based LCS; swap for `evaluate` if you want full ROUGE).
- `energy_total_J` — total joules across the batch (NaN if NVML unavailable).
- `energy_J_per_ex` — mean joules per example.
- `tokens_total` — total generated tokens.
- `tpj` — tokens per joule (higher is better).
- `latency_s` — mean latency (s).
---

**Metrics Evaluate**
```python
# 1) Per‐example + summary outputs via explicit prompt config
python scripts/evaluate.py \
  --prompt-config '{"style":"role","reasoning":"bounded","format":"bullets","brevity":"word50"}' \
  --dataset data/xsum_sample.jsonl \
  --per-example-output results/eval_per_example.csv \
  --summary-output   results/eval_summary.json

# 2) Or read prompt config (and all other settings) from config.yaml
python scripts/evaluate.py \
  --config-yaml config.yaml \
  --per-example-output results/eval_per_example.csv \
  --summary-output   results/eval_summary.json
```
```json
{
  "mean_rougeL": 0.3846153846153846,
  "mean_energy_J": 250.10835848583076,
  "total_energy_J": 1500.6501509149846,
  "total_tokens": 768,
  "tpj": 0.5117781779662174,
  "mean_latency_s": 2.24891336250001
}

```

## Models & Quantization Tips

- **Open, lightweight**:  
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (run with `quantization: "none"`).
- **Phi-3 (works well)**:  
  `microsoft/Phi-3-mini-4k-instruct` with `attn_implementation: "eager"`.
- **Gated models** (e.g., some Mistral/Llama variants): request access + `huggingface-cli login`.

**Quantization (bitsandbytes)**
- Use `quantization: "4bit"` for larger models on limited VRAM.  
- Do **not** call `model.to(...)` after loading a quantized model.  
- Keep `device_map: "auto"`; EAPO’s loader handles correct placement.  
- `bnb_compute_dtype: "auto"` chooses bf16 if available, else fp16 (faster than fp32).

---

## Troubleshooting

**403 / gated repo**  
- Request access on the model’s HF page, then:
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  ```

**`.to` not supported for 4-/8-bit`**  
- Ensure `device_map: "auto"` and do **not** call `model.to(...)`.  
- Use our loader (`eapo/models.py`) unchanged.

**Phi-3 error: “does not support SDPA”**  
- Set `attn_implementation: "eager"` in `config.yaml`.  
- (Optional) Install `flash-attn` and set `flash_attention_2` if your GPU/CUDA stack supports it.

**Energy shows NaN**  
- NVML not accessible. Check:
  ```python
  import pynvml as nv; nv.nvmlInit(); nv.nvmlDeviceGetHandleByIndex(0)
  ```
- If on CPU/WSL/no NVIDIA GPU, energy is unavailable; TPJ is N/A but accuracy/tokens/latency still work.

**CUDA OOM**  
- Try `quantization: "8bit"` or `"4bit"`.  
- Reduce `max_new_tokens`, `eval_batch`, or choose a smaller model.

**Slow generation**  
- Ensure `bnb_compute_dtype` is bf16/fp16 (not fp32).  
- Use `attn_implementation: "eager"` (or FA2 if installed).  
- Consider shorter outputs with brevity constraints in the prompt space.

---

## Extending EAPO

- **Add new prompt factors**: edit arrays in `eapo/prompts.py` and update `render_prompt`.  
- **Swap metrics**: replace ROUGE-L proxy with `evaluate`:
  ```bash
  pip install evaluate rouge_score
  ```
  Then adapt `eapo/evaluator.py`.

- **Multi-objective**: convert scalar objective to true Pareto (`optuna.create_study(directions=[...])`) and visualize fronts.

- **API models**: wrap provider calls (OpenAI/Anthropic) and use **latency + tokens** as energy proxies.

---

## License

MIT License. See LICENSE for details.