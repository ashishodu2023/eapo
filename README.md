# EAPO — Energy-Aware Prompt Optimization

EAPO is a small toolkit to **optimize prompts** for **accuracy, energy, and token efficiency** (Tokens-per-Joule, TPJ) using Bayesian Optimization. It measures GPU power via NVML where available.

## Features
- Prompt search space (style, reasoning, format, brevity)
- Real-time GPU power logging (NVML) → energy (J)
- Evaluation on small sample dataset (ROUGE-L proxy)
- Optuna for multi-objective scalarized search
- CSV logging + Pareto plotting scripts

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
python scripts/plot_pareto.py --csv results/trials.csv --out results/pareto_accuracy_vs_energy.png

```bash 


---

## 📦 `requirements.txt`

```txt
transformers==4.40.2
accelerate>=0.29.2
bitsandbytes>=0.43.0
optuna>=3.6.1
pyyaml>=6.0.2
tqdm>=4.66.4
pynvml>=11.5.0
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8

```

## Attention & Quantization
- Default attention is **SDPA** (no extra install). Set in `config.yaml`:
  ```yaml
  attn_implementation: "sdpa"
  ```