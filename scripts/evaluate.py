#!/usr/bin/env python3
import sys, os
# ──────────────────────────────────────────────────────────────────────────
# Ensure the project root is on PYTHONPATH
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ──────────────────────────────────────────────────────────────────────────

import argparse
import json
import csv
import time
from pathlib import Path
from typing import List, Dict
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from eapo.models import load_model
from eapo.prompts import render_prompt
from eapo.power import measure_energy

def load_jsonl(path: str, limit: int = None) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data

def rouge_l(pred: str, ref: str) -> float:
    a, b = pred.split(), ref.split()
    if not b:
        return 0.0
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1] / len(b)

def generate_and_measure(model, tokenizer, prompt: str, max_new_tokens: int, nvml_interval_ms: int):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_toks = inputs["input_ids"].numel()
    with torch.no_grad(), measure_energy(nvml_interval_ms) as p:
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False
        )
        elapsed = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_toks = out[0].numel() - start_toks
    energy_j = p.energy_joules(p.elapsed)
    return text, energy_j, elapsed, gen_toks

def evaluate_all(model, tokenizer, data: List[Dict], cfg_obj: SimpleNamespace,
                 max_new_tokens: int, nvml_interval_ms: int):
    per_rows = []
    rouge_scores, energies, latencies, token_counts = [], [], [], []

    for ex in data:
        # <<< SWAPPED ARG ORDER HERE >>>
        prompt = render_prompt(cfg_obj, ex["doc"])
        pred, E, t, toks = generate_and_measure(
            model, tokenizer, prompt, max_new_tokens, nvml_interval_ms
        )
        r = rouge_l(pred, ex.get("ref",""))
        rouge_scores.append(r)
        energies.append(E)
        latencies.append(t)
        token_counts.append(toks)
        per_rows.append({
            "doc": ex["doc"],
            "ref": ex["ref"],
            "pred": pred,
            "rougeL": r,
            "energy_J": E,
            "latency_s": t,
            "tokens_gen": toks
        })

    summary = {
        "mean_rougeL": float(np.mean(rouge_scores)),
        "mean_energy_J": float(np.nanmean(energies)) if not np.isnan(energies).all() else float("nan"),
        "total_energy_J": float(np.nansum(energies)),
        "total_tokens": int(np.sum(token_counts)),
        "tpj": float(np.sum(token_counts)/np.nansum(energies)) if np.nansum(energies)>0 else float("nan"),
        "mean_latency_s": float(np.mean(latencies))
    }
    return per_rows, summary

def main():
    parser = argparse.ArgumentParser(description="Standalone evaluation using EAPO codebase")
    parser.add_argument("--prompt-config", required=True,
        help='JSON string, e.g. \'{"style":"role","reasoning":"bounded","format":"bullets","brevity":"word50"}\'')
    parser.add_argument("--dataset", default="data/xsum_sample.jsonl",
        help="Path to JSONL dataset (each line {'doc','ref'})")
    parser.add_argument("--per-example-output", default="results/per_example.csv",
        help="CSV path for per-example output")
    parser.add_argument("--summary-output", default="results/summary.json",
        help="JSON path for summary output")
    parser.add_argument("--config-yaml", default="config.yaml",
        help="Path to your main config.yaml for model settings")
    args = parser.parse_args()

    # Load main config for model settings
    root_cfg = yaml.safe_load(Path(args.config_yaml).read_text())
    tokenizer, model = load_model(
        model_id=root_cfg["model_id"],
        quantization=root_cfg.get("quantization","none"),
        device_map=root_cfg.get("device_map","auto"),
        attn_implementation=root_cfg.get("attn_implementation","eager"),
        bnb_compute_dtype=root_cfg.get("bnb_compute_dtype","auto"),
    )
    model.eval()

    # Parse prompt config dict into SimpleNamespace
    cfg_dict = json.loads(args.prompt_config)
    cfg_obj = SimpleNamespace(**cfg_dict)

    # Load data
    data = load_jsonl(args.dataset)

    # Run evaluation
    per_rows, summary = evaluate_all(
        model, tokenizer, data, cfg_obj,
        max_new_tokens=root_cfg.get("max_new_tokens",128),
        nvml_interval_ms=root_cfg.get("nvml_interval_ms",50),
    )

    # Write per-example CSV
    pe_path = Path(args.per_example_output)
    pe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pe_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)
    print(f"Wrote per-example CSV to {pe_path.resolve()}")

    # Write summary JSON
    sum_path = Path(args.summary_output)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary JSON to {sum_path.resolve()}")

if __name__=="__main__":
    main()
