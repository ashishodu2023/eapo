import json, time
from typing import List, Dict
import numpy as np
import torch
from .power import measure_energy

def load_jsonl(path: str, limit: int = None) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            data.append(json.loads(line))
    return data

def rouge_l(pred: str, ref: str) -> float:
    # lightweight ROUGE-L proxy
    def lcs(a, b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    a = pred.split()
    b = ref.split()
    if not b: return 0.0
    return lcs(a, b)/max(1,len(b))

def generate_and_measure(model, tokenizer, prompt: str, max_new_tokens: int, nvml_interval_ms: int):
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start_tokens = inputs["input_ids"].numel()
        with measure_energy(nvml_interval_ms) as pl:
            t0 = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False  # avoids DynamicCache seen_tokens errors on newer transformers
            )
            elapsed = time.perf_counter() - t0
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_tokens = out[0].numel() - start_tokens
        return text, pl.energy_joules(pl.elapsed), elapsed, gen_tokens

def evaluate_batch(model, tokenizer, batch: List[Dict], prompts: List[str], max_new_tokens: int, nvml_interval_ms: int) -> Dict:
    rouge_scores, energies, latencies, gen_tokens = [], [], [], []
    for ex, p in zip(batch, prompts):
        y, E, t, toks = generate_and_measure(model, tokenizer, p, max_new_tokens, nvml_interval_ms)
        rouge_scores.append(rouge_l(y, ex.get("ref","")))
        energies.append(E)
        latencies.append(t)
        gen_tokens.append(toks)

    mean_rouge = float(np.mean(rouge_scores))
    mean_energy = float(np.nanmean(energies)) if not np.isnan(energies).all() else float("nan")
    total_tokens = int(np.sum(gen_tokens))
    total_energy = float(np.nansum(energies))
    tpj = float(total_tokens / total_energy) if total_energy and not np.isnan(total_energy) and total_energy>0 else float("nan")
    return {
        "rougeL": mean_rouge,
        "energy_J_per_ex": mean_energy,
        "tokens_total": total_tokens,
        "energy_total_J": total_energy,
        "tpj": tpj,
        "latency_s": float(np.mean(latencies))
    }