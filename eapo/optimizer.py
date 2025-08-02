import optuna
from .prompts import PromptConfig, render_prompt, STYLE, REASON, FORMAT, BREVITY
from .evaluator import evaluate_batch, load_jsonl

def objective(trial, model, tokenizer, cfg):
    style = trial.suggest_categorical("style", STYLE)
    reason = trial.suggest_categorical("reason", REASON)
    fmt   = trial.suggest_categorical("format", FORMAT)
    brev  = trial.suggest_categorical("brevity", BREVITY)
    pcfg = PromptConfig(style, reason, fmt, brev)

    data = load_jsonl(cfg["dataset_path"], limit=cfg.get("eval_batch", 6))
    prompts = [render_prompt(pcfg, ex["doc"]) for ex in data]
    res = evaluate_batch(
        model, tokenizer, data, prompts,
        max_new_tokens=cfg["max_new_tokens"],
        nvml_interval_ms=cfg["nvml_interval_ms"]
    )

    acc_w = cfg.get("accuracy_weight", 1.0)
    tpj_w = cfg.get("tpj_weight", 1.0)
    en_w  = cfg.get("energy_weight", 1.0)

    acc = res["rougeL"]
    tpj = res["tpj"] if res["tpj"]==res["tpj"] else 0.0
    energy = res["energy_total_J"] if res["energy_total_J"]==res["energy_total_J"] else 0.0

    score = acc_w*acc + tpj_w*tpj - en_w*(energy*1e-3)  # penalize kJ
    trial.set_user_attr("metrics", res)
    trial.set_user_attr("config", pcfg.__dict__)
    return score

def run_search(model, tokenizer, cfg):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, model, tokenizer, cfg), n_trials=cfg["num_trials"], show_progress_bar=True)
    return study
