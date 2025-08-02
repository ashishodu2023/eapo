import csv
from pathlib import Path

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_trials_csv(study, out_path: str | Path = "results/trials.csv"):
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    param_keys, config_keys, metric_keys = set(), set(), set()
    for t in study.trials:
        param_keys.update(t.params.keys())
        cfg = t.user_attrs.get("config", {})
        if isinstance(cfg, dict):
            config_keys.update(cfg.keys())
        m = t.user_attrs.get("metrics", {})
        if isinstance(m, dict):
            metric_keys.update(m.keys())

    fieldnames = (
        ["number", "state", "value"] +
        sorted(param_keys) +
        [f"cfg_{k}" for k in sorted(config_keys)] +
        [f"metric_{k}" for k in sorted(metric_keys)]
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in study.trials:
            row = {"number": t.number, "state": str(t.state), "value": t.value}
            for k in param_keys:
                row[k] = t.params.get(k)
            cfg = t.user_attrs.get("config", {})
            if isinstance(cfg, dict):
                for k in config_keys:
                    row[f"cfg_{k}"] = cfg.get(k)
            m = t.user_attrs.get("metrics", {})
            if isinstance(m, dict):
                for k in metric_keys:
                    row[f"metric_{k}"] = m.get(k)
            w.writerow(row)
    return out_path
