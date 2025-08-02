from eapo.utils import load_cfg, set_seed
from eapo.models import load_model
from eapo.optimizer import run_search
from eapo.logger import write_trials_csv

def main():
    cfg = load_cfg("config.yaml")
    set_seed(cfg.get("seed", 42))
    tok, model = load_model(
        cfg["model_id"],
        device_map=cfg.get("device_map", "auto"),
        quantization=cfg.get("quantization", "4bit"),
        trust_remote_code=True
    )
    study = run_search(model, tok, cfg)
    best = study.best_trial
    print("\n=== Best Trial ===")
    print("Score:", best.value)
    print("Prompt config:", best.user_attrs["config"])
    print("Metrics:", best.user_attrs["metrics"])

    csv_path = write_trials_csv(study, "results/trials.csv")
    print(f"\nCSV written to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
