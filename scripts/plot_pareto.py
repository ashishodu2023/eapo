#!/usr/bin/env python3
import csv
from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt

def load_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def compute_pareto(points, x_key, y_key, maximize_y=True):
    pts = []
    for p in points:
        x = to_float(p.get(x_key)); y = to_float(p.get(y_key))
        if x is None or y is None: continue
        pts.append((x,y,p))
    pts.sort(key=lambda t: t[0])
    frontier = []
    best_y = -float("inf") if maximize_y else float("inf")
    for x, y, p in pts:
        is_better = (y > best_y) if maximize_y else (y < best_y)
        if is_better:
            frontier.append((x, y, p)); best_y = y
    return frontier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/trials.csv")
    ap.add_argument("--out", default="results/pareto_accuracy_vs_energy.png")
    ap.add_argument("--x", default="metric_energy_total_J")  # lower is better
    ap.add_argument("--y", default="metric_rougeL")         # higher is better
    ap.add_argument("--label", default="cfg_style")
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    xs, ys, labels = [], [], []
    for r in rows:
        x = to_float(r.get(args.x)); y = to_float(r.get(args.y))
        if x is None or y is None: continue
        xs.append(x); ys.append(y); labels.append(r.get(args.label, ""))

    plt.figure()
    plt.scatter(xs, ys, alpha=0.75)
    plt.xlabel(args.x.replace("metric_", "").replace("_", " ").title())
    plt.ylabel(args.y.replace("metric_", "").replace("_", " ").title())
    plt.title("Accuracy vs Energy (All Trials)")

    frontier = compute_pareto(rows, args.x, args.y, maximize_y=True)
    if frontier:
        fx = [x for x,_,_ in frontier]; fy = [y for _,y,_ in frontier]
        plt.plot(fx, fy)

    for x, y, lab in zip(xs, ys, labels):
        if lab:
            plt.annotate(lab, (x, y), fontsize=8, alpha=0.6)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=180)
    print(f"Saved Pareto plot to {out.resolve()}")

if __name__ == "__main__":
    main()
