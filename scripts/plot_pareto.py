#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

def to_float(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None

def prettify(col):
    return (col.replace("metric_", "")
               .replace("cfg_", "")
               .replace("_", " ")
               .title())

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser(description="Scatter + true Pareto frontier")
    p.add_argument("--csv",      default="results/trials.csv")
    p.add_argument("--out",      default="results/pareto.png")
    p.add_argument("--x",        default="metric_energy_total_J")
    p.add_argument("--y",        default="metric_rougeL")
    p.add_argument("--label",    default="cfg_brevity")
    p.add_argument("--title",    default="Accuracy vs Energy (Pareto Frontier)")
    args = p.parse_args()

    data = load_rows(args.csv)
    pts = []
    for r in data:
        x = to_float(r.get(args.x))
        y = to_float(r.get(args.y))
        if x is None or y is None:
            continue
        pts.append((x,y,r.get(args.label,""), r))

    if not pts:
        print("No valid data!")
        return

    # Build frontier: for each unique X, keep the max-Y row
    best_for_x = {}
    for x,y,lab,r in pts:
        if x not in best_for_x or y > best_for_x[x][0]:
            best_for_x[x] = (y,lab)

    # Sort frontier points by X
    frontier = sorted(((x,y_lab[0],y_lab[1]) for x,y_lab in best_for_x.items()),
                      key=lambda t: t[0])

    # Now filter to the monotonic envelope: only keep ones that improve or tie as X increases
    envelope = []
    max_y = -math.inf
    for x,y,lab in frontier:
        if y >= max_y:
            envelope.append((x,y,lab))
            max_y = y

    # Plot all points
    xs, ys, labs = zip(*[(x,y,lab) for x,y,lab,_ in pts])
    plt.figure(figsize=(10,7))
    plt.scatter(xs, ys, s=80, alpha=0.6, label="All Trials")

    # Plot and annotate frontier
    fx = [x for x,y,lab in envelope]
    fy = [y for x,y,lab in envelope]
    plt.plot(fx, fy, color="C1", lw=2, label="Pareto Frontier")
    plt.scatter(fx, fy, s=120, color="C1", edgecolor="k", linewidth=1.2)

    for x,y,lab in envelope:
        if lab:
            plt.annotate(lab, (x,y),
                         fontsize=9, xytext=(4,4),
                         textcoords="offset points", color="C1")

    # Clean up
    plt.title(args.title)
    plt.xlabel(prettify(args.x))
    plt.ylabel(prettify(args.y))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    outp = Path(args.out)
    outp.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outp, dpi=180)
    print("Saved Pareto plot to", outp.resolve())

if __name__=="__main__":
    main()
