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
    except Exception:
        return None

def load_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def prettify(name: str) -> str:
    return (name.replace("metric_", "")
                .replace("cfg_", "")
                .replace("_", " ")
                .title())

def non_dominated(points, minimize_x=True, maximize_y=True, eps_x=0.0, eps_y=0.0):
    """
    Epsilon‐dominance frontier.
    A dominates B if:
      X: A.x <= B.x + eps_x    (when minimizing)
         or A.x >= B.x - eps_x (when maximizing)
      AND
      Y: A.y >= B.y - eps_y    (when maximizing)
         or A.y <= B.y + eps_y (when minimizing)
      AND at least one strict by > eps.
    """
    nd = []
    for i, a in enumerate(points):
        ax, ay = a["x"], a["y"]
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            bx, by = b["x"], b["y"]
            # X dominance check
            if minimize_x:
                x_ok = bx <= ax + eps_x
                x_strict = bx < ax - eps_x
            else:
                x_ok = bx >= ax - eps_x
                x_strict = bx > ax + eps_x
            # Y dominance check
            if maximize_y:
                y_ok = by >= ay - eps_y
                y_strict = by > ay + eps_y
            else:
                y_ok = by <= ay + eps_y
                y_strict = by < ay - eps_y
            # If B dominates A
            if x_ok and y_ok and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            nd.append(a)
    # sort frontier for plotting
    nd.sort(key=lambda p: p["x"], reverse=not minimize_x)
    return nd

def main():
    parser = argparse.ArgumentParser(
        description="Plot ε‐Pareto frontier for EAPO results"
    )
    parser.add_argument("--csv", default="results/trials.csv",
                        help="Path to EAPO trials CSV")
    parser.add_argument("--out", default="results/pareto.png",
                        help="Output image path")
    parser.add_argument("--x", default="metric_energy_total_J",
                        help="Column for X axis")
    parser.add_argument("--y", default="metric_rougeL",
                        help="Column for Y axis")
    parser.add_argument("--label", default="",
                        help="Column to annotate points (e.g., cfg_style)")
    parser.add_argument("--title", default="Accuracy vs Energy (All Trials)",
                        help="Plot title")
    parser.add_argument("--show_frontier", action="store_true",
                        help="Overlay ε‐Pareto frontier")
    parser.add_argument("--eps_x", type=float, default=0.0,
                        help="ε tolerance on X dominance")
    parser.add_argument("--eps_y", type=float, default=0.0,
                        help="ε tolerance on Y dominance")
    parser.add_argument("--minimize_x", action="store_true",
                        help="Treat X as to-be-minimized")
    parser.add_argument("--maximize_y", action="store_true",
                        help="Treat Y as to-be-maximized")
    args = parser.parse_args()

    rows = load_rows(args.csv)

    # sensibly default objectives
    minimize_x = args.minimize_x or args.x in {"metric_energy_total_J", "metric_latency_s"}
    maximize_y = args.maximize_y or args.y in {"metric_rougeL", "metric_tpj"}

    # gather points
    points = []
    for r in rows:
        x = to_float(r.get(args.x))
        y = to_float(r.get(args.y))
        if x is None or y is None:
            continue
        lab = r.get(args.label, "") if args.label else ""
        points.append({"x": x, "y": y, "lab": lab, "row": r})

    if not points:
        print("No valid data found; check CSV and column names.")
        return

    # compute ε-Pareto if requested
    frontier = []
    if args.show_frontier:
        frontier = non_dominated(points,
                                  minimize_x=minimize_x,
                                  maximize_y=maximize_y,
                                  eps_x=args.eps_x,
                                  eps_y=args.eps_y)

    # plot
    plt.figure(figsize=(10, 7))
    # non-frontier
    fset = {(p["x"], p["y"]) for p in frontier}
    xs = [p["x"] for p in points if (p["x"], p["y"]) not in fset]
    ys = [p["y"] for p in points if (p["x"], p["y"]) not in fset]
    plt.scatter(xs, ys, s=80, alpha=0.8, label="All Trials")

    # frontier
    if frontier:
        fx = [p["x"] for p in frontier]
        fy = [p["y"] for p in frontier]
        plt.scatter(fx, fy, s=120, edgecolor="k", linewidth=1.2,
                    alpha=0.9, label="ε‐Pareto Frontier")
        plt.plot(fx, fy, linewidth=2.0)

    # labels
    if args.label:
        for p in points:
            if p["lab"]:
                plt.annotate(p["lab"],
                             (p["x"], p["y"]),
                             fontsize=9,
                             xytext=(4, 4),
                             textcoords="offset points",
                             alpha=0.9)

    plt.title(args.title)
    plt.xlabel(prettify(args.x))
    plt.ylabel(prettify(args.y))
    plt.grid(True, linestyle="--", alpha=0.3)
    if frontier:
        plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved plot to {out_path.resolve()}")

if __name__ == "__main__":
    main()
