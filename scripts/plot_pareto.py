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
    except:
        return None

def load_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def prettify(name):
    return (name.replace('metric_', '').replace('cfg_', '').replace('_', ' ').title())

def non_dominated(points, minimize_x, maximize_y, eps_x, eps_y):
    nd = []
    for i, a in enumerate(points):
        ax, ay = a['x'], a['y']
        dominated = False
        for j, b in enumerate(points):
            if i == j: continue
            bx, by = b['x'], b['y']
            if minimize_x:
                x_ok = bx <= ax + eps_x
                x_strict = bx < ax - eps_x
            else:
                x_ok = bx >= ax - eps_x
                x_strict = bx > ax + eps_x
            if maximize_y:
                y_ok = by >= ay - eps_y
                y_strict = by > ay + eps_y
            else:
                y_ok = by <= ay + eps_y
                y_strict = by < ay - eps_y
            if x_ok and y_ok and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            nd.append(a)
    nd.sort(key=lambda p: p['x'], reverse=not minimize_x)
    return nd

def main():
    parser = argparse.ArgumentParser(
        description='Pareto plot with epsilon-dominance'
    )
    parser.add_argument('--csv', default='results/trials.csv', help='Input CSV path')
    parser.add_argument('--out', default='results/pareto.png', help='Output image path')
    parser.add_argument('--x', default='metric_energy_total_J', help='X-axis column')
    parser.add_argument('--y', default='metric_rougeL', help='Y-axis column')
    parser.add_argument('--label', default='', help='Column for point labels')
    parser.add_argument('--title', default='', help='Plot title')
    parser.add_argument('--eps_x', type=float, default=0.0, help='Epsilon tolerance on X')
    parser.add_argument('--eps_y', type=float, default=0.005, help='Epsilon tolerance on Y')
    parser.add_argument('--minimize_x', action='store_true', help='Minimize X (default for energy/latency)')
    parser.add_argument('--maximize_y', action='store_true', help='Maximize Y (default for accuracy/TPJ)')
    args = parser.parse_args()

    rows = load_rows(args.csv)
    minimize_x = args.minimize_x or args.x in ('metric_energy_total_J', 'metric_latency_s')
    maximize_y = args.maximize_y or args.y in ('metric_rougeL', 'metric_tpj')

    points = []
    for r in rows:
        x = to_float(r.get(args.x)); y = to_float(r.get(args.y))
        if x is None or y is None: continue
        lab = r.get(args.label, '') if args.label else ''
        points.append({'x': x, 'y': y, 'lab': lab})

    if not points:
        print('No valid points found.')
        return

    frontier = non_dominated(points, minimize_x, maximize_y, args.eps_x, args.eps_y)

    plt.figure(figsize=(10, 7))
    fset = {(p['x'], p['y']) for p in frontier}
    xs = [p['x'] for p in points if (p['x'], p['y']) not in fset]
    ys = [p['y'] for p in points if (p['x'], p['y']) not in fset]
    plt.scatter(xs, ys, s=80, alpha=0.6, label='All Trials')

    fx = [p['x'] for p in frontier]
    fy = [p['y'] for p in frontier]
    plt.scatter(fx, fy, s=120, edgecolor='black', alpha=0.9, label='ε-Pareto Frontier')
    plt.plot(fx, fy, linewidth=2.0)

    if args.label:
        for p in points:
            if p['lab']:
                plt.annotate(p['lab'], (p['x'], p['y']), fontsize=9, xytext=(3,3), textcoords='offset points')

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel(prettify(args.x))
    plt.ylabel(prettify(args.y))
    plt.title(args.title or f'{prettify(args.y)} vs {prettify(args.x)}')
    plt.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f'Saved Pareto plot to {out.resolve()}')

if __name__ == '__main__':
    main()