#!/usr/bin/env python3
"""Aggregate v2 results into method-mean table (mean±std across seeds)."""
import csv
import statistics
from pathlib import Path

CSV = Path("/jizhicfs/aldenliang/campaign_v2/eval_summary.csv")
METRICS = ["math", "aime24", "olympiadbench", "minerva_math", "gsm8k", "gpqa"]


def load():
    rows = []
    with open(CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def method_stats(rows, scale):
    """Return {method: {metric: (mean, std, n)}}"""
    by_method = {}
    for row in rows:
        if row["scale"] != scale:
            continue
        m = row["name"]
        by_method.setdefault(m, []).append(row)

    out = {}
    for m, group in by_method.items():
        stats = {}
        for metric in METRICS:
            vals = [float(r[metric]) for r in group if r[metric] and r[metric] != ""]
            if not vals:
                stats[metric] = (None, None, 0)
            elif len(vals) == 1:
                stats[metric] = (vals[0], 0.0, 1)
            else:
                stats[metric] = (statistics.mean(vals), statistics.pstdev(vals), len(vals))
        out[m] = stats
    return out


def print_table(rows, scale, title):
    stats = method_stats(rows, scale)
    method_order = ["baseline", "ar", "difffilter", "gfpo", "maxvar", "topk", "per", "softmax", "diffband"]
    print(f"\n### {title} — mean over seeds\n")
    hdr = "| method | " + " | ".join(METRICS) + " | avg |"
    sep = "|" + "|".join(["-" * 8] * (len(METRICS) + 2)) + "|"
    print(hdr)
    print(sep)
    for m in method_order:
        if m not in stats:
            continue
        s = stats[m]
        cells = [m]
        vals_present = []
        for metric in METRICS:
            mn, sd, n = s[metric]
            if mn is None:
                cells.append("-")
            else:
                cells.append(f"{mn:.1f}" + (f"±{sd:.1f}" if sd and sd > 0 else ""))
                vals_present.append(mn)
        avg = statistics.mean(vals_present) if vals_present else None
        cells.append(f"**{avg:.1f}**" if avg else "-")
        print("| " + " | ".join(cells) + " |")


def main():
    rows = load()
    for scale, title in [("7b", "7B Scale"), ("05b", "0.5B Scale")]:
        print_table(rows, scale, title)


if __name__ == "__main__":
    main()
