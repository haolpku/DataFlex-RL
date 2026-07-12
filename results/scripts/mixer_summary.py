#!/usr/bin/env python3
"""Aggregate mixer results - like method_summary but for mix_7b/mix_05b scales."""
import csv, statistics
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
    by_method = {}
    for row in rows:
        if row["scale"] != scale: continue
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
    if not stats:
        print(f"\n### {title} — (no data)\n")
        return
    method_order = ["static", "reward_gap", "dump_ucb", "tscl"]
    print(f"\n### {title} — mean over seeds\n")
    hdr = "| method | " + " | ".join(METRICS) + " | avg |"
    sep = "|" + "|".join(["-" * 8] * (len(METRICS) + 2)) + "|"
    print(hdr)
    print(sep)
    for m in method_order:
        if m not in stats: continue
        s = stats[m]
        cells = [m]
        vals_present = []
        for metric in METRICS:
            mn, sd, n = s[metric]
            if mn is None:
                cells.append("-")
            else:
                cells.append(f"{mn:.1f}" + (f"(n={n})" if n < 3 else ""))
                vals_present.append(mn)
        avg = statistics.mean(vals_present) if vals_present else None
        cells.append(f"**{avg:.1f}**" if avg else "-")
        print("| " + " | ".join(cells) + " |")

def main():
    rows = load()
    for scale, title in [("mix_7b", "Mixer 7B"), ("mix_05b", "Mixer 0.5B")]:
        print_table(rows, scale, title)
    print()
    print("Note: `static` = fixed 1/3 per domain (control). Other 3 are dynamic mixers.")

if __name__ == "__main__":
    main()
