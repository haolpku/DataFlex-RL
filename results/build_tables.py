"""Rebuild every results table in the DataFlex-RL paper directly from the
single public CSV (results/all_runs_171.csv). Run this to regenerate the
LaTeX table bodies used in main.tex -- if the paper's tables and this
script's output ever diverge, this script is the source of truth.

Usage: python3 build_tables.py
"""
import csv
from collections import defaultdict

MATH5 = ["math500", "aime24", "olympiadbench", "minerva_math", "gsm8k"]
SIX = MATH5 + ["gpqa_diamond"]
LOGIC = ["kk_hard", "bbh_logical_deduction", "bbh_tracking", "zebra_logic_mc"]
SCI = ["gpqa_diamond", "mmlu_pro_chemistry", "mmlu_pro_physics"]

rows = list(csv.DictReader(open("all_runs_171.csv")))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def group_by(campaign=None, model=None, regime=None):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if campaign is not None and r["campaign"] != campaign:
            continue
        if model is not None and r["model"] != model:
            continue
        if regime is not None and r["regime"] != regime:
            continue
        for c in SIX + LOGIC + SCI:
            v = fnum(r.get(c))
            if v is not None:
                out[r["method"]][c].append(v)
    return out


def six_row(d):
    vals = [sum(d[c]) / len(d[c]) if d.get(c) else None for c in SIX]
    avg = sum(v for v in vals if v is not None) / len([v for v in vals if v is not None])
    return vals, avg


def dom(d, cols):
    vs = [sum(d[c]) / len(d[c]) for c in cols if d.get(c)]
    return sum(vs) / len(vs) if len(vs) == len(cols) else None


def print_six_table(title, agg, method_order):
    print(f"\n=== {title} ===")
    for m in method_order:
        if m not in agg:
            continue
        vals, avg = six_row(agg[m])
        cells = " & ".join(f"{v:.1f}" if v is not None else "--" for v in vals)
        n = len(agg[m].get("math500", []))
        print(f"    {m:12s} & {cells} & \\textbf{{{avg:.1f}}} \\\\  %% n={n}")


# ---- tab:7b-main / tab:05b-main ----
order_main = ["baseline", "ar", "per", "softmax", "difffilter", "diffband", "maxvar", "gfpo", "topk"]
print_six_table("7B-Instruct main round", group_by("main", "Qwen2.5-7B-Instruct"), order_main)
print_six_table("0.5B-Instruct main round", group_by("main", "Qwen2.5-0.5B-Instruct"), order_main)

# ---- tab:mixer ----
order_mix = ["static", "reward_gap", "dump_ucb", "tscl"]
print_six_table("Mixer 7B-Instruct", group_by("mixer", "Qwen2.5-7B-Instruct"), order_mix)
print_six_table("Mixer 0.5B-Instruct", group_by("mixer", "Qwen2.5-0.5B-Instruct"), order_mix)

# ---- tab:crossfamily ----
order_cf = ["baseline", "difffilter", "maxvar", "topk"]
for model in ["Qwen2.5-7B-base", "Llama-3.1-8B-base", "Qwen2.5-14B-base"]:
    print_six_table(f"Crossfamily {model}", group_by("crossfamily", model), order_cf)

# ---- tab:constrained ----
print_six_table("Constrained sub25 (J1)", group_by("constrained", "Qwen2.5-7B-Instruct", "sub25_25pct_data_subset"), order_cf)
print_six_table("Constrained noise20 (J2)", group_by("constrained", "Qwen2.5-7B-Instruct", "noise20_20pct_label_noise"), order_cf)

# ---- tab:scalecurve ----
print_six_table("Scale-curve 1.5B", group_by("scale_curve", "Qwen2.5-1.5B-Instruct"), order_cf)
print_six_table("Scale-curve 3B", group_by("scale_curve", "Qwen2.5-3B-Instruct"), order_cf)

# ---- tab:longtrain ----
order_lt = ["baseline", "difffilter", "topk"]
print_six_table("Long-training 7B (1000 steps)", group_by("long_training", "Qwen2.5-7B-Instruct"), order_lt)

# ---- tab:threedomain (3-domain summary per model, baseline + [min,max]) ----
print("\n=== 3-domain summary (baseline, [min-max]) ===")
model_campaign = [
    ("Qwen2.5-0.5B-Instruct", "main"), ("Qwen2.5-1.5B-Instruct", "scale_curve"),
    ("Qwen2.5-3B-Instruct", "scale_curve"), ("Qwen2.5-7B-Instruct", "main"),
    ("Qwen2.5-7B-Instruct", "long_training"), ("Qwen2.5-7B-base", "crossfamily"),
    ("Qwen2.5-14B-base", "crossfamily"), ("Llama-3.1-8B-base", "crossfamily"),
]
for model, camp in model_campaign:
    agg = group_by(camp, model)
    for dname, cols in [("MATH", MATH5), ("LOGIC", LOGIC), ("SCI", SCI)]:
        vals = {m: dom(agg[m], cols) for m in agg}
        vals = {m: v for m, v in vals.items() if v is not None}
        if not vals:
            continue
        base = vals.get("baseline")
        lo, hi = min(vals.values()), max(vals.values())
        print(f"  {model:24s} {camp:14s} {dname:6s} base={base:.1f} [{lo:.1f}-{hi:.1f}]")
