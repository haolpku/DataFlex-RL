"""Merge the three source aggregation CSVs into one canonical, documented
results table covering all 171 released DataFlex-RL runs.

Inputs (already-aggregated per-checkpoint accuracy, 12 benchmark columns):
  agg_jizhi.csv       -> v2 main(54)+mixer(24) + v3 crossfamily(36) + J2 noise20(12) = 126
  agg_j1.csv          -> J1 sub25(12)
  agg_scale_lt7b.csv  -> scale-curve 1.5B/3B(24) + long-training 7B-1000step(9) = 33
Total = 126 + 12 + 33 = 171.

Output: all_runs_171.csv with explicit campaign/model/method/seed/steps columns
(not the internal group/exp shorthand), so a reader does not need this script's
source to understand the table -- only to regenerate it byte-for-byte.
"""
import csv
import re

BENCH_COLS = [
    "math500", "aime24", "olympiadbench", "minerva_math", "gsm8k",
    "kk_hard", "bbh_logical_deduction", "bbh_tracking", "zebra_logic_mc",
    "gpqa_diamond", "mmlu_pro_chemistry", "mmlu_pro_physics",
]

SCALE_MODEL = {
    "v2_7b": "Qwen2.5-7B-Instruct", "v2_05b": "Qwen2.5-0.5B-Instruct",
    "v2_mix7b": "Qwen2.5-7B-Instruct", "v2_mix05b": "Qwen2.5-0.5B-Instruct",
    "v3_qwen7b": "Qwen2.5-7B-base", "v3_llama8b": "Llama-3.1-8B-base",
    "v3_qwen14b": "Qwen2.5-14B-base",
    "J2": "Qwen2.5-7B-Instruct", "J1": "Qwen2.5-7B-Instruct",
}
CAMPAIGN = {
    "v2_7b": "main", "v2_05b": "main",
    "v2_mix7b": "mixer", "v2_mix05b": "mixer",
    "v3_qwen7b": "crossfamily", "v3_llama8b": "crossfamily", "v3_qwen14b": "crossfamily",
    "J2": "constrained", "J1": "constrained",
}
REGIME = {"J2": "noise20_20pct_label_noise", "J1": "sub25_25pct_data_subset"}


def parse_method_seed(exp, strip_prefixes):
    e = exp
    for p in strip_prefixes:
        if e.startswith(p):
            e = e[len(p):]
            break
    m = re.match(r"(.+)_s(\d+)$", e)
    assert m, f"cannot parse method/seed from {exp!r}"
    return m.group(1), int(m.group(2))


rows_out = []

# --- agg_jizhi.csv : v2 main/mixer, v3 crossfamily, J2 constrained ---
for r in csv.DictReader(open("agg_jizhi.csv")):
    g, exp = r["group"], r["exp"]
    prefixes = {"J2": ["noise20_"], "J1": ["sub25_"]}.get(g, [])
    method, seed = parse_method_seed(exp, prefixes)
    out = {
        "run_id": f"{g}__{exp}", "campaign": CAMPAIGN[g], "model": SCALE_MODEL[g],
        "method": method, "seed": seed, "steps": 300, "regime": REGIME.get(g, ""),
    }
    for c in BENCH_COLS:
        out[c] = r.get(c, "")
    rows_out.append(out)

# --- agg_j1.csv : J1 constrained (sub25) ---
for r in csv.DictReader(open("agg_j1.csv")):
    g, exp = r["group"], r["exp"]
    method, seed = parse_method_seed(exp, ["sub25_"])
    out = {
        "run_id": f"{g}__{exp}", "campaign": CAMPAIGN[g], "model": SCALE_MODEL[g],
        "method": method, "seed": seed, "steps": 300, "regime": REGIME.get(g, ""),
    }
    for c in BENCH_COLS:
        out[c] = r.get(c, "")
    rows_out.append(out)

# --- agg_scale_lt7b.csv : scale-curve (1.5B/3B) + long-training (7B, 1000 steps) ---
for r in csv.DictReader(open("agg_scale_lt7b.csv")):
    g, exp, step = r["group"], r["exp"], int(r["step"])
    if exp.startswith("scale_15b_"):
        model, prefix, campaign = "Qwen2.5-1.5B-Instruct", "scale_15b_", "scale_curve"
    elif exp.startswith("scale_3b_"):
        model, prefix, campaign = "Qwen2.5-3B-Instruct", "scale_3b_", "scale_curve"
    elif exp.startswith("lt7b_"):
        model, prefix, campaign = "Qwen2.5-7B-Instruct", "lt7b_", "long_training"
    else:
        raise ValueError(exp)
    method, seed = parse_method_seed(exp, [prefix])
    out = {
        "run_id": f"{g}__{exp}", "campaign": campaign, "model": model,
        "method": method, "seed": seed, "steps": step, "regime": "",
    }
    for c in BENCH_COLS:
        out[c] = r.get(c, "")
    rows_out.append(out)

fieldnames = ["run_id", "campaign", "model", "method", "seed", "steps", "regime"] + BENCH_COLS
with open("all_runs_171.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

full = sum(1 for r in rows_out if all(r[c] not in ("", None) for c in BENCH_COLS))
print(f"wrote {len(rows_out)} rows to all_runs_171.csv; full-12-benchmark rows = {full}")
from collections import Counter
print("by campaign:", Counter(r["campaign"] for r in rows_out))
