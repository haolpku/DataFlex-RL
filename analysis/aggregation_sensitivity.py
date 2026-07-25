"""Aggregation sensitivity analysis (P0.1).

Purpose: recompute method-vs-baseline effects and rank stability under 8
different aggregation rules over the 12 released benchmarks, so we can
report which conclusions are invariant vs which depend on the aggregation
choice.

Reads:  ../results/all_runs_171.csv
Writes: outputs/aggregation_effects.csv     — one row per (aggregation, method, campaign, model, regime)
        outputs/rank_correlations.csv       — Spearman/Kendall between primary aggregate and each alternative
        outputs/leave_one_out.csv           — primary comparison recomputed after removing each benchmark
        outputs/aggregate_definitions.json  — machine-readable spec of every rule
"""
from __future__ import annotations
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_IN = HERE.parent / "results" / "all_runs_171.csv"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MATH = ["math500", "aime24", "olympiadbench", "minerva_math", "gsm8k"]
LOGIC = ["kk_hard", "bbh_logical_deduction", "bbh_tracking", "zebra_logic_mc"]
SCI = ["gpqa_diamond", "mmlu_pro_chemistry", "mmlu_pro_physics"]
ALL12 = MATH + LOGIC + SCI
LEGACY6 = MATH + ["gpqa_diamond"]
BENCH_ITEMS = {
    "math500": 500, "aime24": 30, "olympiadbench": 675, "minerva_math": 272, "gsm8k": 1319,
    "kk_hard": 200, "bbh_logical_deduction": 750, "bbh_tracking": 750, "zebra_logic_mc": 500,
    "gpqa_diamond": 198, "mmlu_pro_chemistry": 500, "mmlu_pro_physics": 500,
}
DOMAIN_OF = {b: "MATH" for b in MATH} | {b: "LOGIC" for b in LOGIC} | {b: "SCIENCE" for b in SCI}


def load():
    rows = list(csv.DictReader(open(CSV_IN)))
    for r in rows:
        for b in ALL12:
            v = r.get(b, "")
            r[b] = float(v) if v not in ("", None) else None
        r["seed"] = int(r["seed"])
        r["steps"] = int(r["steps"])
    return rows


# ---------- aggregation rules ----------
def agg_legacy6(row):
    vs = [row[b] for b in LEGACY6 if row.get(b) is not None]
    return sum(vs) / len(vs) if len(vs) == 6 else None


def agg_macro12(row):
    vs = [row[b] for b in ALL12 if row.get(b) is not None]
    return sum(vs) / len(vs) if len(vs) == 12 else None


def agg_domain_balanced(row):
    if any(row.get(b) is None for b in ALL12):
        return None
    m = sum(row[b] for b in MATH) / len(MATH)
    l = sum(row[b] for b in LOGIC) / len(LOGIC)
    s = sum(row[b] for b in SCI) / len(SCI)
    return (m + l + s) / 3


def agg_item_micro(row):
    if any(row.get(b) is None for b in ALL12):
        return None
    num = sum(row[b] * BENCH_ITEMS[b] / 100.0 for b in ALL12)
    den = sum(BENCH_ITEMS[b] for b in ALL12)
    return 100.0 * num / den


def agg_no_aime(row):
    cols = [b for b in ALL12 if b != "aime24"]
    vs = [row[b] for b in cols if row.get(b) is not None]
    return sum(vs) / len(vs) if len(vs) == len(cols) else None


def agg_no_gpqa(row):
    cols = [b for b in ALL12 if b != "gpqa_diamond"]
    vs = [row[b] for b in cols if row.get(b) is not None]
    return sum(vs) / len(vs) if len(vs) == len(cols) else None


def agg_no_small(row):
    cols = [b for b in ALL12 if b not in ("aime24", "gpqa_diamond")]
    vs = [row[b] for b in cols if row.get(b) is not None]
    return sum(vs) / len(vs) if len(vs) == len(cols) else None


AGGREGATES = {
    "domain_balanced_12": agg_domain_balanced,   # PRIMARY going forward
    "macro_mean_12": agg_macro12,
    "legacy_mean_6": agg_legacy6,
    "item_weighted_micro_12": agg_item_micro,
    "no_aime24": agg_no_aime,
    "no_gpqa_diamond": agg_no_gpqa,
    "no_small_sets": agg_no_small,
}


# ---------- per (method, model, campaign, regime) mean aggregate ----------
def group_key(r):
    return (r["campaign"], r["model"], r["regime"], r["method"])


def method_means(rows, agg_fn):
    per_run = {}
    for r in rows:
        v = agg_fn(r)
        if v is None:
            continue
        per_run[(r["campaign"], r["model"], r["regime"], r["method"], r["seed"])] = v
    means = defaultdict(list)
    for (c, m, rg, meth, s), v in per_run.items():
        means[(c, m, rg, meth)].append(v)
    return {k: sum(vs) / len(vs) for k, vs in means.items()}, per_run


# ---------- paired method-vs-baseline diffs ----------
def paired_diffs(rows, agg_fn):
    per_run = {}
    for r in rows:
        v = agg_fn(r)
        if v is None:
            continue
        per_run.setdefault((r["campaign"], r["model"], r["regime"], r["seed"]), {})[r["method"]] = v
    diffs = defaultdict(list)
    for (c, m, rg, s), by_method in per_run.items():
        if "baseline" not in by_method:
            continue
        base = by_method["baseline"]
        for meth, v in by_method.items():
            if meth == "baseline":
                continue
            diffs[(c, m, rg, meth)].append(v - base)
    return diffs


# ---------- rank correlation ----------
def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    def rank(a):
        s = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        for i, idx in enumerate(s):
            r[idx] = i + 1
        # simple tie-blind
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else None


def kendall(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    c = d = 0
    for i, j in combinations(range(n), 2):
        s = (xs[i] - xs[j]) * (ys[i] - ys[j])
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
    total = n * (n - 1) / 2
    return (c - d) / total if total > 0 else None


# ---------- main ----------
def main():
    rows = load()
    print(f"loaded {len(rows)} runs from {CSV_IN}")

    # 1) aggregation_effects.csv — every (agg, group, method) with mean + paired-diff-vs-baseline
    rows_out = []
    for agg_name, agg_fn in AGGREGATES.items():
        means, _ = method_means(rows, agg_fn)
        diffs = paired_diffs(rows, agg_fn)
        for (c, m, rg, meth), mean_val in means.items():
            d_list = diffs.get((c, m, rg, meth), [])
            mean_d = sum(d_list) / len(d_list) if d_list else None
            sd_d = None
            if len(d_list) > 1:
                mu = mean_d
                sd_d = (sum((x - mu) ** 2 for x in d_list) / (len(d_list) - 1)) ** 0.5
            rows_out.append({
                "aggregation": agg_name, "campaign": c, "model": m, "regime": rg,
                "method": meth, "n_seed": len(d_list), "mean_agg": round(mean_val, 3),
                "mean_delta_vs_baseline": round(mean_d, 3) if mean_d is not None else "",
                "sd_delta": round(sd_d, 3) if sd_d is not None else "",
            })
    with open(OUT / "aggregation_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows -> aggregation_effects.csv")

    # 2) rank_correlations.csv — 7B-Instruct main+mixer 13-method ranking, primary vs alternatives
    key = lambda r: (r["campaign"] in ("main", "mixer")) and r["model"] == "Qwen2.5-7B-Instruct"
    subset = [r for r in rows if key(r)]
    per_agg_ranking = {}
    for agg_name, agg_fn in AGGREGATES.items():
        means, _ = method_means(subset, agg_fn)
        # collapse over campaign so each method has one mean at 7B-Instruct
        by_method = defaultdict(list)
        for (c, m, rg, meth), v in means.items():
            by_method[meth].append(v)
        per_agg_ranking[agg_name] = {meth: sum(vs) / len(vs) for meth, vs in by_method.items()}

    methods = sorted(per_agg_ranking["domain_balanced_12"].keys())
    primary = [per_agg_ranking["domain_balanced_12"][m] for m in methods]
    corr_rows = []
    for agg_name in AGGREGATES:
        if agg_name == "domain_balanced_12":
            continue
        alt = [per_agg_ranking[agg_name].get(m) for m in methods]
        if any(v is None for v in alt):
            continue
        corr_rows.append({
            "aggregation_vs_primary": agg_name,
            "n_methods": len(methods),
            "spearman_rho": round(spearman(primary, alt), 4),
            "kendall_tau": round(kendall(primary, alt), 4),
        })
    with open(OUT / "rank_correlations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader()
        w.writerows(corr_rows)
    print(f"wrote {len(corr_rows)} rank correlations -> rank_correlations.csv")

    # 3) leave_one_out.csv — 7B-Instruct main-round, drop each benchmark and recompute domain-balanced
    subset_main = [r for r in rows if r["campaign"] == "main" and r["model"] == "Qwen2.5-7B-Instruct"]
    loo_rows = []
    for drop in ALL12:
        def agg_loo(row, drop=drop):
            cols = [b for b in ALL12 if b != drop]
            if any(row.get(b) is None for b in cols):
                return None
            groups = defaultdict(list)
            for b in cols:
                groups[DOMAIN_OF[b]].append(row[b])
            dom_means = [sum(vs) / len(vs) for vs in groups.values()]
            return sum(dom_means) / len(dom_means)
        diffs = paired_diffs(subset_main, agg_loo)
        for (c, m, rg, meth), d_list in diffs.items():
            if not d_list:
                continue
            mean_d = sum(d_list) / len(d_list)
            loo_rows.append({
                "dropped_benchmark": drop, "method": meth, "n_seed": len(d_list),
                "mean_delta_vs_baseline": round(mean_d, 3),
            })
    with open(OUT / "leave_one_out.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loo_rows[0].keys()))
        w.writeheader()
        w.writerows(loo_rows)
    print(f"wrote {len(loo_rows)} leave-one-out rows -> leave_one_out.csv")

    # 4) aggregate_definitions.json — machine-readable spec
    defn = {
        "domain_balanced_12": {
            "description": "average within {MATH, LOGIC, SCIENCE} first, then equal weight across 3 domains",
            "role": "PRIMARY for the multi-seed extension and going forward",
            "benchmarks": {"MATH": MATH, "LOGIC": LOGIC, "SCIENCE": SCI},
        },
        "macro_mean_12": {"description": "equal weight for each of 12 benchmarks"},
        "legacy_mean_6": {"description": "equal weight over MATH-500/AIME24/Olympiad/Minerva/GSM8K/GPQA", "role": "the pre-revision headline, kept as sensitivity"},
        "item_weighted_micro_12": {"description": "sum of correct-count across 12 sets divided by sum of set sizes"},
        "no_aime24": {"description": "12-macro without AIME24 (n=11)"},
        "no_gpqa_diamond": {"description": "12-macro without GPQA-Diamond (n=11)"},
        "no_small_sets": {"description": "12-macro without both AIME24 and GPQA-Diamond (n=10)"},
    }
    with open(OUT / "aggregate_definitions.json", "w") as f:
        json.dump(defn, f, indent=2)
    print("wrote aggregate_definitions.json")

    print("\nP0.1 aggregation sensitivity complete.")


if __name__ == "__main__":
    main()
