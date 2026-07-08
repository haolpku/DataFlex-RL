#!/usr/bin/env python3
"""Aggregate per-benchmark eval JSONs into a wide (algo x scale x benchmark) main table.

Reads /jizhicfs/aldenliang/campaign_v1/{7b,05b,mix_7b,mix_05b}/<algo>_s<seed>/global_step_300/actor/huggingface/benchmark_evals/*.json
and produces a wide table with mean±std over seeds per (scale, algo, benchmark).
"""
import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


BENCHMARK_ORDER = [
    # math (easy -> hard)
    "gsm8k", "math_500", "minerva_math", "olympiadbench", "amc23", "aime24", "aime25",
    # science
    "gpqa_diamond", "gpqa_main_minus_diamond", "mmlu_pro_physics", "mmlu_pro_chemistry",
    # logic
    "kk_hard", "bbh_logical_deduction", "bbh_tracking", "zebra_logic_mc",
]


def load_evals(root_pairs):
    """root_pairs: list of (label, path) where path/<run>/global_step_300/actor/huggingface/benchmark_evals/*.json"""
    per_run = []
    for label, root in root_pairs:
        root = Path(root)
        if not root.exists(): continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir(): continue
            name = run_dir.name
            m = name.rsplit("_s", 1)
            if len(m) != 2: continue
            algo, seed = m
            try: seed = int(seed)
            except: continue
            eval_dir = run_dir / "global_step_300" / "actor" / "huggingface" / "benchmark_evals"
            if not eval_dir.exists(): continue
            row = {"scale": label, "algo": algo, "seed": seed}
            for bn in BENCHMARK_ORDER:
                jf = eval_dir / f"{bn}.json"
                if jf.exists():
                    try:
                        d = json.load(open(jf))
                        row[bn] = d.get("acc", 0.0) * 100
                    except Exception:
                        row[bn] = None
                else:
                    row[bn] = None
            per_run.append(row)
    return per_run


def aggregate(per_run):
    """Group by (scale, algo) -> mean±std per benchmark."""
    grouped = defaultdict(list)
    for r in per_run:
        grouped[(r["scale"], r["algo"])].append(r)
    table = []
    for (scale, algo), rows in grouped.items():
        entry = {"scale": scale, "algo": algo, "n_seeds": len(rows)}
        for bn in BENCHMARK_ORDER:
            vals = [r[bn] for r in rows if r.get(bn) is not None]
            entry[f"{bn}_n"] = len(vals)
            entry[f"{bn}_mean"] = statistics.mean(vals) if vals else None
            entry[f"{bn}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        # macro-avg over available benchmarks per run, then mean+/-std across seeds
        macros = []
        for r in rows:
            avail = [r[bn] for bn in BENCHMARK_ORDER if r.get(bn) is not None]
            if avail: macros.append(sum(avail) / len(avail))
        entry["macro_mean"] = statistics.mean(macros) if macros else None
        entry["macro_std"] = statistics.stdev(macros) if len(macros) > 1 else 0.0
        table.append(entry)
    return table


def format_table(table, scales=("7b", "05b", "mix_7b", "mix_05b")):
    lines = []
    for scale in scales:
        rows = [r for r in table if r["scale"] == scale]
        if not rows: continue
        base = next((r for r in rows if r["algo"] == "baseline"), None)
        # For mixer round, use static as "baseline" (uniform mixing)
        if base is None:
            base = next((r for r in rows if r["algo"] == "static"), None)
        base_macro = base["macro_mean"] if base and base["macro_mean"] is not None else None
        rows.sort(key=lambda r: -(r["macro_mean"] or 0.0))
        n_seeds = rows[0]["n_seeds"] if rows else 0
        lines.append(f"\n### {scale.upper()} (n_seeds={n_seeds})\n")
        # header
        header = "| algo | " + " | ".join(BENCHMARK_ORDER) + " | **macro** | Δ vs base |"
        sep = "|" + "|".join(["---"] * (len(BENCHMARK_ORDER) + 3)) + "|"
        lines.append(header); lines.append(sep)
        for r in rows:
            cells = []
            for bn in BENCHMARK_ORDER:
                m = r.get(f"{bn}_mean"); s = r.get(f"{bn}_std", 0.0)
                cells.append(f"{m:.1f}±{s:.1f}" if m is not None else "—")
            m = r.get("macro_mean"); s = r.get("macro_std", 0.0)
            macro_cell = f"**{m:.2f}±{s:.2f}**" if m is not None else "—"
            delta = ""
            if base_macro is not None and m is not None:
                d = m - base_macro
                delta = f"{d:+.2f}" if r["algo"] not in ("baseline", "static") else "—"
            algo_cell = f"**{r['algo']}**" if r["algo"] in ("baseline", "static") else r["algo"]
            lines.append(f"| {algo_cell} | " + " | ".join(cells) + f" | {macro_cell} | {delta} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/jizhicfs/aldenliang/campaign_v1")
    ap.add_argument("--include_mix", action="store_true", help="also include campaign_v1/mix_{7b,05b}/")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    roots = [
        ("7b", os.path.join(args.root, "7b")),
        ("05b", os.path.join(args.root, "05b")),
    ]
    if args.include_mix:
        roots += [
            ("mix_7b", os.path.join(args.root, "mix_7b")),
            ("mix_05b", os.path.join(args.root, "mix_05b")),
        ]

    per_run = load_evals(roots)
    print(f"[aggregate] loaded {len(per_run)} run x benchmark-set entries")
    table = aggregate(per_run)
    md = "# Campaign v1 — Per-Benchmark Main Table\n" + format_table(table)
    print(md)
    if args.out_md:
        with open(args.out_md, "w") as f: f.write(md)
        print(f"[aggregate] wrote {args.out_md}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"per_run": per_run, "aggregate": table}, f, indent=2)
        print(f"[aggregate] wrote {args.out_json}")


if __name__ == "__main__":
    main()
