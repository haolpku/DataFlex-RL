#!/usr/bin/env python3
"""Aggregate per-run multidomain_eval.json files into a mean±std main table.

Reads /jizhicfs/aldenliang/campaign_v1/{7b,05b}/<algo>_s<seed>/global_step_300/actor/huggingface/multidomain_eval.json
(and any missing runs from /apdcephfs.../df_ckpts_05b_seeds/*_s3/... if present),
produces:
- per-run rows (scale, algo, seed, math, logic, science, macro)
- aggregate table (scale x algo -> mean ± std per domain + macro)
- ranked list vs baseline
"""
import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


ALGOS = ["baseline", "ar", "difffilter", "gfpo", "maxvar", "topk", "per", "softmax", "diffband"]
DOMAINS = ["math", "logic", "science"]


def load_evals(roots):
    """roots: list of (label, path) where path contains <run>_s<seed>/... paths."""
    per_run = []  # list of dicts
    for label, root in roots:
        root = Path(root)
        if not root.exists(): continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir(): continue
            name = run_dir.name  # e.g. baseline_s1
            m = name.rsplit("_s", 1)
            if len(m) != 2: continue
            algo, seed = m
            try: seed = int(seed)
            except: continue
            js = run_dir / "global_step_300" / "actor" / "huggingface" / "multidomain_eval.json"
            if not js.exists(): continue
            try:
                d = json.load(open(js))
            except: continue
            row = {"scale": label, "algo": algo, "seed": seed,
                   "macro": d.get("macro_avg", 0.0) * 100}
            for dom in DOMAINS:
                row[dom] = d.get("acc_by_domain", {}).get(dom, 0.0) * 100
            per_run.append(row)
    return per_run


def aggregate(per_run):
    """Group by (scale, algo) -> mean±std per domain + macro."""
    grouped = defaultdict(list)
    for r in per_run:
        grouped[(r["scale"], r["algo"])].append(r)
    table = []
    for (scale, algo), rows in grouped.items():
        row = {"scale": scale, "algo": algo, "n_seeds": len(rows)}
        for col in DOMAINS + ["macro"]:
            vals = [r[col] for r in rows]
            row[f"{col}_mean"] = statistics.mean(vals) if vals else 0.0
            row[f"{col}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        table.append(row)
    return table


def format_table(table, scale_filter=None):
    """Print mean±std table for a scale, ordered by macro-avg desc, with baseline diff."""
    lines = []
    for scale in (["7b", "05b"] if scale_filter is None else [scale_filter]):
        rows = [r for r in table if r["scale"] == scale]
        if not rows: continue
        # find baseline
        base = next((r for r in rows if r["algo"] == "baseline"), None)
        base_macro = base["macro_mean"] if base else 0.0
        rows.sort(key=lambda r: -r["macro_mean"])
        lines.append(f"\n### {scale.upper()} (n_seeds={rows[0]['n_seeds'] if rows else '?'})\n")
        lines.append("| algo | math | logic | science | macro | Δ vs baseline |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            delta = r["macro_mean"] - base_macro
            delta_str = f"{delta:+.2f}" if r["algo"] != "baseline" else "—"
            lines.append(
                f"| {'**' if r['algo']=='baseline' else ''}{r['algo']}{'**' if r['algo']=='baseline' else ''} "
                f"| {r['math_mean']:.1f}±{r['math_std']:.1f} "
                f"| {r['logic_mean']:.1f}±{r['logic_std']:.1f} "
                f"| {r['science_mean']:.1f}±{r['science_std']:.1f} "
                f"| {r['macro_mean']:.2f}±{r['macro_std']:.2f} "
                f"| {delta_str} |"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jizhi_root", default="/jizhicfs/aldenliang/campaign_v1")
    ap.add_argument("--local_root", default="/apdcephfs_zwfy14/share_304380933/aldenliang/df_ckpts_05b_seeds",
                    help="fallback root for local-box s3 runs (if merged there)")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    roots = [
        ("7b", os.path.join(args.jizhi_root, "7b")),
        ("05b", os.path.join(args.jizhi_root, "05b")),
    ]
    # Note: local s3 runs are NOT merged yet (per handoff), so no eval JSON expected there.
    per_run = load_evals(roots)
    print(f"[aggregate] loaded {len(per_run)} run evals", flush=True)

    table = aggregate(per_run)
    md = "# Campaign v1 Main Table (3-domain eval)\n" + format_table(table)
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
