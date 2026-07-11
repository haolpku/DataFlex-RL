#!/usr/bin/env python3
"""Aggregate v2 eval results: math (Qwen2.5-Math) + GPQA (opencompass) into one table."""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/jizhicfs/aldenliang")
QWEN_OUT_BASE = Path("/jizhicfs/aldenliang/frameworks/Qwen2.5-Math/evaluation/outputs")
MATH_DATASETS = ["math", "aime24", "olympiadbench", "minerva_math", "gsm8k"]
CKPT_GLOB = list(ROOT.glob("campaign_v2/*/*/global_step_300/actor/huggingface"))


def parse_math_metrics(ckpt_dir: Path):
    """Return {dataset: acc_pct or None} from qwen-eval outputs mirror.
    qwen-eval writes to <QWEN_OUT_BASE>/<absolute_ckpt_path>/math_eval/<ds>/*_metrics.json
    """
    # Mirror: /jizhicfs/aldenliang/frameworks/Qwen2.5-Math/evaluation/outputs + ckpt_dir
    # Python paths: str(ckpt_dir) starts with '/', joining strips leading '/', so use str concat.
    mirror = Path(str(QWEN_OUT_BASE) + str(ckpt_dir))
    out = {}
    for ds in MATH_DATASETS:
        candidates = list((mirror / "math_eval" / ds).glob("*_metrics.json"))
        if not candidates:
            out[ds] = None
            continue
        try:
            with open(candidates[0]) as f:
                d = json.load(f)
            acc = d.get("acc") or d.get("accuracy") or d.get("score")
            if isinstance(acc, dict):
                acc = acc.get("acc") or acc.get("accuracy") or list(acc.values())[0]
            out[ds] = float(acc) if acc is not None else None
        except Exception:
            out[ds] = None
    return out


def parse_gpqa_result(exp: str):
    """Return gpqa accuracy or None."""
    csv_path = ROOT / "queue_eval_gpqa/results" / f"{exp}.csv"
    if not csv_path.exists():
        return None
    try:
        # opencompass CSV: header,version,metric,mode,<model>
        for line in csv_path.read_text().splitlines():
            if "GPQA" in line or "gpqa" in line:
                parts = [p.strip() for p in line.split(",")]
                # last column is the accuracy
                try:
                    return float(parts[-1])
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return None


def parse_exp(ckpt_dir: Path):
    """From .../campaign_v2/7b/ar_s1/global_step_300/actor/huggingface -> (scale='7b', name='ar', seed=1)"""
    parts = ckpt_dir.parts
    scale = parts[-5]
    exp = parts[-4]  # 'ar_s1'
    m = re.match(r"(.+)_s(\d+)$", exp)
    if m:
        return scale, m.group(1), int(m.group(2)), exp
    return scale, exp, 0, exp


def main():
    rows = []
    for ckpt in sorted(CKPT_GLOB):
        scale, name, seed, exp = parse_exp(ckpt)
        exp_key = f"{scale}__{exp}"
        math_res = parse_math_metrics(ckpt)
        gpqa = parse_gpqa_result(exp_key)
        row = {"scale": scale, "name": name, "seed": seed, "exp": exp, **math_res, "gpqa": gpqa}
        rows.append(row)

    # Sort: scale then name then seed
    rows.sort(key=lambda r: (r["scale"], r["name"], r["seed"]))

    # Format table
    cols = ["scale", "name", "seed"] + MATH_DATASETS + ["gpqa"]
    widths = {c: max(len(c), max((len(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c] or "-")) for r in rows), default=1)) for c in cols}

    print("\n=== DataFlex-RL v2 Full Eval Table (Qwen2.5-Math + opencompass GPQA) ===\n")
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if v is None:
                s = "-"
            elif isinstance(v, float):
                s = f"{v:.1f}"
            else:
                s = str(v)
            cells.append(s.ljust(widths[c]))
        print(" | ".join(cells))

    # Progress
    done_math = sum(1 for r in rows if any(r[d] is not None for d in MATH_DATASETS))
    done_gpqa = sum(1 for r in rows if r["gpqa"] is not None)
    print(f"\nProgress: math {done_math}/{len(rows)} | gpqa {done_gpqa}/{len(rows)}")

    # Also write CSV
    csv_path = ROOT / "campaign_v2/eval_summary.csv"
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) if r[c] is not None else "" for c in cols) + "\n")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
