#!/usr/bin/env python3
"""Convert dapo-math-17k.jsonl -> verl parquet (train/test split).

The jsonl has {prompt: [chat msgs], label: answer}. verl needs the standard columns
data_source / prompt / ability / reward_model / extra_info. We set data_source=
"math_dapo" so verl routes to the math_dapo grader (boxed-answer extraction), which
matches the prompt's "Answer: \\boxed{...}" instruction.

Usage:
  python examples/build_dapo_parquet.py --src <jsonl> --dst <dir> [--n_test 500]
"""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.src) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            prompt = o["prompt"]  # already [{role, content}]
            gt = str(o.get("label", o.get("answer", ""))).strip()
            rows.append({
                "data_source": "math_dapo",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"ground_truth": gt, "style": "rule"},
                "extra_info": {"index": i, "answer": gt},
            })
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_test = min(args.n_test, len(df) // 10)
    test = df.iloc[:n_test].reset_index(drop=True)
    train = df.iloc[n_test:].reset_index(drop=True)

    os.makedirs(args.dst, exist_ok=True)
    train.to_parquet(f"{args.dst}/train.parquet")
    test.to_parquet(f"{args.dst}/test.parquet")
    print(f"train={len(train)}  test={len(test)}  data_source=math_dapo")
    print("sample gt:", train.iloc[0]["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()
