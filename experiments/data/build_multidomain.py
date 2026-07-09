#!/usr/bin/env python3
"""Build the DataFlex-RL 3-domain RLVR training set: math + logic(K&K) + science(SciQ).

Each domain keeps its own `data_source` (drives verl reward routing via the multidomain
reward fn) AND carries a `domain` column (drives DataFlex mix/select, independent of
data_source). Difficulty is heterogeneous within and across domains:
  - math:    gsm8k (easy) + deepscaler (hard competition)
  - logic:   K&K people2 (easy) .. people8 (hard)  -- difficulty = #people
  - science: SciQ 4-option MCQ

verl schema (matches datasets/math/*.parquet):
  data_source, prompt=[{role,content}], ability, reward_model={style,ground_truth}, extra_info, domain

Usage:
  python build_multidomain.py --per-domain 5000 --out ../data/multidomain_3
"""
import argparse
import glob
import json
import os
import random

import pandas as pd

MATH_INSTR = "Let's think step by step and output the final answer within \\boxed{}."
KK_INSTR = ("Reason step by step, then state your final answer as a list like "
            "'X is a knight/knave' for every person.")
SCIQ_INSTR = "Reason step by step, then give the final answer as a single letter (A, B, C, or D)."


def make_row(data_source, question, ground_truth, ability, domain, extra=None):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": extra or {},
        "domain": domain,
    }


# ------------------------------- math -------------------------------
def build_math(root, n, split, rng):
    rows = []
    if split == "train":
        srcs = [("datasets/math/gsm8k_train.parquet", n // 2),
                ("datasets/math/deepscaler_math.parquet", n - n // 2)]
    else:
        srcs = [("datasets/math/gsm8k_train.parquet", n)]  # small held-out slice
    for path, k in srcs:
        df = pd.read_parquet(os.path.join(root, path))
        idx = rng.sample(range(len(df)), min(k, len(df)))
        for i in idx:
            r = df.iloc[i]
            prompt = r["prompt"]
            q = prompt[0]["content"] if isinstance(prompt, (list, tuple)) else str(prompt)
            # strip any source-specific answer instruction, append the dapo `Answer:` one
            for tail in ("Let's think step by step",):
                pos = q.find(tail)
                if pos != -1:
                    q = q[:pos].rstrip()
                    break
            q = f"{q}\n\n{MATH_INSTR}"
            gt = r["reward_model"]["ground_truth"]
            ds = r["data_source"]
            # Route to verl's `math_reward` verifier by using data_source='HuggingFaceH4/MATH-500'
            # (verl __init__.py maps lighteval/MATH-500 IDs -> math_reward, which is latex2sympy-
            # based, accepts \boxed{X}, and matches Qwen2.5-Math benchmark verifiers).
            # OLD (bug fixed 2026-07-09): 'math_dapo' + Answer: instruction — mismatched benchmarks.
            rows.append(make_row("HuggingFaceH4/MATH-500", q, gt, "math", "math",
                                  {"orig_source": str(ds)}))
    return rows


# ------------------------------- logic: K&K -------------------------------
def build_logic(root, n, split, rng):
    rows = []
    files = sorted(glob.glob(os.path.join(root, f"datasets/logic/knights-and-knaves/{split}/*.jsonl")))
    pool = []
    for f in files:
        npeople = int(os.path.basename(f).split("_")[0].replace("people", ""))
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pool.append((npeople, d))
    rng.shuffle(pool)
    for npeople, d in pool[:n]:
        q = f"{d['quiz']}\n\n{KK_INSTR}"
        rows.append(make_row("kk_logic", q, d["solution_text"], "logic", "logic",
                             {"n_people": npeople}))
    return rows


# ------------------------------- science: SciQ -------------------------------
def build_science(root, n, split, rng):
    fn = {"train": "train", "test": "validation"}[split]
    df = pd.read_parquet(os.path.join(root, f"datasets/science/sciq/data/{fn}-00000-of-00001.parquet"))
    rows = []
    idx = rng.sample(range(len(df)), min(n, len(df)))
    for i in idx:
        r = df.iloc[i]
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        labels = "ABCD"
        correct_letter = labels[order.index(0)]
        opt_lines = "\n".join(f"{labels[j]}. {opts[order[j]]}" for j in range(4))
        q = f"{r['question']}\n\n{opt_lines}\n\n{SCIQ_INSTR}"
        rows.append(make_row("sciq", q, correct_letter, "science", "science",
                             {"support": str(r.get("support", ""))[:500]}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--per-domain", type=int, default=5000)
    ap.add_argument("--test-per-domain", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = args.out or os.path.join(args.root, "data/multidomain_3")
    os.makedirs(out, exist_ok=True)

    for split, per in [("train", args.per_domain), ("test", args.test_per_domain)]:
        rows = []
        rows += build_math(args.root, per, split, rng)
        rows += build_logic(args.root, per, split, rng)
        rows += build_science(args.root, per, split, rng)
        rng.shuffle(rows)
        df = pd.DataFrame(rows)
        path = os.path.join(out, f"{split}.parquet")
        df.to_parquet(path)
        counts = df["domain"].value_counts().to_dict()
        print(f"  {split}: {len(df)} rows -> {path}   domains={counts}")


if __name__ == "__main__":
    main()
