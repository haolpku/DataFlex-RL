#!/usr/bin/env python3
"""Build the two new per-benchmark parquets that aren't already under benchmarks/:
  - benchmarks/science/gpqa_diamond.parquet   (198 PhD-level MCQ, from CSV)
  - benchmarks/logic/kk_hard.parquet          (people7 + people8 hard held-out, 200 total)

Schema matches verl (data_source, prompt=[{role,content}], ability, reward_model, extra_info).
Reward routing:
  gpqa_diamond -> compute_gpqa in multidomain_reward (choice A-D exact match)
  kk_logic_hard -> compute_kk (existing K&K parser; same ground_truth format as train)
"""
import argparse, glob, json, os, random
import pandas as pd

GPQA_INSTR = "Reason step by step, then give the final answer as a single letter (A, B, C, or D)."
KK_INSTR = ("Reason step by step, then state your final answer as a list like "
            "'X is a knight/knave' for every person.")


def build_gpqa(csv_path, out_path, seed=42):
    df = pd.read_csv(csv_path)
    rng = random.Random(seed)
    rows = []
    for i, r in df.iterrows():
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # strip newlines/whitespace within options
        opts = [str(o).replace("\n", " ").strip() for o in opts]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        labels = "ABCD"
        correct_letter = labels[order.index(0)]
        opt_lines = "\n".join(f"{labels[j]}. {opts[order[j]]}" for j in range(4))
        question = str(r["Question"]).replace("\n", " ").strip()
        prompt = f"{question}\n\n{opt_lines}\n\n{GPQA_INSTR}"
        rows.append({
            "data_source": "gpqa_diamond",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "science",
            "reward_model": {"style": "rule", "ground_truth": correct_letter},
            "extra_info": {"subdomain": str(r.get("Subdomain", "")),
                           "domain": str(r.get("High-level domain", ""))},
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[gpqa] wrote {len(rows)} rows -> {out_path}")


def build_kk_hard(kk_root, out_path):
    """Take K&K people7 (100) + people8 (100) test files as hard held-out."""
    rows = []
    for npeople in [7, 8]:
        fp = os.path.join(kk_root, f"test/people{npeople}_num100.jsonl")
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                q = f"{d['quiz']}\n\n{KK_INSTR}"
                rows.append({
                    "data_source": "kk_logic_hard",
                    "prompt": [{"role": "user", "content": q}],
                    "ability": "logic",
                    "reward_model": {"style": "rule", "ground_truth": d["solution_text"]},
                    "extra_info": {"n_people": npeople},
                })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[kk_hard] wrote {len(rows)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/apdcephfs_zwfy14/share_304380933/aldenliang")
    args = ap.parse_args()
    build_gpqa(f"{args.root}/benchmarks/science/gpqa/gpqa_diamond.csv",
               f"{args.root}/benchmarks/science/gpqa_diamond.parquet")
    build_kk_hard(f"{args.root}/datasets/logic/knights-and-knaves",
                  f"{args.root}/benchmarks/logic/kk_hard.parquet")


if __name__ == "__main__":
    main()
