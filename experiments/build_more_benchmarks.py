#!/usr/bin/env python3
"""Build the 5 additional benchmark parquets (per user's '中弋公' choice):
  - GPQA main\\diamond   -> benchmarks/science/gpqa_main_minus_diamond.parquet  (~250)
  - MMLU-Pro physics    -> benchmarks/science/mmlu_pro_physics.parquet         (sampled 500)
  - MMLU-Pro chemistry  -> benchmarks/science/mmlu_pro_chemistry.parquet       (sampled 500)
  - BBH logical_deduction (3+5+7 objects) -> benchmarks/logic/bbh_logical_deduction.parquet (~750)
  - BBH tracking_shuffled_objects (3+5+7) -> benchmarks/logic/bbh_tracking.parquet         (~750)
  - ZebraLogic MC       -> benchmarks/logic/zebra_logic_mc.parquet             (sampled 500)

All in verl-schema (data_source, prompt=[{role,content}], ability, reward_model, extra_info).
Reward routing via multidomain_reward (letters A-D/A-J are all treated by compute_sciq;
BBH mcq answers are wrapped in parens so we canonicalize before storing GT).
"""
import argparse
import json
import os
import random

import pandas as pd
from pathlib import Path


MCQ_INSTR = ("Reason step by step, then give the final answer as a single letter "
             "(e.g., 'The answer is: A').")


def make_row(data_source, prompt, ground_truth, ability, extra=None):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": extra or {},
    }


# =============================================================================
# 1) GPQA main\diamond  (uses local CSVs; identifies diamond via Question text)
# =============================================================================
def build_gpqa_main_minus_diamond(root, out_path, seed=42):
    diamond = pd.read_csv(f"{root}/benchmarks/science/gpqa/gpqa_diamond.csv")
    main    = pd.read_csv(f"{root}/benchmarks/science/gpqa/gpqa_main.csv")
    diamond_qs = set(diamond["Question"].astype(str).tolist())
    non_diamond = main[~main["Question"].astype(str).isin(diamond_qs)].reset_index(drop=True)
    print(f"[gpqa_main_minus_diamond] diamond={len(diamond)} main={len(main)} disjoint={len(non_diamond)}")
    rng = random.Random(seed)
    rows = []
    GPQA_INSTR = "Reason step by step, then give the final answer as a single letter (A, B, C, or D)."
    for _, r in non_diamond.iterrows():
        opts = [str(r["Correct Answer"]), str(r["Incorrect Answer 1"]),
                str(r["Incorrect Answer 2"]), str(r["Incorrect Answer 3"])]
        opts = [o.replace("\n", " ").strip() for o in opts]
        order = [0, 1, 2, 3]; rng.shuffle(order)
        labels = "ABCD"
        correct = labels[order.index(0)]
        opt_lines = "\n".join(f"{labels[j]}. {opts[order[j]]}" for j in range(4))
        q = str(r["Question"]).replace("\n", " ").strip()
        prompt = f"{q}\n\n{opt_lines}\n\n{GPQA_INSTR}"
        rows.append(make_row("gpqa_main", prompt, correct, "science",
                              {"subdomain": str(r.get("Subdomain", "")),
                               "domain": str(r.get("High-level domain", ""))}))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[gpqa_main] wrote {len(rows)} rows -> {out_path}")


# =============================================================================
# 2 & 3) MMLU-Pro physics + chemistry (HF via proxy)
# =============================================================================
def build_mmlu_pro(category, out_path, n_sample=500, seed=42):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.filter(lambda x: x["category"] == category)
    print(f"[mmlu_pro:{category}] full test = {len(ds)}")
    idx = list(range(len(ds)))
    if n_sample and len(idx) > n_sample:
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n_sample])
    labels = "ABCDEFGHIJ"
    rows = []
    for i in idx:
        item = ds[i]
        opts = item["options"]  # list, may contain 'N/A' placeholders
        answer_letter = item["answer"]   # e.g. 'B'
        opt_lines = []
        for k, opt in enumerate(opts):
            if opt == "N/A": continue
            opt_lines.append(f"{labels[k]}. {opt}")
        prompt = (f"{item['question']}\n\n" + "\n".join(opt_lines) +
                  "\n\n" + MCQ_INSTR)
        rows.append(make_row(f"mmlu_pro_{category}", prompt, answer_letter, "science",
                              {"category": category, "src_idx": i}))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[mmlu_pro:{category}] wrote {len(rows)} rows -> {out_path}")


# =============================================================================
# 4 & 5) BBH — logical_deduction + tracking_shuffled_objects, all 3 sizes
# =============================================================================
def build_bbh(subtasks, out_path, ds_name):
    """Load BBH subtask JSONs from HF (lukaemon/bbh)."""
    from datasets import load_dataset
    rows = []
    for st in subtasks:
        ds = load_dataset("lukaemon/bbh", st, split="test")
        for i, item in enumerate(ds):
            q = item["input"]
            # BBH answer format: e.g. "(A)" or a raw string. For MCQ subtasks it's "(A)-(G)".
            ans = str(item["target"]).strip()
            # Extract letter if wrapped in parens
            import re
            m = re.match(r"^\(([A-Z])\)$", ans)
            gt = m.group(1) if m else ans
            prompt = f"{q}\n\nReason step by step. When you are done, write 'The answer is: (X)' where X is the letter of your final choice."
            rows.append(make_row(ds_name, prompt, gt, "logic",
                                  {"subtask": st, "src_idx": i}))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[{ds_name}] wrote {len(rows)} rows -> {out_path}")


# =============================================================================
# 6) ZebraLogic-MC (HF)
# =============================================================================
def build_zebra_logic(out_path, n_sample=500, seed=42):
    from datasets import load_dataset
    ds = load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
    print(f"[zebra_mc] full test = {len(ds)}")
    idx = list(range(len(ds)))
    if n_sample and len(idx) > n_sample:
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n_sample])
    labels = "ABCDEFGHIJ"
    rows = []
    for i in idx:
        item = ds[i]
        choices = item["choices"]
        answer_str = item["answer"]
        # find letter
        if answer_str in choices:
            gt = labels[choices.index(answer_str)]
        else:
            # sometimes the answer field is already a letter
            gt = answer_str if answer_str in labels else "?"
        opt_lines = " ".join(f"({labels[k]}) {c}" for k, c in enumerate(choices))
        prompt = (f"The following is a logic grid puzzle. Read the clues and answer.\n\n"
                  f"Puzzle:\n{item['puzzle']}\n\n"
                  f"Question: {item['question']}\n\n"
                  f"Choices: {opt_lines}\n\n"
                  f"Reason step by step. When you provide the final answer, "
                  f"use the prefix 'The answer is:' followed by only the answer letter "
                  f"(e.g., 'The answer is: A').")
        rows.append(make_row("zebra_logic_mc", prompt, gt, "logic",
                              {"src_idx": i}))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[zebra_mc] wrote {len(rows)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/apdcephfs_zwfy14/share_304380933/aldenliang")
    ap.add_argument("--mmlu_n", type=int, default=500)
    ap.add_argument("--zebra_n", type=int, default=500)
    args = ap.parse_args()

    # 1
    build_gpqa_main_minus_diamond(args.root,
        f"{args.root}/benchmarks/science/gpqa_main_minus_diamond.parquet")
    # 2, 3
    build_mmlu_pro("physics",   f"{args.root}/benchmarks/science/mmlu_pro_physics.parquet",   n_sample=args.mmlu_n)
    build_mmlu_pro("chemistry", f"{args.root}/benchmarks/science/mmlu_pro_chemistry.parquet", n_sample=args.mmlu_n)
    # 4
    build_bbh(
        ["logical_deduction_three_objects","logical_deduction_five_objects","logical_deduction_seven_objects"],
        f"{args.root}/benchmarks/logic/bbh_logical_deduction.parquet",
        ds_name="bbh_logical_deduction",
    )
    # 5
    build_bbh(
        ["tracking_shuffled_objects_three_objects","tracking_shuffled_objects_five_objects","tracking_shuffled_objects_seven_objects"],
        f"{args.root}/benchmarks/logic/bbh_tracking.parquet",
        ds_name="bbh_tracking",
    )
    # 6
    build_zebra_logic(f"{args.root}/benchmarks/logic/zebra_logic_mc.parquet", n_sample=args.zebra_n)


if __name__ == "__main__":
    main()
