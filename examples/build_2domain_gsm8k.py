#!/usr/bin/env python3
"""Build a 2-domain GSM8K parquet for the MIX example.

Splits GSM8K by question length into two domains (gsm8k_short / gsm8k_long) as a
proxy for difficulty. IMPORTANT: the real `data_source` column is preserved so
verl's reward function still resolves; the domain label goes in a separate `domain`
column, which the DataFlex mix trainer reads via config.dataflex.domain_key.

Usage:
    python examples/build_2domain_gsm8k.py \
        --src /path/to/data/gsm8k --dst /path/to/data/gsm8k_2domain
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with GSM8K train.parquet/test.parquet")
    ap.add_argument("--dst", required=True, help="output dir")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for split in ["train", "test"]:
        df = pd.read_parquet(f"{args.src}/{split}.parquet").copy()
        qlen = df["extra_info"].apply(lambda e: len(str(e.get("question", ""))))
        med = qlen.median()
        df["domain"] = np.where(qlen <= med, "gsm8k_short", "gsm8k_long")
        df.to_parquet(f"{args.dst}/{split}.parquet")
        print(split, df["domain"].value_counts().to_dict(),
              "| data_source:", list(df["data_source"].unique()))
    print("written to", args.dst)


if __name__ == "__main__":
    main()
