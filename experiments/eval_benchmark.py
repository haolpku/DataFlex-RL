#!/usr/bin/env python3
"""Eval a merged HF ckpt on ONE benchmark parquet.

Input: verl-schema parquet (data_source, prompt, ability, reward_model, extra_info).
Uses multidomain_reward.compute_score to route by data_source — verl builtins for math,
kk parser for logic, SciQ/GPQA MCQ letter matching for science.

Output: JSON with per-example scores + summary stats.

Usage:
  python eval_benchmark.py --model <hf_dir> --benchmark <parquet> --out <json> [--gpu 0]
"""
import argparse, json, os, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd


def load_rows(path):
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        prompt = r["prompt"][0]["content"] if isinstance(r["prompt"], (list, tuple)) else str(r["prompt"])
        rows.append({
            "prompt": prompt,
            "data_source": r["data_source"],
            "ground_truth": r["reward_model"]["ground_truth"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True, help="path to verl-schema parquet")
    ap.add_argument("--out", required=True, help="path to output JSON")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_USE_V1", "1")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from dataflex_verl.rewards.multidomain_reward import compute_score

    rows = load_rows(args.benchmark)
    bench_name = os.path.splitext(os.path.basename(args.benchmark))[0]
    print(f"[bench:{bench_name}] {len(rows)} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    t0 = time.time()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              enforce_eager=False, trust_remote_code=True,
              max_model_len=args.max_tokens + 1024)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, n=1)
    print(f"[bench:{bench_name}] LLM loaded in {time.time()-t0:.1f}s; generating…", flush=True)

    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"[bench:{bench_name}] generated {len(outs)} in {time.time()-t0:.1f}s", flush=True)

    per_row = []
    for r, o in zip(rows, outs):
        resp = o.outputs[0].text
        try:
            sc = compute_score(r["data_source"], resp, r["ground_truth"], extra_info={})
            if isinstance(sc, dict):
                acc = 1.0 if sc.get("acc") else 0.0
                score = float(sc.get("score", 0.0))
            else:
                acc = 1.0 if float(sc) > 0 else 0.0
                score = float(sc)
        except Exception:
            acc = 0.0; score = 0.0
        per_row.append({"acc": acc, "score": score, "resp_len": len(resp)})

    n = len(per_row)
    accs = [x["acc"] for x in per_row]
    acc_mean = sum(accs) / n if n else 0.0

    result = {
        "model": args.model,
        "benchmark": bench_name,
        "data_path": args.benchmark,
        "n": n,
        "acc": acc_mean,        # convenience: main scalar
        "per_row": per_row,     # keep for later inspection
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench:{bench_name}] {os.path.basename(args.model.rstrip('/'))}: acc={acc_mean*100:.2f}  (n={n})", flush=True)


if __name__ == "__main__":
    main()
