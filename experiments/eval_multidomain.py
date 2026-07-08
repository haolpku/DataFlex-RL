#!/usr/bin/env python3
"""Eval a merged HF ckpt on the 3-domain multidomain_3 test set.

Uses the SAME reward verifiers as training (dataflex_verl.rewards.multidomain_reward.compute_score),
so eval numbers are on the exact reward signal the model was trained against.

Output: JSON with per-domain accuracy + macro-avg, plus per-example scores for later inspection.

Usage:
  python eval_multidomain.py --model <hf_dir> --data <parquet> --out <json> [--gpu 0] [--max_tokens 4096]
"""
import argparse, json, os, sys, time
from pathlib import Path

# Ensure our reward fn is importable
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd


def load_test_rows(path):
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        prompt = r["prompt"][0]["content"] if isinstance(r["prompt"], (list, tuple)) else str(r["prompt"])
        rows.append({
            "prompt": prompt,
            "data_source": r["data_source"],
            "ground_truth": r["reward_model"]["ground_truth"],
            "domain": r["domain"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override")
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

    rows = load_test_rows(args.data)
    print(f"[eval] loaded {len(rows)} test rows from {args.data}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    t0 = time.time()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              enforce_eager=False, trust_remote_code=True, max_model_len=args.max_tokens + 1024)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, n=1)
    print(f"[eval] LLM loaded in {time.time()-t0:.1f}s; generating…", flush=True)

    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"[eval] generated {len(outs)} responses in {time.time()-t0:.1f}s", flush=True)

    # Score each output
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
        except Exception as e:
            acc = 0.0; score = 0.0
        per_row.append({"domain": r["domain"], "data_source": r["data_source"],
                         "acc": acc, "score": score, "resp_len": len(resp)})

    # Aggregate
    from collections import defaultdict
    by_domain = defaultdict(list)
    for x in per_row: by_domain[x["domain"]].append(x["acc"])
    dom_acc = {d: sum(v)/len(v) if v else 0.0 for d, v in by_domain.items()}
    macro = sum(dom_acc.values()) / len(dom_acc) if dom_acc else 0.0

    result = {
        "model": args.model,
        "data": args.data,
        "n_total": len(rows),
        "n_by_domain": {d: len(v) for d, v in by_domain.items()},
        "acc_by_domain": dom_acc,
        "macro_avg": macro,
        "per_row": per_row,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    # print short summary
    line = "  ".join(f"{d}={dom_acc[d]*100:.1f}" for d in sorted(dom_acc))
    print(f"[eval] {os.path.basename(args.model.rstrip('/'))}: {line}  macro={macro*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
