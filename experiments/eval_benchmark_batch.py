#!/usr/bin/env python3
"""Eval ONE merged HF ckpt on MANY benchmarks (amortize LLM cold-start cost).

Loads the LLM once, then for each benchmark parquet: generate, score via
multidomain_reward.compute_score, write per-benchmark JSON. Idempotent (skips
benchmarks whose JSON already exists in the output dir).

Usage:
  python eval_benchmark_batch.py --model <hf_dir> \
    --benchmarks path1.parquet path2.parquet ... \
    --out_dir <hf_dir>/benchmark_evals/ \
    [--gpu 0] [--max_tokens 4096]
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


def score_and_save(rows, outs, out_json, bench_name, model_path, data_path):
    from dataflex_verl.rewards.multidomain_reward import compute_score
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
    acc_mean = sum(x["acc"] for x in per_row) / n if n else 0.0
    result = {"model": model_path, "benchmark": bench_name, "data_path": data_path,
              "n": n, "acc": acc_mean, "per_row": per_row}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench:{bench_name}] acc={acc_mean*100:.2f}  (n={n}) -> {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    # Determine what needs eval'ing (idempotent)
    todo = []
    for bp in args.benchmarks:
        bn = os.path.splitext(os.path.basename(bp))[0]
        out_json = os.path.join(args.out_dir, f"{bn}.json")
        if os.path.exists(out_json):
            print(f"[bench:{bn}] SKIP (already evaled)", flush=True)
            continue
        todo.append((bp, bn, out_json))
    if not todo:
        print("[all benchmarks already evaled, nothing to do]", flush=True)
        return

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_USE_V1", "1")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    t0 = time.time()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              enforce_eager=False, trust_remote_code=True,
              max_model_len=args.max_tokens + 1024)
    print(f"[batch] LLM loaded in {time.time()-t0:.1f}s; {len(todo)} benchmarks to run", flush=True)

    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, n=1)

    for bp, bn, out_json in todo:
        rows = load_rows(bp)
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                    tokenize=False, add_generation_prompt=True)
            for r in rows
        ]
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        print(f"[bench:{bn}] generated {len(outs)} in {time.time()-t0:.1f}s", flush=True)
        score_and_save(rows, outs, out_json, bn, args.model, bp)


if __name__ == "__main__":
    main()
