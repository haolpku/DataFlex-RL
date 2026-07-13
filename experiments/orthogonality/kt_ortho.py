#!/usr/bin/env python3
"""Paper 1 Option B: Offline k_t orthogonality analysis.

Take a trained student checkpoint + a teacher, roll out N prompts × G=5 rollouts each
on multidomain_3, and compute per-rollout:
  - k_t   = student_logp(sampled) - teacher_logp(sampled)  (mean over response tokens)
  - reward = 0/1 correctness from multidomain_reward
  - |A|    = |group-normalized advantage|

Then report:
  - Pearson correlation of k_t vs reward, k_t vs |A|, reward vs |A|
  - Scatter plots
  - Rescue examples: high k_t & low reward (wrong-but-teachable)

Usage:
  python offline_kt_ortho.py --student <path> --teacher <path> --data <parquet> \
    --n_prompts 100 --n_rollouts 5 --out results/kt_ortho/
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="student HF dir (has safetensors)")
    p.add_argument("--teacher", required=True, help="teacher HF dir")
    p.add_argument("--data", required=True, help="parquet with 'prompt' column")
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument("--n_rollouts", type=int, default=5)
    p.add_argument("--max_response", type=int, default=512)
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--reward_py", default="/apdcephfs_zwfy14/share_304380933/qifengcai/old/DataFlex-RL-opd/src/dataflex_verl/rewards/multidomain_reward.py")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_reward_fn(path):
    """Import compute_score from multidomain_reward.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_score


def get_logp(model, tokenizer, prompt_ids, response_ids, device):
    """Compute per-token logp of response given prompt.

    Returns tensor of shape (response_len,) = logp of each sampled response token.
    """
    input_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0, :-1]  # (seq_len-1, vocab)
    targets = input_ids[0, 1:]  # (seq_len-1,)
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)
    logp_at_target = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (seq_len-1,)
    # response starts at position prompt_len in original, so shift-by-1 aligned starts at prompt_len-1
    resp_start = len(prompt_ids) - 1
    resp_logp = logp_at_target[resp_start:resp_start + len(response_ids)]
    return resp_logp.cpu().numpy()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f">>> Loading student vLLM: {args.student}")
    student_vllm = LLM(
        model=args.student,
        gpu_memory_utilization=0.4,  # leave room for teacher
        tensor_parallel_size=1,
        seed=args.seed,
        enforce_eager=True,
        max_model_len=2048,
    )

    print(f">>> Loading student HF (for logp): {args.student}")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    student_hf = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="cuda:1"
    )
    student_hf.eval()

    print(f">>> Loading teacher HF (for logp): {args.teacher}")
    teacher_hf = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map="cuda:2"
    )
    teacher_hf.eval()

    # Load data
    df = pd.read_parquet(args.data)
    if len(df) > args.n_prompts:
        df = df.sample(args.n_prompts, random_state=args.seed).reset_index(drop=True)
    print(f">>> Loaded {len(df)} prompts")

    reward_fn = load_reward_fn(args.reward_py)

    # Roll out
    print(f">>> Rolling out {args.n_prompts} x {args.n_rollouts}")
    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        n=args.n_rollouts, max_tokens=args.max_response, seed=args.seed,
    )
    records = []
    for i, row in df.iterrows():
        prompt_data = row["prompt"]
        # prompt may be list of chat msgs or string
        if isinstance(prompt_data, (list, np.ndarray)):
            prompt_text = tokenizer.apply_chat_template(
                list(prompt_data), tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = str(prompt_data)

        out_vllm = student_vllm.generate([prompt_text], sampling)
        rollouts = out_vllm[0].outputs

        prompt_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long)

        group_rewards = []
        group_kts = []
        for r in rollouts:
            resp_text = r.text
            resp_ids = torch.tensor(r.token_ids, dtype=torch.long)
            # skip empty
            if len(resp_ids) < 2:
                group_rewards.append(0.0)
                group_kts.append(0.0)
                continue

            # Student logp
            s_logp = get_logp(student_hf, tokenizer, prompt_ids, resp_ids, "cuda:1")
            # Teacher logp
            t_logp = get_logp(teacher_hf, tokenizer, prompt_ids, resp_ids, "cuda:2")
            # k_t = student - teacher, mean over response tokens
            L = min(len(s_logp), len(t_logp))
            k_t = float((s_logp[:L] - t_logp[:L]).mean())

            # Reward via reward_fn
            data_source = row.get("data_source", "math_dapo")
            ground_truth = row.get("reward_model", {}).get("ground_truth", row.get("ground_truth", ""))
            if isinstance(ground_truth, dict):
                ground_truth = ground_truth.get("ground_truth", "")
            try:
                rew = reward_fn(data_source, resp_text, ground_truth, {})
                if isinstance(rew, dict):
                    rew_val = rew.get("score", rew.get("acc", 0.0))
                else:
                    rew_val = float(rew)
            except Exception as e:
                rew_val = 0.0

            group_rewards.append(rew_val)
            group_kts.append(k_t)

        # Advantage = reward - group_mean
        group_mean = float(np.mean(group_rewards)) if group_rewards else 0.0
        for rew, kt in zip(group_rewards, group_kts):
            records.append({
                "prompt_idx": i,
                "reward": rew,
                "k_t": kt,
                "advantage": rew - group_mean,
                "abs_advantage": abs(rew - group_mean),
                "group_mean_reward": group_mean,
                "data_source": row.get("data_source", "unknown"),
            })

        if (i + 1) % 10 == 0:
            print(f"  ...processed {i+1}/{len(df)} prompts, records so far={len(records)}")

    df_out = pd.DataFrame(records)
    df_out.to_csv(out / "records.csv", index=False)
    print(f">>> Wrote {len(df_out)} records to {out}/records.csv")

    # Correlations
    from scipy.stats import pearsonr, spearmanr
    corrs = {}
    for a, b in [("k_t", "reward"), ("k_t", "abs_advantage"), ("reward", "abs_advantage")]:
        pr, pp = pearsonr(df_out[a], df_out[b])
        sr, sp = spearmanr(df_out[a], df_out[b])
        corrs[f"{a}_vs_{b}"] = {"pearson": (float(pr), float(pp)), "spearman": (float(sr), float(sp))}
    print("\n=== Correlations ===")
    for k, v in corrs.items():
        print(f"  {k}: pearson r={v['pearson'][0]:+.3f} (p={v['pearson'][1]:.3g}), "
              f"spearman={v['spearman'][0]:+.3f} (p={v['spearman'][1]:.3g})")

    (out / "correlations.json").write_text(json.dumps(corrs, indent=2))

    # Rescue examples: low reward (< 0.5) + high k_t (top quartile)
    q75_kt = df_out["k_t"].quantile(0.75)
    rescue = df_out[(df_out["reward"] < 0.5) & (df_out["k_t"] > q75_kt)]
    print(f"\n=== Rescue candidates (reward<0.5 AND k_t>Q75={q75_kt:.3f}) ===")
    print(f"  {len(rescue)} of {len(df_out)} records ({100*len(rescue)/len(df_out):.1f}%)")
    rescue.head(20).to_csv(out / "rescue_examples.csv", index=False)

    # Simple text summary
    summary = f"""
=== k_t Orthogonality Analysis ===
Records: {len(df_out)}
Domains: {df_out['data_source'].value_counts().to_dict()}

Correlations:
  k_t vs reward:        pearson={corrs['k_t_vs_reward']['pearson'][0]:+.3f}, spearman={corrs['k_t_vs_reward']['spearman'][0]:+.3f}
  k_t vs |advantage|:   pearson={corrs['k_t_vs_abs_advantage']['pearson'][0]:+.3f}, spearman={corrs['k_t_vs_abs_advantage']['spearman'][0]:+.3f}
  reward vs |advantage|: pearson={corrs['reward_vs_abs_advantage']['pearson'][0]:+.3f}, spearman={corrs['reward_vs_abs_advantage']['spearman'][0]:+.3f}

Interpretation:
  |pearson(k_t, reward)| < 0.3     => k_t is (weakly) orthogonal to reward ✓
  |pearson(k_t, |adv|)| < 0.3      => k_t is (weakly) orthogonal to |advantage| ✓
  Rescue candidates: {len(rescue)}/{len(df_out)} ({100*len(rescue)/len(df_out):.1f}%)
  → samples where RL sees no signal (low reward) but teacher can still teach (high k_t).
"""
    (out / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
