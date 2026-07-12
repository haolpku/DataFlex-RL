"""Multi-domain reward function for the DataFlex-RL 3-domain RLVR set (math / logic / science).

verl selects the reward by `data_source` (config `reward_fn_key: data_source`). Built-in
verl only knows math/code/etc., so for the logic (Knights-and-Knaves) and science (SciQ)
domains we dispatch here. Math falls through to verl's own `math_dapo` verifier.

Wire it in (zero fork) via:
    custom_reward_function.path=<this file> custom_reward_function.name=compute_score

Signature matches verl's reward manager call:
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
returns a float in [0, 1] (or a dict with a "score" key).
"""
from __future__ import annotations

import re


# ----------------------------- logic: Knights & Knaves -----------------------------
# Ground truth is the canonical solution_text, e.g.
#   "Michael is a knight, Zoey is a knave, and Ethan is a knight."
# We parse each person's role from the student's final answer and compare to the truth
# assignment (name -> knight/knave), which we recover from the ground_truth string.
_ROLE_RE = re.compile(r"\b([A-Z][a-zA-Z]+)\b\s+is\s+a\s+(knight|knave)", re.IGNORECASE)


def _parse_kk_assignment(text: str) -> dict[str, str]:
    """name(lower) -> 'knight'|'knave' from the LAST mention of each name."""
    out: dict[str, str] = {}
    for m in _ROLE_RE.finditer(text or ""):
        out[m.group(1).lower()] = m.group(2).lower()
    return out


def compute_kk(solution_str: str, ground_truth: str) -> float:
    truth = _parse_kk_assignment(ground_truth)
    if not truth:
        return 0.0
    # student's answer: prefer text after a final-answer marker if present
    ans = solution_str or ""
    for marker in ("final answer", "answer:", "conclusion", "therefore"):
        idx = ans.lower().rfind(marker)
        if idx != -1:
            ans = ans[idx:]
            break
    pred = _parse_kk_assignment(ans)
    if not pred:
        pred = _parse_kk_assignment(solution_str)  # fall back to whole text
    if not pred:
        return 0.0
    # all named people must be assigned correctly (exact-match on the full assignment)
    for name, role in truth.items():
        if pred.get(name) != role:
            return 0.0
    return 1.0


# ----------------------------- science: SciQ / GPQA / MMLU-Pro (MCQ letter) -----------------
# We format SciQ/GPQA as 4-option MCQ (A-D), MMLU-Pro as 10-option (A-J).
# Ground truth is the correct option letter.
# Student answer: extract the chosen letter near an answer marker.
_LETTER_RE = re.compile(r"\b([A-J])\b")


def compute_mcq_letter(solution_str: str, ground_truth: str, letters: str = "ABCD") -> float:
    gt = (ground_truth or "").strip().upper()[:1]
    if gt not in letters:
        return 0.0
    ans = solution_str or ""
    for marker in ("final answer", "answer is:", "answer:", "the answer is", "correct option"):
        idx = ans.lower().rfind(marker)
        if idx != -1:
            ans = ans[idx:]
            break
    letters_found = re.findall(rf"\b([{letters}])\b", ans)
    if not letters_found:
        letters_found = re.findall(rf"\b([{letters}])\b", solution_str or "")
    if not letters_found:
        return 0.0
    return 1.0 if letters_found[-1] == gt else 0.0


def compute_sciq(solution_str: str, ground_truth: str) -> float:
    return compute_mcq_letter(solution_str, ground_truth, "ABCD")


def compute_mmlu_pro(solution_str: str, ground_truth: str) -> float:
    return compute_mcq_letter(solution_str, ground_truth, "ABCDEFGHIJ")


# ----------------------------- dispatcher -----------------------------
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return a UNIFORM dict {score, acc, pred} for EVERY domain.

    verl's process_validation_metrics averages each extra-info field per data_source;
    if math returns {score,acc,pred} but logic/science return a bare float, the missing
    acc/pred become None and np.mean() crashes. So we normalize all domains to the same
    dict shape (see bug-001).
    """
    ds = str(data_source)
    if ds in ("kk_logic", "knights-and-knaves", "logic", "kk_logic_hard"):
        s = compute_kk(solution_str, ground_truth)
        return {"score": s, "acc": bool(s >= 1.0), "pred": ""}
    if ds in ("sciq", "science", "gpqa", "gpqa_diamond", "gpqa_main"):
        # GPQA uses the same A/B/C/D letter format as SciQ, so compute_sciq reuses cleanly.
        s = compute_sciq(solution_str, ground_truth)
        return {"score": s, "acc": bool(s >= 1.0), "pred": ""}
    if ds.startswith("mmlu_pro"):
        # MMLU-Pro has 10 options A-J
        s = compute_mmlu_pro(solution_str, ground_truth)
        return {"score": s, "acc": bool(s >= 1.0), "pred": ""}
    if ds in ("bbh_logical_deduction", "bbh_tracking", "zebra_logic_mc"):
        # BBH MCQ subtasks + ZebraLogic MC use letter answers (up to 7-choice for BBH,
        # up to 6-choice for zebra) — treat as A-J range.
        s = compute_mmlu_pro(solution_str, ground_truth)
        return {"score": s, "acc": bool(s >= 1.0), "pred": ""}
    # math (and anything else) -> verl's built-in dispatcher (math_dapo returns a dict)
    from verl.utils.reward_score import default_compute_score

    res = default_compute_score(data_source, solution_str, ground_truth, extra_info)
    if isinstance(res, dict):
        # ensure the same keys exist as the logic/science branch
        res.setdefault("score", float(res.get("acc", 0.0)))
        res.setdefault("acc", bool(res.get("score", 0.0) > 0))
        res.setdefault("pred", "")
        return res
    return {"score": float(res), "acc": bool(res > 0), "pred": ""}

