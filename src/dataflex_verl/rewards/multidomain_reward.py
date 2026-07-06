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


# ----------------------------- science: SciQ (multiple choice) ----------------------
# We format SciQ as a 4-option MCQ (A-D). Ground truth is the correct option letter.
# Student answer: extract the chosen letter (last "A/B/C/D" near an answer marker).
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def compute_sciq(solution_str: str, ground_truth: str) -> float:
    gt = (ground_truth or "").strip().upper()[:1]
    if gt not in "ABCD":
        return 0.0
    ans = solution_str or ""
    for marker in ("final answer", "answer:", "the answer is", "correct option"):
        idx = ans.lower().rfind(marker)
        if idx != -1:
            ans = ans[idx:]
            break
    letters = _LETTER_RE.findall(ans)
    if not letters:
        letters = _LETTER_RE.findall(solution_str or "")
    if not letters:
        return 0.0
    return 1.0 if letters[-1] == gt else 0.0


# ----------------------------- dispatcher -----------------------------
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ds = str(data_source)
    if ds in ("kk_logic", "knights-and-knaves", "logic"):
        return compute_kk(solution_str, ground_truth)
    if ds in ("sciq", "science"):
        return compute_sciq(solution_str, ground_truth)
    # math (and anything else) -> verl's built-in dispatcher
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(data_source, solution_str, ground_truth, extra_info)
