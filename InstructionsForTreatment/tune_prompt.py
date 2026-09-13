"""Optimize the English LLM system prompt against the template (ground truth).

The template renderer (`render_instructions_en`) is treated as the *true* result.
For each eval case this script:

  1. renders the template  (ground truth, no LLM)
  2. generates the LLM version with the same parameters and the current prompt
  3. scores the LLM text: how many of the template's facts are missing, and how
     many values it contradicts (deterministic checks)
  4. asks the judge model to rewrite the system prompt to fix those errors

Steps 1-4 repeat for N iterations (default 5); the lowest-scoring prompt is
written to ``prompts/system_en.txt`` (loaded automatically by ``llm_renderer``),
and a full run log is written to ``prompts/tuning_log.md``.

Usage:
    python -m InstructionsForTreatment.tune_prompt
    python -m InstructionsForTreatment.tune_prompt --iterations 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .data_loader import load_flower_data
from .llm_renderer import (
    AUDIENCE_GUIDANCE_EN,
    PROMPTS_DIR,
    _build_data_prompt,
    _DEFAULT_SYSTEM_BASE_EN,
    _parse_ethylene_sensitivity,
    system_base_en,
)
from .template_renderer import (
    STERILIZER_TOOLS,
    STS_SENSITIVITY_THRESHOLD,
    render_instructions_en,
)

OLLAMA_URL = "http://localhost:11434/api/chat"

# (flower, is_open) - chosen for varied preservation data (rates, STS, additives,
# DOC-listed blocker, no-STS). The loop tunes against these...
EVAL_CASES = [
    ("Rose", True),
    ("Anemone", False),
    ("Sunflower", False),
    ("Achillea", False),
]

# ...and the winner is only applied if it also beats the default on these
# held-out flowers (guards against a prompt that games the eval set).
VALIDATION_CASES = [
    ("Ageratum", False),
    ("Alstromeria", False),
    ("Gladiolus", False),
]

DEFAULT_GEN_MODEL = "llama3.2:latest"
DEFAULT_JUDGE_MODEL = "qwen2.5-coder:7b-instruct"


# ─── Ollama ──────────────────────────────────────────────────────────────────


def ollama_chat(model: str, system: str, user: str, num_predict: int = 900) -> str:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2, "repeat_penalty": 1.3, "num_predict": num_predict},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["message"]["content"]


# ─── Deterministic scoring ───────────────────────────────────────────────────


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _looks_complete(text: str) -> bool:
    """Same truncation check the renderer uses (see llm_renderer._looks_complete)."""
    text = (text or "").strip()
    return len(text) >= 350 and text.count("\n") >= 5


def _near(tx: str, a: str, b: str, window: int = 60) -> bool:
    """True if `b` occurs within `window` chars of some occurrence of `a`."""
    for m in re.finditer(re.escape(a), tx):
        if b in tx[max(0, m.start() - window) : m.end() + window]:
            return True
    return False


def expected_facts(flower_data: dict, is_open: bool) -> list[tuple[str, "callable"]]:
    """(label, predicate(normalized_text) -> bool) for every checkable datum."""
    p = flower_data.get("preservation", {})
    facts: list[tuple[str, callable]] = []

    for tool in STERILIZER_TOOLS:
        facts.append((f"tool '{tool}'", lambda tx, t=tool: t in tx))
    facts.append(("sterilant 'acetone'", lambda tx: "acetone" in tx))

    for tr in p.get("farmerTreatment", []):
        name = str(tr.get("productName", "")).strip()
        rate = str(tr.get("concentrationRate", "")).strip()
        if name:
            facts.append((f"product '{name}'", lambda tx, n=name.lower(): n in tx))
        if rate:
            # the rate must sit next to its own product, not just appear somewhere
            facts.append(
                (
                    f"'{name}' rate {rate}",
                    lambda tx, n=name.lower(), r=rate.lower(): _near(tx, n, r),
                )
            )

    blocker = p.get("ethyleneBlocker")
    sens = _parse_ethylene_sensitivity(p.get("ethyleneSensitivity"))
    if blocker:
        facts.append(("STS / TOG-L-101", lambda tx: "sts" in tx or "tog-l-101" in tx))
        r = str(blocker.get("concentrationRate", "")).strip()
        if r:
            facts.append(
                (
                    f"STS rate {r}",
                    lambda tx, r=r.lower(): _near(tx, "sts", r) or _near(tx, "tog-l-101", r),
                )
            )
    elif sens is not None and sens >= STS_SENSITIVITY_THRESHOLD:
        facts.append(("STS (ethylene sensitivity)", lambda tx: "sts" in tx or "tog-l-101" in tx))

    leaf = p.get("leafTreatment")
    if leaf:
        ln = str(leaf.get("productName", "")).strip()
        if ln:
            facts.append((f"leaf treatment '{ln}'", lambda tx, n=ln.lower(): n in tx))

    additives = p.get("additives")
    if additives:
        first = str(additives).lower().split()[0]
        facts.append((f"additive '{additives}'", lambda tx, a=first: a in tx))

    facts.append(
        ("temperature 20°C", lambda tx: re.search(r"20\s*°?\s*c\b|20\s*degree", tx) is not None)
    )
    facts.append(
        ("moisture 30%", lambda tx: re.search(r"30\s*%|30\s*percent", tx) is not None)
    )

    wbh = p.get("waterBucketHeight")
    if wbh is not None:
        facts.append(
            (
                f"water bucket height {wbh} cm",
                lambda tx, h=str(wbh): re.search(rf"\b{h}\s*cm|\b{h}\s*centimet", tx) is not None,
            )
        )

    # The 15-minute water-entry window is a CLOSED-flower (controlled harvest) rule only -
    # is_open=True renders that CLOSED wording (see template_renderer.py); is_open=False
    # renders OPEN wording, which has no fixed time to check for.
    if is_open:
        facts.append(("water entry 15 minutes", lambda tx: re.search(r"15\s*min", tx) is not None))
    return facts


def find_contradictions(candidate: str, flower_data: dict) -> list[str]:
    """Values the LLM stated that disagree with the data."""
    tx = _norm(candidate)
    out: set[str] = set()

    for m in re.finditer(r"(\d{1,3})\s*°?\s*c\b", tx):
        v = int(m.group(1))
        if v != 20:
            out.add(f"temperature stated as {v}°C (data: 20°C)")

    for m in re.finditer(r"(\d{1,3})\s*%", tx):
        ctx = tx[max(0, m.start() - 35) : m.start()]
        if ("humid" in ctx or "moist" in ctx) and int(m.group(1)) != 30:
            out.add(f"humidity stated as {m.group(1)}% (data: 30%)")

    wbh = flower_data.get("preservation", {}).get("waterBucketHeight")
    if wbh is not None:
        for m in re.finditer(r"(\d{1,3})\s*cm", tx):
            if int(m.group(1)) != int(wbh):
                out.add(f"bucket height stated as {m.group(1)} cm (data: {wbh} cm)")

    return sorted(out)


# ─── Judge: propose extra RULE lines (never content) ─────────────────────────

_JUDGE_SYSTEM = """\
You improve an LLM system prompt by proposing ADDITIONAL RULE LINES. You are given the \
factual errors a model made. Propose 1-4 short, general rules that would prevent those \
errors.

STRICT output format:
- One rule per line, each starting with "- ".
- Rules are GENERAL instructions only. NEVER include a specific product name, number, \
concentration, temperature, percentage, duration, or an example instruction.
- Each rule under 140 characters.
- Output ONLY the rule lines. Nothing else.

Good: "- Include every product listed in the data, each paired with its exact concentration value; do not omit any."
Good: "- Reproduce every numeric value exactly as given; never round, rescale, convert, or invent numbers."
Good: "- Do not add steps, materials, or advice that are not present in the data."
Bad:  "- Use TOG-30 at 0.035."
Bad:  "- Sterilize with acetone for 20 minutes."
"""

# A proposed rule is rejected if it smuggles in specifics instead of staying general.
_RULE_BLOCKLIST = re.compile(
    r"\d|\btog\b|tog-|acetone|distilled|rinse|halfway|two weeks|celsius|°", re.IGNORECASE
)


def propose_rules(judge_model: str, missing: list[str], contradictions: list[str]) -> list[str]:
    report = "Missing facts (model omitted these):\n" + (
        "\n".join(f"- {m}" for m in missing) or "- (none)"
    )
    report += "\n\nContradicted values (model said something other than the data):\n" + (
        "\n".join(f"- {c}" for c in contradictions) or "- (none)"
    )
    try:
        out = ollama_chat(judge_model, _JUDGE_SYSTEM, report, num_predict=300)
    except Exception as e:  # noqa: BLE001
        print(f"  judge call failed ({e}); no new rules", file=sys.stderr)
        return []

    rules: list[str] = []
    for line in out.splitlines():
        line = line.strip().lstrip("-*•").strip().rstrip(".")
        if not (15 <= len(line) <= 160):
            continue
        if _RULE_BLOCKLIST.search(line):
            continue
        rules.append(f"- {line}.")
    return rules[:4]


def build_prompt(extra_rules: list[str]) -> str:
    """Default base prompt with the accepted extra rule lines appended in Rules."""
    base = _DEFAULT_SYSTEM_BASE_EN.rstrip()
    if not extra_rules:
        return base + "\n"
    return base + "\n" + "\n".join(extra_rules) + "\n"


def score_cases(cases, gen_model: str, extra_rules: list[str]):
    """Run the generator on `cases`.

    Returns (total_score, missing, contra, details) where each `details` row is
    (flower, is_open, missing, contra, llm_text).
    """
    sys_prompt = build_prompt(extra_rules) + "\n" + AUDIENCE_GUIDANCE_EN["farmer"]
    total = 0
    all_missing: set[str] = set()
    all_contra: set[str] = set()
    details: list[tuple[str, bool, list[str], list[str], str]] = []
    for flower, is_open, data in cases:
        user_msg = _build_data_prompt(data, is_open, False, False, "en")
        candidate = ""
        for _ in range(2):  # same stub-and-retry as render_instructions_llm
            try:
                candidate = ollama_chat(gen_model, sys_prompt, user_msg)
            except Exception as e:  # noqa: BLE001
                print(f"  {flower}: generation failed ({e})", file=sys.stderr)
                candidate = ""
            if _looks_complete(candidate):
                break
        tx = _norm(candidate)
        missing = [label for label, pred in expected_facts(data, is_open) if not pred(tx)]
        contra = find_contradictions(candidate, data)
        total += len(missing) + len(contra)
        all_missing.update(missing)
        all_contra.update(contra)
        details.append((flower, is_open, missing, contra, candidate))
    return total, sorted(all_missing), sorted(all_contra), details


def compare(gen_model: str) -> int:
    """One pass, no tuning: write template vs LLM text for every case to a file."""
    try:
        ollama_chat(gen_model, "ping", "ping", num_predict=1)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_URL} ({e}). Run `ollama serve`.", file=sys.stderr)
        return 1

    all_cases = [(f, o, load_flower_data(f)) for f, o in EVAL_CASES + VALIDATION_CASES]
    applied = "prompts/system_en.txt (tuned)" if system_base_en() != _DEFAULT_SYSTEM_BASE_EN else "built-in default"

    total, _, _, details = score_cases(all_cases, gen_model, [])

    out: list[str] = [
        f"# Template vs LLM - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- gen model: `{gen_model}`  |  prompt: {applied}  |  audience: farmer  |  language: en",
        f"- total score (missing + contradictions across {len(details)} cases): **{total}**",
        "",
        "For each flower: the template is the TRUE result; the LLM version is scored against it.",
        "",
    ]
    for (flower, is_open, missing, contra, llm_text), (_, _, data) in zip(details, all_cases):
        truth = render_instructions_en(data, is_open, False, False)
        out += [
            "---",
            "",
            f"## {flower} ({'open' if is_open else 'closed'}) - {len(missing)} missing, {len(contra)} contradictions",
            "",
            f"**Missing from LLM:** {missing or 'none'}",
            "",
            f"**Contradicted by LLM:** {contra or 'none'}",
            "",
            "### Template (true result)",
            "",
            "```text",
            truth.strip(),
            "```",
            "",
            "### LLM",
            "",
            "```text",
            (llm_text.strip() or "(generation failed)"),
            "```",
            "",
        ]

    prompts_dir = Path(PROMPTS_DIR)
    prompts_dir.mkdir(exist_ok=True)
    dest = prompts_dir / "comparison.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {dest}  (total score {total})")
    return 0


# ─── Main loop ───────────────────────────────────────────────────────────────


def _fmt_cases(cases) -> str:
    return ", ".join(f"{f} ({'open' if o else 'closed'})" for f, o, _ in cases)


def run(iterations: int, gen_model: str, judge_model: str, dry_run: bool) -> int:
    try:
        ollama_chat(gen_model, "ping", "ping", num_predict=1)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_URL} ({e}). Run `ollama serve`.", file=sys.stderr)
        return 1

    eval_cases = [(f, o, load_flower_data(f)) for f, o in EVAL_CASES]
    val_cases = [(f, o, load_flower_data(f)) for f, o in VALIDATION_CASES]

    extra_rules: list[str] = []
    best_score = float("inf")
    best_rules: list[str] = []
    log: list[str] = [
        f"# Prompt tuning log - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- gen model: `{gen_model}`",
        f"- judge model: `{judge_model}`",
        f"- eval cases: {_fmt_cases(eval_cases)}",
        f"- validation cases: {_fmt_cases(val_cases)}",
        "- ground truth: `render_instructions_en` (template)",
        "- the judge may only append general RULE lines - never data or examples",
        "",
    ]

    for i in range(1, iterations + 1):
        print(f"=== iteration {i}/{iterations} ===")
        score, missing, contra, details = score_cases(eval_cases, gen_model, extra_rules)
        for flower, is_open, m, c, _ in details:
            print(f"  {flower} ({'open' if is_open else 'closed'}): {len(m)} missing, {len(c)} contradictions")
        print(f"  -> score {score}")

        log += [
            f"## Iteration {i} - score {score}",
            "",
            "Extra rules in effect:",
            *( [f"  {r}" for r in extra_rules] or ["  (none - default prompt)"]),
            "",
            *[
                f"- **{f} ({'open' if o else 'closed'})** - missing: {m or 'none'}; "
                f"contradictions: {c or 'none'}"
                for f, o, m, c, _ in details
            ],
            "",
        ]

        if score < best_score:
            best_score, best_rules = score, list(extra_rules)

        if score == 0 or i == iterations:
            break

        new_rules = propose_rules(judge_model, missing, contra)
        added = [r for r in new_rules if r not in extra_rules]
        if not added:
            print("  judge proposed no usable new rules; stopping early")
            log += ["_Judge proposed no usable new rules - stopped early._", ""]
            break
        print(f"  + {len(added)} rule(s): " + " | ".join(added))
        extra_rules = (extra_rules + added)[:8]

    # Validation: apply only if the best rule set beats the default on held-out flowers.
    default_val, *_ = score_cases(val_cases, gen_model, [])
    tuned_val, v_missing, v_contra, _ = score_cases(val_cases, gen_model, best_rules)
    final_prompt = build_prompt(best_rules)

    log += [
        "## Result",
        "",
        f"- best eval score: **{best_score}** (default start was iteration 1)",
        f"- validation score - default: **{default_val}**, tuned: **{tuned_val}**",
        "",
        "Best rule set:",
        *([f"  {r}" for r in best_rules] or ["  (none)"]),
        "",
        "Final prompt:",
        "",
        "```text",
        final_prompt,
        "```",
        "",
    ]

    prompts_dir = Path(PROMPTS_DIR)
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / "tuning_log.md").write_text("\n".join(log), encoding="utf-8")
    print(f"\nLog: {prompts_dir / 'tuning_log.md'}")
    print(f"Validation - default: {default_val}, tuned: {tuned_val}")

    target = prompts_dir / "system_en.txt"
    if not best_rules:
        print("No rules improved on the default - nothing to apply.")
    elif dry_run:
        (prompts_dir / "system_en.candidate.txt").write_text(final_prompt, encoding="utf-8")
        print(f"Dry run - candidate written to {prompts_dir / 'system_en.candidate.txt'} (not applied)")
    elif tuned_val < default_val:
        target.write_text(final_prompt, encoding="utf-8")
        print(f"Applied tuned prompt (eval {best_score}, validation {tuned_val} < {default_val}) -> {target}")
        print("Revert with: del InstructionsForTreatment\\prompts\\system_en.txt")
    else:
        print(
            f"Tuned prompt did not beat the default on validation "
            f"({tuned_val} >= {default_val}); NOT applied."
        )
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tune_prompt", description=__doc__)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="model whose prompt is tuned")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="model that rewrites the prompt")
    ap.add_argument("--dry-run", action="store_true", help="do not write prompts/system_en.txt")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="skip tuning: just write template vs LLM text for every case to "
        "prompts/comparison.md so you can read the output",
    )
    args = ap.parse_args(argv)
    if args.compare:
        sys.exit(compare(args.gen_model))
    sys.exit(run(args.iterations, args.gen_model, args.judge_model, args.dry_run))


if __name__ == "__main__":
    main()
