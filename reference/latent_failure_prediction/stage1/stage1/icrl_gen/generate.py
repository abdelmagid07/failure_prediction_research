#!/usr/bin/env python
"""
Generate ICRL conversations for value-axis construction (paper Appendix A).

Backends:
  anthropic   — Claude/Opus via API (faithful Stage 1)
  local_qwen  — Qwen3-8B on local GPU

Examples:
  python -m stage1.icrl_gen.generate --n 300 --output data/icrl.json --resume

  # Dev preset (Qwen-local ICRL, separate artifacts)
  python -m stage1.icrl_gen.generate --n 100 --backend local_qwen \
    --output data/icrl_proxy.json --resume --syntactic-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from stage1.common.config import criteria_by_id, load_criteria
from stage1.common.paths import data_file
from stage1.icrl.boundaries import extract_paragraph
from stage1.icrl.schema import Conversation, Turn, load_conversations, save_conversations
from stage1.icrl_gen.backends.factory import get_backend
from stage1.icrl_gen.llm import complete
from stage1.icrl_gen.prompts import GAME_SYSTEM_PROMPT, META_PROMPT_TEMPLATE, PHASE_INSTRUCTIONS
from stage1.icrl_gen.verify import check_and_locate, check_syntactic_criterion, judge_semantic
from stage1.icrl_gen.wikipedia import fetch_paragraphs

WRONG_HYP_CACHE = data_file("wrong_hypotheses.json")
DEFAULT_MAX_TURN_RETRIES = 5


def assign_criteria(n: int, rng: random.Random, criteria_ids: list[str] | None = None) -> list[str]:
    """Distribute n conversations across the given criteria (default: all 50)."""
    ids = criteria_ids if criteria_ids is not None else [c["id"] for c in load_criteria()]
    base, extra = divmod(n, len(ids))
    assignment: list[str] = []
    order = ids.copy()
    rng.shuffle(order)
    for i, cid in enumerate(order):
        assignment.extend([cid] * (base + (1 if i < extra else 0)))
    rng.shuffle(assignment)
    return assignment


def load_wrong_hypotheses() -> dict[str, str]:
    if WRONG_HYP_CACHE.exists():
        with open(WRONG_HYP_CACHE) as f:
            return json.load(f)
    return {}


def save_wrong_hypotheses(cache: dict[str, str]) -> None:
    WRONG_HYP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(WRONG_HYP_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def get_wrong_hypothesis(backend, criterion_id: str, criterion_text: str, cache: dict[str, str]) -> str:
    if criterion_id in cache:
        return cache[criterion_id]
    system = "You invent plausible WRONG hypotheses for a hidden paragraph-editing game."
    user = f"""The TRUE hidden criterion is: the modified text must {criterion_text}

Invent ONE plausible but INCORRECT hypothesis a model might guess (one short sentence).
Do NOT describe the true criterion. Examples: "use past tense", "capitalize key nouns", "make sentences shorter".

Reply with only the wrong hypothesis sentence."""
    hyp = complete(backend, system, user, max_tokens=128, temperature=0.9)
    cache[criterion_id] = hyp.strip()
    save_wrong_hypotheses(cache)
    return cache[criterion_id]


def build_meta_prompt(
    criterion_text: str,
    phase: str,
    seed_paragraph: str,
    wrong_hypothesis: str | None = None,
) -> str:
    instr = PHASE_INSTRUCTIONS[phase]
    extra = ""
    if phase == "before_discovery" and wrong_hypothesis:
        extra = f"\nYour wrong hypothesis for this turn: {wrong_hypothesis}\n"
    return META_PROMPT_TEMPLATE.format(
        reward_description=criterion_text,
        thinking_instruction=instr["thinking"],
        text_instruction=instr["text"],
    ) + extra + f"\n\nSeed paragraph to rewrite:\n{seed_paragraph}"


def generate_assistant_turn(
    backend,
    criterion_text: str,
    phase: str,
    seed_paragraph: str,
    wrong_hypothesis: str | None,
) -> str:
    system = build_meta_prompt(criterion_text, phase, seed_paragraph, wrong_hypothesis)
    user = "Write the assistant's single turn now (thinking tags + modified paragraph only)."
    text = complete(backend, system, user, max_tokens=768, temperature=0.8)
    if "<thinking>" not in text.lower():
        text = f"<thinking>I'll try a modification based on my current hypothesis.</thinking>{text}"
    return text.strip()


def parse_assistant_or_retry(raw: str) -> str:
    raw = raw.strip()
    if not extract_paragraph(raw):
        raise ValueError("Empty paragraph after thinking tags")
    return raw


def paragraph_satisfies(
    backend,
    criterion_id: str,
    criterion_text: str,
    criterion_type: str,
    assistant_content: str,
) -> bool:
    paragraph = extract_paragraph(assistant_content)
    if not paragraph:
        return False
    if criterion_type == "syntactic":
        return check_syntactic_criterion(paragraph, criterion_id)
    satisfies, _ = judge_semantic(backend, criterion_text, paragraph)
    return satisfies


def generate_conversation(
    backend,
    conv_id: str,
    criterion_id: str,
    criterion_text: str,
    criterion_type: str,
    paragraphs: list[str],
    discovery_idx: int,
    wrong_hyp_cache: dict[str, str],
    rng: random.Random,
    *,
    max_turn_retries: int = DEFAULT_MAX_TURN_RETRIES,
    verbose: bool = True,
) -> Conversation | None:
    n_para = len(paragraphs)
    system_prompt = GAME_SYSTEM_PROMPT.replace("$N", str(n_para))
    turns: list[Turn] = [Turn(role="system", content=system_prompt)]
    wrong_hyp = get_wrong_hypothesis(backend, criterion_id, criterion_text, wrong_hyp_cache)

    first_post_idx: int | None = None
    satisfying_char: int | None = None

    for p_idx, seed in enumerate(paragraphs):
        if p_idx > discovery_idx + 1:
            break

        turns.append(Turn(role="user", content=seed))

        if p_idx < discovery_idx:
            phase = "before_discovery"
            success = False
            for attempt in range(max_turn_retries):
                raw = generate_assistant_turn(backend, criterion_text, phase, seed, wrong_hyp)
                try:
                    content = parse_assistant_or_retry(raw)
                except ValueError:
                    continue
                if not paragraph_satisfies(backend, criterion_id, criterion_text, criterion_type, content):
                    turns.append(Turn(role="assistant", content=content))
                    turns.append(Turn(role="user", content="-1"))
                    success = True
                    break
                if verbose:
                    print(f"    retry before_discovery {conv_id} p{p_idx} attempt {attempt}", flush=True)
            if not success:
                if verbose:
                    print(f"  SKIP {conv_id}: failed before_discovery p{p_idx}", flush=True)
                return None

        elif p_idx == discovery_idx:
            phase = "discovery"
            feedback = "+1"
            success = False
            for attempt in range(max_turn_retries):
                raw = generate_assistant_turn(backend, criterion_text, phase, seed, wrong_hyp)
                try:
                    content = parse_assistant_or_retry(raw)
                except ValueError:
                    continue
                ok, _ = check_and_locate(backend, criterion_id, criterion_text, criterion_type, content)
                if ok:
                    turns.append(Turn(role="assistant", content=content))
                    turns.append(Turn(role="user", content=feedback))
                    success = True
                    break
                if verbose:
                    print(f"    retry discovery {conv_id} attempt {attempt}", flush=True)
            if not success:
                if verbose:
                    print(f"  SKIP {conv_id}: failed discovery p{p_idx}", flush=True)
                return None

        else:
            phase = "after_discovery"
            feedback = "+1"
            first_post_idx = len(turns)
            success = False
            for attempt in range(max_turn_retries):
                raw = generate_assistant_turn(backend, criterion_text, phase, seed, wrong_hyp)
                try:
                    content = parse_assistant_or_retry(raw)
                except ValueError:
                    continue
                ok, char_start = check_and_locate(
                    backend, criterion_id, criterion_text, criterion_type, content
                )
                if ok and char_start is not None and char_start > 0:
                    turns.append(Turn(role="assistant", content=content))
                    turns.append(Turn(role="user", content=feedback))
                    satisfying_char = char_start
                    success = True
                    break
                if verbose:
                    print(f"    retry post_discovery {conv_id} attempt {attempt}", flush=True)
            if not success:
                if verbose:
                    print(f"  SKIP {conv_id}: failed post_discovery p{p_idx}", flush=True)
                return None
            break

    if first_post_idx is None or satisfying_char is None:
        return None

    return Conversation(
        conv_id=conv_id,
        criterion_id=criterion_id,
        discovery_paragraph_idx=discovery_idx,
        turns=turns,
        first_post_discovery_turn_idx=first_post_idx,
        satisfying_char_start=satisfying_char,
    )


def generate_batch(
    n: int,
    output_path: Path,
    *,
    backend_name: str = "anthropic",
    seed: int = 42,
    resume: bool = False,
    min_paragraphs: int = 4,
    max_paragraphs: int = 8,
    max_turn_retries: int = DEFAULT_MAX_TURN_RETRIES,
    verbose: bool = True,
    syntactic_only: bool = False,
) -> list[Conversation]:
    rng = random.Random(seed)
    backend = get_backend(backend_name)
    if verbose:
        print(f"Using ICRL backend: {backend.name}", flush=True)

    by_id = criteria_by_id()
    wrong_cache = load_wrong_hypotheses()

    criteria_ids: list[str] | None = None
    if syntactic_only:
        criteria_ids = [c["id"] for c in load_criteria() if c["type"] == "syntactic"]
        if verbose:
            print(f"Syntactic-only mode: {len(criteria_ids)} criteria, exact programmatic "
                  f"labels (no LLM judge): {criteria_ids}", flush=True)

    existing: list[Conversation] = []
    done_ids: set[str] = set()
    if resume and output_path.exists():
        existing = load_conversations(output_path)
        done_ids = {c.conv_id for c in existing}
        if verbose:
            print(f"Resuming: {len(existing)} conversations already in {output_path}", flush=True)

    assignment = assign_criteria(n, rng, criteria_ids)
    results = list(existing)

    for i, criterion_id in enumerate(assignment):
        conv_id = f"icrl_{criterion_id}_{i:04d}"
        if conv_id in done_ids:
            continue

        crit = by_id[criterion_id]
        n_para = rng.randint(min_paragraphs, max_paragraphs)
        discovery_idx = rng.randint(1, min(5, n_para - 2))

        if verbose:
            print(f"[{len(results)+1}/{n}] {conv_id} criterion={criterion_id} "
                  f"paragraphs={n_para} discovery={discovery_idx}", flush=True)

        try:
            paragraphs = fetch_paragraphs(
                n_para,
                rng,
                on_retry=lambda m: verbose and print(f"  {m}", flush=True),
            )
        except RuntimeError as exc:
            print(f"  SKIP {conv_id}: {exc}", flush=True)
            continue

        try:
            conv = generate_conversation(
                backend,
                conv_id,
                criterion_id,
                crit["text"],
                crit["type"],
                paragraphs,
                discovery_idx,
                wrong_cache,
                rng,
                max_turn_retries=max_turn_retries,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 — one bad conversation must not kill the batch
            print(f"  SKIP {conv_id}: generation error ({type(exc).__name__}: {exc})", flush=True)
            continue
        if conv is None:
            continue

        results.append(conv)
        save_conversations(results, output_path)
        if verbose:
            para = extract_paragraph(get_first_post_turn(conv).content if get_first_post_turn(conv) else "")
            print(f"  OK {conv_id} post_para={para[:60]}... char_start={conv.satisfying_char_start}", flush=True)

        if backend.name != "local_qwen":
            time.sleep(0.3)

    return results


def get_first_post_turn(conv: Conversation) -> Turn | None:
    if conv.first_post_discovery_turn_idx is None:
        return None
    t = conv.turns[conv.first_post_discovery_turn_idx]
    return t if t.role == "assistant" else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="Number of conversations to generate")
    ap.add_argument("--output", type=Path, default=data_file("icrl.json"))
    ap.add_argument("--backend", default=None, help="anthropic or local_qwen (default: ICRL_BACKEND env)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Append to existing output, skip completed IDs")
    ap.add_argument("--min-paragraphs", type=int, default=4)
    ap.add_argument("--max-paragraphs", type=int, default=8)
    ap.add_argument(
        "--max-turn-retries",
        type=int,
        default=None,
        help="Retries per turn (default 5 anthropic, 8 local_qwen)",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--syntactic-only",
        action="store_true",
        help="Only generate the 15 syntactic criteria (c000-c014). Labels are exact "
        "(programmatic), bypassing the LLM judge.",
    )
    args = ap.parse_args()

    backend_name = args.backend or os.environ.get("ICRL_BACKEND", "anthropic")
    max_retries = args.max_turn_retries
    if max_retries is None:
        max_retries = 8 if backend_name in ("local_qwen", "qwen", "local") else DEFAULT_MAX_TURN_RETRIES

    args.output.parent.mkdir(parents=True, exist_ok=True)
    convs = generate_batch(
        args.n,
        args.output,
        backend_name=backend_name,
        seed=args.seed,
        resume=args.resume,
        min_paragraphs=args.min_paragraphs,
        max_paragraphs=args.max_paragraphs,
        max_turn_retries=max_retries,
        verbose=not args.quiet,
        syntactic_only=args.syntactic_only,
    )
    print(f"\nDone. {len(convs)} conversations -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
