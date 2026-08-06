"""Load and sample DebugBench (Tian et al., 2024) problems.

Public dataset on Hugging Face: ``Rtian/DebugBench``. Fields used: ``slug``,
``category``, ``subtype``, ``language``, ``question``, ``solution``,
``buggy_code`` (the reference script's ``code_quality/problems.json`` used
these same field names).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Problem:
    slug: str
    category: str
    subtype: str
    question: str
    solution: str
    buggy_code: str


def load_debugbench(
    *,
    dataset_name: str = "Rtian/DebugBench",
    language: str = "Python",
    n_problems: int | str = 150,
    seed: int = 42,
) -> list[Problem]:
    """Load, filter to one language, and (optionally) sample DebugBench problems.

    ``n_problems="all"`` keeps every filtered instance; an int takes a seeded
    random sample (without replacement) of that size, or every instance if
    fewer are available.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")

    problems: list[Problem] = []
    seen_slugs: set[str] = set()
    for row in ds:
        if row.get("language") != language:
            continue
        slug = row.get("slug")
        solution = row.get("solution")
        buggy = row.get("buggy_code")
        if not slug or not solution or not buggy:
            continue
        # DebugBench has multiple bug categories per LeetCode problem; keep
        # the first one seen per slug so each problem contributes one pair.
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        problems.append(
            Problem(
                slug=str(slug),
                category=str(row.get("category", "")),
                subtype=str(row.get("subtype", "")),
                question=str(row.get("question", "")),
                solution=str(solution),
                buggy_code=str(buggy),
            )
        )

    problems.sort(key=lambda p: p.slug)  # deterministic order before sampling

    if n_problems == "all":
        return problems

    n = int(n_problems)
    if n >= len(problems):
        return problems

    rng = random.Random(seed)
    return rng.sample(problems, n)
