"""Offline tests for mini instance catalog (no model API)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stage2.devbugs.catalog import MINI_INSTANCES, list_instance_ids
from stage2.devbugs.evaluate import evaluate_instance
from stage2.devbugs.sandbox import materialize_repo


def test_catalog_has_instances():
    assert len(list_instance_ids()) >= 10


def test_buggy_repos_fail_pytest():
    for iid in list_instance_ids():
        instance = MINI_INSTANCES[iid]
        with tempfile.TemporaryDirectory() as tmp:
            repo = materialize_repo(instance, Path(tmp))
            assert not evaluate_instance(instance, repo), iid


def test_fixed_repo_passes():
    """Sanity: applying the obvious fix makes tests pass."""
    fixes = {
        # easy: store handlers in a list and fan out
        "mini_eventbus_001": (
            "eventbus/bus.py",
            "class EventBus:\n"
            '    """Tiny synchronous publish/subscribe bus."""\n\n'
            "    def __init__(self):\n"
            "        self._handlers = {}\n\n"
            "    def subscribe(self, event, handler):\n"
            "        self._handlers.setdefault(event, []).append(handler)\n\n"
            "    def emit(self, event, **payload):\n"
            "        for handler in self._handlers.get(event, []):\n"
            "            handler(payload)\n",
        ),
        # medium: round dollars to nearest cent
        "mini_money_006": (
            "money/amount.py",
            "class Money:\n"
            '    """Integer-cent money value."""\n\n'
            "    def __init__(self, cents):\n"
            "        self.cents = cents\n\n"
            "    @classmethod\n"
            "    def from_dollars(cls, dollars):\n"
            "        return cls(round(dollars * 100))\n\n"
            "    def __add__(self, other):\n"
            "        return Money(self.cents + other.cents)\n\n"
            "    def __eq__(self, other):\n"
            "        return isinstance(other, Money) and self.cents == other.cents\n\n"
            "    def dollars(self):\n"
            "        return self.cents / 100\n",
        ),
        # hard: merge nested dicts recursively without mutating inputs
        "mini_confmerge_008": (
            "confmerge/deepmerge.py",
            "def deep_merge(base, override):\n"
            '    """Merge `override` onto `base`, recursing into nested dicts."""\n'
            "    result = dict(base)\n"
            "    for key, value in override.items():\n"
            "        if (\n"
            "            key in result\n"
            "            and isinstance(result[key], dict)\n"
            "            and isinstance(value, dict)\n"
            "        ):\n"
            "            result[key] = deep_merge(result[key], value)\n"
            "        else:\n"
            "            result[key] = value\n"
            "    return result\n",
        ),
    }
    for iid, (rel, content) in fixes.items():
        instance = MINI_INSTANCES[iid]
        with tempfile.TemporaryDirectory() as tmp:
            repo = materialize_repo(instance, Path(tmp))
            (repo / rel).write_text(content, encoding="utf-8")
            assert evaluate_instance(instance, repo), iid
