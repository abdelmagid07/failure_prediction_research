"""Mini bug-fix instances for the local agent environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MiniInstance:
    instance_id: str
    difficulty: Literal["easy", "medium", "hard"]
    problem_statement: str
    files: dict[str, str]
    test_cmd: str = "python -m pytest -q"


def _conftest() -> str:
    """conftest.py placed at tests/ so the repo root is importable as a package."""
    return """import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""


MINI_INSTANCES: dict[str, MiniInstance] = {
    # ---------------------------------------------------------------- easy ----
    "mini_eventbus_001": MiniInstance(
        instance_id="mini_eventbus_001",
        difficulty="easy",
        problem_statement=(
            "Subscribing two handlers to the same event and emitting it raises a "
            "TypeError instead of invoking both handlers.\n\n"
            "Reproduction:\n"
            "    bus = EventBus()\n"
            "    bus.subscribe('tick', on_tick_a)\n"
            "    bus.subscribe('tick', on_tick_b)\n"
            "    bus.emit('tick', n=1)\n\n"
            "Traceback (most recent call last):\n"
            "  ...\n"
            "TypeError: 'function' object is not iterable\n\n"
            "Both handlers should be called with the payload dict. Emitting an event "
            "with no subscribers should be a no-op."
        ),
        files={
            "eventbus/__init__.py": "from eventbus.bus import EventBus\n",
            "eventbus/bus.py": (
                "class EventBus:\n"
                '    """Tiny synchronous publish/subscribe bus."""\n\n'
                "    def __init__(self):\n"
                "        self._handlers = {}\n\n"
                "    def subscribe(self, event, handler):\n"
                "        self._handlers[event] = handler\n\n"
                "    def emit(self, event, **payload):\n"
                "        for handler in self._handlers.get(event, []):\n"
                "            handler(payload)\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_bus.py": (
                "from eventbus import EventBus\n\n\n"
                "def test_fanout_to_multiple_handlers():\n"
                "    bus = EventBus()\n"
                "    seen = []\n"
                "    bus.subscribe('tick', lambda p: seen.append(('a', p['n'])))\n"
                "    bus.subscribe('tick', lambda p: seen.append(('b', p['n'])))\n"
                "    bus.emit('tick', n=1)\n"
                "    assert ('a', 1) in seen\n"
                "    assert ('b', 1) in seen\n\n\n"
                "def test_emit_with_no_subscribers_is_noop():\n"
                "    EventBus().emit('nothing', n=1)\n"
            ),
        },
    ),
    "mini_pagination_002": MiniInstance(
        instance_id="mini_pagination_002",
        difficulty="easy",
        problem_statement=(
            "Pagination is off. With 10 items at 3 per page the UI shows 3 pages, but "
            "there should be 4 (the last page holds the remaining item). Also, asking "
            "for page 1 returns the *second* page of results — pages are supposed to be "
            "1-indexed.\n\n"
            "Reproduction:\n"
            "    page_count(10, 3)            # -> 3, expected 4\n"
            "    slice_page(range(10), 1, 3)  # -> [3, 4, 5], expected [0, 1, 2]\n"
        ),
        files={
            "paging/__init__.py": (
                "from paging.paginator import page_count, slice_page\n"
            ),
            "paging/paginator.py": (
                "def page_count(total, per_page):\n"
                "    return total // per_page\n\n\n"
                "def slice_page(items, page, per_page):\n"
                "    items = list(items)\n"
                "    start = page * per_page\n"
                "    return items[start:start + per_page]\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_paginator.py": (
                "from paging import page_count, slice_page\n\n\n"
                "def test_page_count_counts_partial_last_page():\n"
                "    assert page_count(10, 3) == 4\n"
                "    assert page_count(9, 3) == 3\n"
                "    assert page_count(0, 3) == 0\n\n\n"
                "def test_pages_are_one_indexed():\n"
                "    items = list(range(10))\n"
                "    assert slice_page(items, 1, 3) == [0, 1, 2]\n"
                "    assert slice_page(items, 2, 3) == [3, 4, 5]\n"
            ),
        },
    ),
    "mini_textstats_003": MiniInstance(
        instance_id="mini_textstats_003",
        difficulty="easy",
        problem_statement=(
            "Word statistics break on real text. Counting words in a string that uses "
            "multiple spaces or newlines as separators returns too many words, and "
            "`top_word` sometimes returns an empty string instead of the most frequent "
            "word.\n\n"
            "Reproduction:\n"
            "    word_count('a  b c')        # -> 4, expected 3\n"
            "    top_word('hi   hi  bye')    # -> '', expected 'hi'\n"
        ),
        files={
            "textstats/__init__.py": (
                "from textstats.words import word_count, top_word\n"
            ),
            "textstats/words.py": (
                "def word_count(text):\n"
                '    return len(text.split(" "))\n\n\n'
                "def top_word(text):\n"
                "    counts = {}\n"
                '    for word in text.split(" "):\n'
                "        counts[word] = counts.get(word, 0) + 1\n"
                "    return max(counts, key=counts.get)\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_words.py": (
                "from textstats import word_count, top_word\n\n\n"
                "def test_word_count_collapses_whitespace():\n"
                "    assert word_count('a  b c') == 3\n"
                "    assert word_count('one\\ntwo\\tthree') == 3\n\n\n"
                "def test_top_word_ignores_blank_runs():\n"
                "    assert top_word('the cat the dog the') == 'the'\n"
                "    assert top_word('hi   hi  bye') == 'hi'\n"
            ),
        },
    ),
    "mini_fsm_004": MiniInstance(
        instance_id="mini_fsm_004",
        difficulty="easy",
        problem_statement=(
            "The state machine never advances. Firing a valid event returns the current "
            "state and leaves `.state` unchanged, so the object is stuck in its initial "
            "state forever.\n\n"
            "Reproduction:\n"
            "    sm = StateMachine('idle', {('idle', 'start'): 'running'})\n"
            "    sm.fire('start')   # -> 'idle', expected 'running'\n"
            "    sm.state           # -> 'idle', expected 'running'\n\n"
            "Firing an event with no matching transition should leave the state "
            "unchanged (no error)."
        ),
        files={
            "fsm/__init__.py": "from fsm.machine import StateMachine\n",
            "fsm/machine.py": (
                "class StateMachine:\n"
                "    def __init__(self, initial, transitions):\n"
                "        self.state = initial\n"
                "        self.transitions = transitions  # {(state, event): next_state}\n\n"
                "    def fire(self, event):\n"
                "        key = (self.state, event)\n"
                "        if key in self.transitions:\n"
                "            self.transitions[key]\n"
                "        return self.state\n\n"
                "    def can_fire(self, event):\n"
                "        return (self.state, event) in self.transitions\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_machine.py": (
                "from fsm import StateMachine\n\n\n"
                "def test_valid_transition_advances_state():\n"
                "    sm = StateMachine('idle', {\n"
                "        ('idle', 'start'): 'running',\n"
                "        ('running', 'stop'): 'idle',\n"
                "    })\n"
                "    assert sm.fire('start') == 'running'\n"
                "    assert sm.state == 'running'\n"
                "    assert sm.fire('stop') == 'idle'\n\n\n"
                "def test_unknown_event_is_noop():\n"
                "    sm = StateMachine('idle', {('idle', 'start'): 'running'})\n"
                "    assert sm.fire('unknown') == 'idle'\n"
                "    assert sm.state == 'idle'\n"
            ),
        },
    ),
    # -------------------------------------------------------------- medium ----
    "mini_router_005": MiniInstance(
        instance_id="mini_router_005",
        difficulty="medium",
        problem_statement=(
            "Routes with named parameters never match. Registering '/users/<id>' and "
            "then matching '/users/42' returns no handler, even though static routes "
            "like '/health' work fine.\n\n"
            "Reproduction:\n"
            "    r = Router()\n"
            "    r.add('/users/<id>', user_handler)\n"
            "    r.match('/users/42')   # -> (None, {}), expected (user_handler, {'id': '42'})\n"
        ),
        files={
            "router/__init__.py": "from router.core import Router\n",
            "router/core.py": (
                "class Router:\n"
                '    """Minimal path router with <name> placeholders."""\n\n'
                "    def __init__(self):\n"
                "        self.routes = []\n\n"
                "    def add(self, pattern, handler):\n"
                '        self.routes.append((pattern.split("/"), handler))\n\n'
                "    def match(self, path):\n"
                '        parts = path.split("/")\n'
                "        for pattern, handler in self.routes:\n"
                "            if len(pattern) != len(parts):\n"
                "                continue\n"
                "            params = {}\n"
                "            ok = True\n"
                "            for seg_pat, seg in zip(pattern, parts):\n"
                '                if seg_pat.startswith("{") and seg_pat.endswith("}"):\n'
                "                    params[seg_pat[1:-1]] = seg\n"
                "                elif seg_pat != seg:\n"
                "                    ok = False\n"
                "                    break\n"
                "            if ok:\n"
                "                return handler, params\n"
                "        return None, {}\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_router.py": (
                "from router import Router\n\n\n"
                "def test_named_parameter_route():\n"
                "    r = Router()\n"
                "    r.add('/users/<id>', 'user')\n"
                "    handler, params = r.match('/users/42')\n"
                "    assert handler == 'user'\n"
                "    assert params == {'id': '42'}\n\n\n"
                "def test_static_route_still_matches():\n"
                "    r = Router()\n"
                "    r.add('/health', 'ok')\n"
                "    assert r.match('/health') == ('ok', {})\n\n\n"
                "def test_no_match_returns_none():\n"
                "    r = Router()\n"
                "    r.add('/users/<id>', 'user')\n"
                "    assert r.match('/posts/1') == (None, {})\n"
            ),
        },
    ),
    "mini_money_006": MiniInstance(
        instance_id="mini_money_006",
        difficulty="medium",
        problem_statement=(
            "Currency amounts are occasionally a cent short. Building a Money value from "
            "a dollar float loses a penny for some inputs because of binary float "
            "representation.\n\n"
            "Reproduction:\n"
            "    Money.from_dollars(0.29).cents   # -> 28, expected 29\n"
            "    Money.from_dollars(0.58).cents   # -> 57, expected 58\n\n"
            "Whole-cent inputs must be exact; addition of two amounts must stay exact."
        ),
        files={
            "money/__init__.py": "from money.amount import Money\n",
            "money/amount.py": (
                "class Money:\n"
                '    """Integer-cent money value."""\n\n'
                "    def __init__(self, cents):\n"
                "        self.cents = cents\n\n"
                "    @classmethod\n"
                "    def from_dollars(cls, dollars):\n"
                "        return cls(int(dollars * 100))\n\n"
                "    def __add__(self, other):\n"
                "        return Money(self.cents + other.cents)\n\n"
                "    def __eq__(self, other):\n"
                "        return isinstance(other, Money) and self.cents == other.cents\n\n"
                "    def dollars(self):\n"
                "        return self.cents / 100\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_amount.py": (
                "from money import Money\n\n\n"
                "def test_from_dollars_rounds_to_nearest_cent():\n"
                "    assert Money.from_dollars(0.29).cents == 29\n"
                "    assert Money.from_dollars(0.58).cents == 58\n"
                "    assert Money.from_dollars(1.00).cents == 100\n\n\n"
                "def test_addition_is_exact():\n"
                "    total = Money.from_dollars(0.29) + Money.from_dollars(0.01)\n"
                "    assert total == Money(30)\n"
            ),
        },
    ),
    "mini_retry_007": MiniInstance(
        instance_id="mini_retry_007",
        difficulty="medium",
        problem_statement=(
            "The @retry decorator gives up one attempt too early and also breaks "
            "introspection.\n\n"
            "A function decorated with retry(times=3) that succeeds on its third call "
            "still raises, because only 2 attempts are made. Separately, decorated "
            "functions lose their original __name__ (it becomes 'wrapper'), which breaks "
            "logging.\n\n"
            "Reproduction:\n"
            "    @retry(times=3)\n"
            "    def flaky(): ...  # fails twice, then succeeds\n"
            "    flaky()           # raises instead of returning the third-attempt value\n"
        ),
        files={
            "resilient/__init__.py": "from resilient.retry import retry\n",
            "resilient/retry.py": (
                "def retry(times):\n"
                '    """Retry a callable up to `times` attempts before re-raising."""\n\n'
                "    def decorator(fn):\n"
                "        def wrapper(*args, **kwargs):\n"
                "            last_error = None\n"
                "            for _ in range(times - 1):\n"
                "                try:\n"
                "                    return fn(*args, **kwargs)\n"
                "                except Exception as error:\n"
                "                    last_error = error\n"
                "            raise last_error\n"
                "        return wrapper\n"
                "    return decorator\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_retry.py": (
                "from resilient import retry\n\n\n"
                "def test_succeeds_on_final_attempt():\n"
                "    calls = {'n': 0}\n\n"
                "    @retry(times=3)\n"
                "    def flaky():\n"
                "        calls['n'] += 1\n"
                "        if calls['n'] < 3:\n"
                "            raise ValueError('not yet')\n"
                "        return 'ok'\n\n"
                "    assert flaky() == 'ok'\n"
                "    assert calls['n'] == 3\n\n\n"
                "def test_preserves_function_name():\n"
                "    @retry(times=2)\n"
                "    def my_function():\n"
                "        return 1\n\n"
                "    assert my_function.__name__ == 'my_function'\n"
            ),
        },
    ),
    "mini_confmerge_008": MiniInstance(
        instance_id="mini_confmerge_008",
        difficulty="medium",
        problem_statement=(
            "Config merging is destructive. Merging an override config over a base "
            "replaces whole nested sections instead of merging them, so keys that only "
            "exist in the base are lost when any key in that section is overridden.\n\n"
            "Reproduction:\n"
            "    base = {'db': {'host': 'local', 'port': 5432}}\n"
            "    over = {'db': {'port': 6000}}\n"
            "    deep_merge(base, over)\n"
            "    # -> {'db': {'port': 6000}}\n"
            "    # expected {'db': {'host': 'local', 'port': 6000}}\n\n"
            "The merge must also not mutate the inputs."
        ),
        files={
            "confmerge/__init__.py": "from confmerge.deepmerge import deep_merge\n",
            "confmerge/deepmerge.py": (
                "def deep_merge(base, override):\n"
                '    """Merge `override` onto `base`, recursing into nested dicts."""\n'
                "    result = dict(base)\n"
                "    for key, value in override.items():\n"
                "        result[key] = value\n"
                "    return result\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_deepmerge.py": (
                "from confmerge import deep_merge\n\n\n"
                "def test_merges_disjoint_keys():\n"
                "    assert deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}\n\n\n"
                "def test_merges_nested_sections():\n"
                "    base = {'db': {'host': 'local', 'port': 5432}}\n"
                "    over = {'db': {'port': 6000}}\n"
                "    assert deep_merge(base, over) == {'db': {'host': 'local', 'port': 6000}}\n\n\n"
                "def test_does_not_mutate_inputs():\n"
                "    base = {'x': {'y': 1}}\n"
                "    deep_merge(base, {'x': {'z': 2}})\n"
                "    assert base == {'x': {'y': 1}}\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- hard ----
    "mini_lru_009": MiniInstance(
        instance_id="mini_lru_009",
        difficulty="hard",
        problem_statement=(
            "The LRU cache evicts the wrong entry. After reading a key it should be "
            "considered most-recently-used, but recently-read keys are being evicted "
            "while stale ones survive, and the cache sometimes drops the value that was "
            "just inserted.\n\n"
            "Reproduction:\n"
            "    c = LRUCache(2)\n"
            "    c.put('a', 1); c.put('b', 2)\n"
            "    c.get('a')        # touch 'a' so 'b' is now least-recently-used\n"
            "    c.put('c', 3)     # should evict 'b'\n"
            "    c.get('b')        # -> 2, expected None\n"
            "    c.get('a')        # -> None, expected 1\n"
        ),
        files={
            "cache/__init__.py": "from cache.lru import LRUCache\n",
            "cache/lru.py": (
                "from collections import OrderedDict\n\n\n"
                "class LRUCache:\n"
                '    """Fixed-capacity least-recently-used cache."""\n\n'
                "    def __init__(self, capacity):\n"
                "        self.capacity = capacity\n"
                "        self._data = OrderedDict()\n\n"
                "    def get(self, key):\n"
                "        if key not in self._data:\n"
                "            return None\n"
                "        return self._data[key]\n\n"
                "    def put(self, key, value):\n"
                "        self._data[key] = value\n"
                "        if len(self._data) > self.capacity:\n"
                "            self._data.popitem(last=True)\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_lru.py": (
                "from cache import LRUCache\n\n\n"
                "def test_evicts_least_recently_used():\n"
                "    c = LRUCache(2)\n"
                "    c.put('a', 1)\n"
                "    c.put('b', 2)\n"
                "    assert c.get('a') == 1\n"
                "    c.put('c', 3)\n"
                "    assert c.get('b') is None\n"
                "    assert c.get('a') == 1\n"
                "    assert c.get('c') == 3\n\n\n"
                "def test_update_existing_key_keeps_it():\n"
                "    c = LRUCache(2)\n"
                "    c.put('a', 1)\n"
                "    c.put('b', 2)\n"
                "    c.put('a', 10)\n"
                "    c.put('c', 3)\n"
                "    assert c.get('a') == 10\n"
                "    assert c.get('b') is None\n"
            ),
        },
    ),
    "mini_graph_010": MiniInstance(
        instance_id="mini_graph_010",
        difficulty="hard",
        problem_statement=(
            "shortest_path does not return the shortest path. On graphs where the goal "
            "is reachable both directly and via a longer chain, it returns a longer "
            "route. It also has no protection against cycles.\n\n"
            "Reproduction:\n"
            "    g = {'a': ['g', 'b'], 'b': ['g'], 'g': []}\n"
            "    shortest_path(g, 'a', 'g')   # -> ['a', 'b', 'g'], expected ['a', 'g']\n"
        ),
        files={
            "graphlib2/__init__.py": "from graphlib2.search import shortest_path\n",
            "graphlib2/search.py": (
                "def shortest_path(graph, start, goal):\n"
                '    """Return the shortest path from start to goal, or None."""\n'
                "    stack = [[start]]\n"
                "    while stack:\n"
                "        path = stack.pop()\n"
                "        node = path[-1]\n"
                "        if node == goal:\n"
                "            return path\n"
                "        for nxt in graph.get(node, []):\n"
                "            stack.append(path + [nxt])\n"
                "    return None\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_search.py": (
                "from graphlib2 import shortest_path\n\n\n"
                "def test_prefers_shorter_path():\n"
                "    g = {'a': ['g', 'b'], 'b': ['g'], 'g': []}\n"
                "    assert shortest_path(g, 'a', 'g') == ['a', 'g']\n\n\n"
                "def test_finds_path_through_chain():\n"
                "    g = {'a': ['b'], 'b': ['c'], 'c': ['d'], 'd': []}\n"
                "    assert shortest_path(g, 'a', 'd') == ['a', 'b', 'c', 'd']\n\n\n"
                "def test_terminates_on_cycle():\n"
                "    g = {'a': ['b'], 'b': ['a', 'c'], 'c': []}\n"
                "    assert shortest_path(g, 'a', 'c') == ['a', 'b', 'c']\n\n\n"
                "def test_unreachable_returns_none():\n"
                "    g = {'a': ['b'], 'b': [], 'c': []}\n"
                "    assert shortest_path(g, 'a', 'c') is None\n"
            ),
        },
    ),
    "mini_intervals_011": MiniInstance(
        instance_id="mini_intervals_011",
        difficulty="hard",
        problem_statement=(
            "Merging overlapping intervals can shrink a range. When a fully-contained "
            "interval follows a larger one, the merged interval's end is replaced by the "
            "smaller interval's end instead of keeping the larger end.\n\n"
            "Reproduction:\n"
            "    merge_intervals([(1, 10), (2, 3), (4, 5)])\n"
            "    # -> [(1, 3), (4, 5)], expected [(1, 10)]\n\n"
            "Adjacent/touching intervals like (1, 2) and (2, 3) should merge into (1, 3)."
        ),
        files={
            "intervals/__init__.py": "from intervals.merge import merge_intervals\n",
            "intervals/merge.py": (
                "def merge_intervals(intervals):\n"
                '    """Merge overlapping/touching (start, end) intervals."""\n'
                "    merged = []\n"
                "    for start, end in sorted(intervals):\n"
                "        if merged and start <= merged[-1][1]:\n"
                "            merged[-1] = (merged[-1][0], end)\n"
                "        else:\n"
                "            merged.append((start, end))\n"
                "    return merged\n"
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_merge.py": (
                "from intervals import merge_intervals\n\n\n"
                "def test_merges_overlap():\n"
                "    assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]\n\n\n"
                "def test_keeps_outer_end_for_contained():\n"
                "    assert merge_intervals([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]\n\n\n"
                "def test_merges_touching():\n"
                "    assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]\n"
            ),
        },
    ),
    "mini_csv_012": MiniInstance(
        instance_id="mini_csv_012",
        difficulty="hard",
        problem_statement=(
            "The CSV line parser splits inside quoted fields. A field wrapped in double "
            "quotes that contains a comma is broken into multiple fields, and the "
            "surrounding quotes are not stripped.\n\n"
            "Reproduction:\n"
            "    parse_line('a,\"b,c\",d')\n"
            "    # -> ['a', '\"b', 'c\"', 'd']\n"
            "    # expected ['a', 'b,c', 'd']\n\n"
            "Plain unquoted lines must still parse, and a trailing newline must be "
            "stripped from the last field."
        ),
        files={
            "tinycsv/__init__.py": "from tinycsv.parser import parse_line\n",
            "tinycsv/parser.py": (
                "def parse_line(line):\n"
                '    """Parse one CSV line into a list of fields."""\n'
                '    return line.rstrip("\\n").split(",")\n'
            ),
            "tests/conftest.py": _conftest(),
            "tests/test_parser.py": (
                "from tinycsv import parse_line\n\n\n"
                "def test_plain_fields():\n"
                "    assert parse_line('a,b,c') == ['a', 'b', 'c']\n\n\n"
                "def test_quoted_field_with_comma():\n"
                "    assert parse_line('a,\"b,c\",d') == ['a', 'b,c', 'd']\n\n\n"
                "def test_strips_trailing_newline():\n"
                "    assert parse_line('x,y\\n') == ['x', 'y']\n"
            ),
        },
    ),
}


def list_instance_ids() -> list[str]:
    return sorted(MINI_INSTANCES.keys())


def get_instance(instance_id: str) -> MiniInstance:
    if instance_id not in MINI_INSTANCES:
        raise KeyError(f"Unknown mini instance: {instance_id}")
    return MINI_INSTANCES[instance_id]


def load_instance_ids_from_file(path) -> list[str]:
    from pathlib import Path

    ids: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids
