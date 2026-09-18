"""Dataset loaders and the cap counter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from fleetbench import datasets as ds
from fleetbench.caps import CapCounter, load_caps


def test_all_datasets_load_and_build_prompts():
    for name in ds.DATASET_NAMES:
        items = ds.load(name, 1)
        assert len(items) == 1
        it = items[0]
        assert it.id and it.prompt and it.dataset == name
        assert it.grader


def test_load_respects_count_and_pinned_order():
    items = ds.load("sanity", 3)
    assert [i.id for i in items] == ["sanity-000", "sanity-001", "sanity-002"]


def test_sdlc_carries_task_type_and_rubric():
    items = ds.load("sdlc", 6)
    task_types = {i.meta["task_type"] for i in items}
    assert task_types == {
        "coding", "design", "requirement writing",
        "test writing", "code review", "requirement review",
    }
    for i in items:
        assert len(i.meta["rubric"]) == 4


def test_fixture_meta_has_license_and_provenance():
    meta = ds.fixture_meta("humaneval")
    assert meta["license"]
    assert meta["provenance"]


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        ds.load("nope")


def test_oversized_count_raises_clear_guard():
    # configs/phase2.yaml requests full-size counts (e.g. humaneval 164) but
    # only Phase-1-sized vendored fixtures exist today; this must fail loudly
    # rather than silently truncate or misbehave.
    with pytest.raises(ValueError, match="humaneval.*164.*only"):
        ds.load("humaneval", 164)


def test_within_fixture_count_still_works():
    items = ds.load("humaneval", 1)
    assert len(items) == 1


def test_cap_counter_enforces_limit():
    caps = {"providers": {"copilot": {"max_calls": 2}, "groq": {"max_calls": 0}}}
    c = CapCounter(caps)
    assert c.try_reserve("copilot") is True
    assert c.try_reserve("copilot") is True
    assert c.try_reserve("copilot") is False  # cap hit
    # 0 == unlimited
    for _ in range(10):
        assert c.try_reserve("groq") is True
    assert c.count("copilot") == 2


def test_judge_cap_enforced_from_top_level_key():
    # caps.yaml declares judge.max_calls as a top-level key, not under providers.
    caps = {"judge": {"max_calls": 3}, "providers": {}}
    c = CapCounter(caps)
    assert [c.try_reserve("judge") for _ in range(4)] == [True, True, True, False]


def test_rpm_and_concurrency_read_from_caps_yaml():
    caps = load_caps()  # the repo caps.yaml
    c = CapCounter(caps)
    assert c.rpm_for("groq") == 20.0
    assert c.concurrency_for("groq") == 1
    # A provider with no concurrency key defaults to 1.
    assert c.concurrency_for("hetzner") == 1


def test_cap_counter_is_thread_safe_under_contention():
    c = CapCounter({"providers": {"p": {"max_calls": 100}}})
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: c.try_reserve("p"), range(1000)))
    assert sum(results) == 100  # never over the cap despite concurrency
    assert c.count("p") == 100
