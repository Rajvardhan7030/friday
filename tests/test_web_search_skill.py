"""Tests for the web search skill."""

import asyncio

import pytest

from friday.skills.web_search_skill import WebSearchSkill


class FakeDDGS:
    """Simple DDGS stub for deterministic tests."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, max_results=5):
        return [{"title": query, "href": f"https://example.com/{query}", "body": "body"}]


@pytest.mark.asyncio
async def test_web_search_skill_caches_results(monkeypatch):
    monkeypatch.setattr("friday.skills.web_search_skill.DDGS", FakeDDGS)

    skill = WebSearchSkill()

    first = await skill.execute("friday", {})
    second = await skill.execute("friday", {})

    assert first.success is True
    assert second.success is True
    assert first.data == second.data


@pytest.mark.asyncio
async def test_web_search_skill_rate_limits_concurrent_requests(monkeypatch):
    current_time = {"value": 100.0}
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        current_time["value"] += duration

    monkeypatch.setattr("friday.skills.web_search_skill.DDGS", FakeDDGS)
    monkeypatch.setattr("friday.skills.web_search_skill.time.monotonic", lambda: current_time["value"])
    monkeypatch.setattr("friday.skills.web_search_skill.asyncio.sleep", fake_sleep)

    skill = WebSearchSkill()

    first, second = await asyncio.gather(
        skill.execute("first", {}),
        skill.execute("second", {}),
    )

    assert first.success is True
    assert second.success is True
    assert sleep_calls == [1.0]
