"""Tests for the web search skill."""

import asyncio

import pytest

from friday.skills.web_search_skill import WebSearchSkill


@pytest.mark.asyncio
async def test_web_search_skill_caches_results(monkeypatch):
    async def fake_fetch(self, query):
        return """
        <html>
          <a class="result__a" href="https://example.com/friday">Friday Title</a>
          <div class="result__snippet">Friday Body</div>
        </html>
        """

    monkeypatch.setattr("friday.skills.web_search_skill.WebSearchSkill._fetch_search_page", fake_fetch)

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

    async def fake_fetch(self, query):
        return f"""
        <html>
          <a class="result__a" href="https://example.com/{query}">{query}</a>
          <div class="result__snippet">body</div>
        </html>
        """

    monkeypatch.setattr("friday.skills.web_search_skill.WebSearchSkill._fetch_search_page", fake_fetch)
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


def test_web_search_skill_parses_duckduckgo_html():
    html = """
    <html>
      <a class="result__a" href="https://example.com/alpha">Alpha <b>Title</b></a>
      <div class="result__snippet">Alpha snippet</div>
      <a class="result__a" href="https://example.com/beta">Beta Title</a>
      <a class="result__snippet">Beta snippet</a>
    </html>
    """

    results = WebSearchSkill._parse_results(html)

    assert results == [
        {
            "title": "Alpha Title",
            "href": "https://example.com/alpha",
            "body": "Alpha snippet",
        },
        {
            "title": "Beta Title",
            "href": "https://example.com/beta",
            "body": "Beta snippet",
        },
    ]
