"""Configurable web-search providers for knowledge-base enrichment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "content": self.content}


class SearchProviderError(RuntimeError):
    pass


def _secret(config: dict[str, Any], name: str, env_name: str) -> str:
    return str(config.get(name, "") or os.environ.get(env_name, "")).strip()


def search_web(query: str, config: dict[str, Any], limit: int = 5) -> list[SearchResult]:
    query = query.strip()
    if not query:
        raise ValueError("搜索关键词不能为空")

    limit = max(1, min(int(limit), 10))
    provider = str(config.get("provider", "disabled")).strip().lower()
    timeout = max(3, min(int(config.get("timeout", 15)), 60))

    if provider in {"", "disabled", "none"}:
        return []
    if provider == "tavily":
        return _search_tavily(query, config, limit, timeout)
    if provider == "bing":
        return _search_bing(query, config, limit, timeout)
    raise SearchProviderError(f"不支持的搜索服务: {provider}")


def _search_tavily(query: str, config: dict[str, Any], limit: int, timeout: int) -> list[SearchResult]:
    api_key = _secret(config, "api_key", "TAVILY_API_KEY")
    if not api_key:
        raise SearchProviderError("未配置 Tavily API Key")

    endpoint = str(config.get("endpoint", "") or "https://api.tavily.com/search").strip()
    try:
        response = requests.post(
            endpoint,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SearchProviderError(f"Tavily 搜索失败: {exc}") from exc

    return [
        SearchResult(
            title=str(item.get("title", "")).strip(),
            url=str(item.get("url", "")).strip(),
            content=str(item.get("content", "")).strip(),
        )
        for item in payload.get("results", [])[:limit]
        if item.get("url") and item.get("content")
    ]


def _search_bing(query: str, config: dict[str, Any], limit: int, timeout: int) -> list[SearchResult]:
    api_key = _secret(config, "api_key", "BING_SEARCH_API_KEY")
    if not api_key:
        raise SearchProviderError("未配置 Bing Search API Key")

    endpoint = str(config.get("endpoint", "") or "https://api.bing.microsoft.com/v7.0/search").strip()
    try:
        response = requests.get(
            endpoint,
            params={"q": query, "count": limit, "mkt": "zh-CN", "textDecorations": False},
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SearchProviderError(f"Bing 搜索失败: {exc}") from exc

    values = payload.get("webPages", {}).get("value", [])
    return [
        SearchResult(
            title=str(item.get("name", "")).strip(),
            url=str(item.get("url", "")).strip(),
            content=str(item.get("snippet", "")).strip(),
        )
        for item in values[:limit]
        if item.get("url") and item.get("snippet")
    ]
