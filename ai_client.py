"""Minimal client helpers for OpenAI-compatible Chat Completions APIs."""
from __future__ import annotations

from typing import Any, Mapping

import requests


class AIClientError(RuntimeError):
    pass


def chat_completions_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("AI API 地址不能为空")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def resolve_ai_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return provider-neutral settings, accepting legacy deepseek config."""
    ai = config.get("ai")
    if isinstance(ai, Mapping) and ai:
        return dict(ai)
    legacy = config.get("deepseek")
    return dict(legacy) if isinstance(legacy, Mapping) else {}


def request_chat_completion(
    config: Mapping[str, Any],
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    timeout: int = 60,
    session: Any = requests,
) -> str:
    api_key = str(config.get("api_key", "")).strip()
    if not api_key:
        raise AIClientError("未配置 AI API Key")
    model = str(config.get("model", "")).strip()
    if not model:
        raise AIClientError("未配置 AI 模型名称")

    payload: dict[str, Any] = {"model": model, "messages": messages}
    payload["temperature"] = float(
        config.get("temperature", 0.7) if temperature is None else temperature
    )
    payload["max_tokens"] = int(
        config.get("max_tokens", 500) if max_tokens is None else max_tokens
    )
    if json_mode and bool(config.get("json_mode", False)):
        payload["response_format"] = {"type": "json_object"}

    try:
        response = session.post(
            chat_completions_url(str(config.get("base_url", ""))),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        raise AIClientError(f"AI 接口请求失败: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise AIClientError("AI 接口返回了空内容")
    return content.strip()
