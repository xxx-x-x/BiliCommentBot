"""AI-assisted knowledge distillation with strict structured-output validation."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

import requests

from ai_client import AIClientError, request_chat_completion


class DistillationError(RuntimeError):
    pass


def distill_sources(
    query: str,
    sources: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    query = str(query).strip()
    if not query:
        raise ValueError("蒸馏问题不能为空")
    if not sources:
        raise ValueError("至少需要一个资料来源")

    evidence = []
    allowed_ids = set()
    for source in sources:
        source_id = int(source["id"])
        allowed_ids.add(source_id)
        content = str(source.get("content", "")).strip()
        evidence.append({
            "source_id": source_id,
            "title": str(source.get("title", "")),
            "url": str(source.get("url", "") or ""),
            "content": content[:6000],
        })

    system = (
        "你是严谨的知识蒸馏助手。只能使用给定证据，不得补充外部知识。"
        "去重并提取可核验事实；来源说法不一致时写入 conflicts；证据不足时降低 confidence。"
        "只输出一个 JSON 对象，不要输出 Markdown。字段必须为："
        "topic(string), summary(string), facts(string[]), limitations(string[]), "
        "conflicts(string[]), source_ids(integer[]), confidence(number, 0到1)。"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"query": query, "evidence": evidence}, ensure_ascii=False)},
    ]
    class _PostAdapter:
        @staticmethod
        def post(*args, **kwargs):
            return post(*args, **kwargs)
    try:
        raw = request_chat_completion(
            config,
            messages,
            temperature=0.1,
            max_tokens=max(500, int(config.get("max_tokens", 1200))),
            json_mode=True,
            timeout=60,
            session=_PostAdapter,
        )
    except AIClientError as exc:
        raise DistillationError(str(exc)) from exc
    return parse_distillation(raw, allowed_ids)


def parse_distillation(raw: str | Mapping[str, Any], allowed_source_ids: set[int]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        data = dict(raw)
    else:
        text = str(raw).strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DistillationError("AI 返回的蒸馏结果不是有效 JSON") from exc
    if not isinstance(data, dict):
        raise DistillationError("AI 蒸馏结果必须是 JSON 对象")

    result = {
        "topic": _required_text(data.get("topic"), "topic"),
        "summary": _required_text(data.get("summary"), "summary"),
        "facts": _string_list(data.get("facts"), "facts", required=True),
        "limitations": _string_list(data.get("limitations", []), "limitations"),
        "conflicts": _string_list(data.get("conflicts", []), "conflicts"),
    }
    source_ids = data.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise DistillationError("source_ids 必须是非空数组")
    try:
        normalized_ids = list(dict.fromkeys(int(item) for item in source_ids))
    except (TypeError, ValueError) as exc:
        raise DistillationError("source_ids 只能包含整数") from exc
    if not set(normalized_ids).issubset(allowed_source_ids):
        raise DistillationError("AI 引用了不存在的资料来源")
    result["source_ids"] = normalized_ids

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise DistillationError("confidence 必须是 0 到 1 的数字")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise DistillationError("confidence 必须在 0 到 1 之间")
    result["confidence"] = confidence
    return result


def format_distillation(item: Mapping[str, Any]) -> str:
    lines = [f"主题：{item['topic']}", "", f"摘要：{item['summary']}", "", "事实："]
    lines.extend(f"- {fact}" for fact in item["facts"])
    if item.get("limitations"):
        lines.extend(["", "限制与注意事项："])
        lines.extend(f"- {value}" for value in item["limitations"])
    if item.get("conflicts"):
        lines.extend(["", "来源冲突："])
        lines.extend(f"- {value}" for value in item["conflicts"])
    lines.extend(["", f"可信度：{float(item['confidence']):.0%}"])
    return "\n".join(lines)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DistillationError(f"{name} 必须是非空字符串")
    return value.strip()


def _string_list(value: Any, name: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DistillationError(f"{name} 必须是字符串数组")
    result = [item.strip() for item in value if item.strip()]
    if required and not result:
        raise DistillationError(f"{name} 不能为空")
    return result
