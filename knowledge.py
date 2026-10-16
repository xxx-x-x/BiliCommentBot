"""Knowledge text preparation helpers."""
from __future__ import annotations

import re
from typing import Any, Iterable


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 3))
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        parts = _split_long_paragraph(paragraph, chunk_size)
        for part in parts:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                prefix = current[-overlap:].lstrip() if overlap else ""
                current = f"{prefix}\n{part}".strip() if prefix else part
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;\.])", paragraph) if s.strip()]
    if len(sentences) == 1:
        return [paragraph[i:i + chunk_size] for i in range(0, len(paragraph), chunk_size)]

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                parts.append(current)
                current = ""
            parts.extend(sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size))
        elif len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def build_knowledge_context(rows: Iterable[dict[str, Any]], max_chars: int = 5000) -> str:
    max_chars = max(500, int(max_chars))
    sections: list[str] = []
    used = 0
    for index, row in enumerate(rows, 1):
        source_title = str(row.get("source_title") or row.get("title") or "视频知识").strip()
        source_url = str(row.get("source_url") or row.get("url") or "").strip()
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        header = f"[{index}] {source_title}"
        if source_url:
            header += f"\n来源：{source_url}"
        section = f"{header}\n{content}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining].rstrip()
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)
