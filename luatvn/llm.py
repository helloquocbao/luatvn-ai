"""Kết nối tới Ollama chạy local (giao thức /api/chat, có streaming)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator

import httpx

from . import config


class OllamaError(RuntimeError):
    pass


def available() -> bool:
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def model_ready(model: str | None = None) -> bool:
    """Ollama đang chạy là chưa đủ — model được cấu hình phải thực sự có mặt."""
    want = (model or config.LLM_MODEL).split(":")[0]
    return any(m.split(":")[0] == want for m in list_models())


# Các model không hỗ trợ suy luận sẽ trả 400 nếu nhận tham số `think`. Ghi nhớ
# lại để chỉ thử một lần cho mỗi model thay vì lỗi đi lỗi lại.
_NO_THINK: set[str] = set()


def _supports_think(model: str) -> bool:
    return model not in _NO_THINK


def _payload(messages: list[dict], *, stream: bool, model: str | None, think: bool) -> dict:
    name = model or config.LLM_MODEL
    body: dict = {
        "model": name,
        "messages": messages,
        "stream": stream,
        "options": {
            "num_ctx": config.LLM_NUM_CTX,
            # nhiệt độ thấp: trả lời pháp luật cần bám sát văn bản, không sáng tạo
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
    }
    if _supports_think(name):
        # tắt phần suy luận dài dòng của qwen3 -> trả lời nhanh hơn nhiều
        body["think"] = think
    return body


def chat(messages: list[dict], *, model: str | None = None, think: bool = False) -> str:
    name = model or config.LLM_MODEL
    for attempt in (1, 2):
        try:
            r = httpx.post(f"{config.OLLAMA_HOST}/api/chat",
                           json=_payload(messages, stream=False, model=model, think=think),
                           timeout=300.0)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if attempt == 1 and e.response.status_code == 400 and _supports_think(name):
                _NO_THINK.add(name)   # model không nhận `think` -> thử lại không có
                continue
            raise OllamaError(f"Ollama trả lỗi {e.response.status_code}: "
                              f"{e.response.text[:200]}") from e
        except httpx.HTTPError as e:
            raise OllamaError(f"Không gọi được Ollama tại {config.OLLAMA_HOST}: {e}") from e
        return _strip_think(r.json().get("message", {}).get("content", ""))
    return ""


def _strip_think(text: str) -> str:
    """Bỏ khối <think>...</think> khi model vẫn suy luận dù đã tắt."""
    import re

    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


async def achat_stream(
    messages: list[dict], *, model: str | None = None, think: bool = False
) -> AsyncIterator[str]:
    name = model or config.LLM_MODEL
    body = _payload(messages, stream=True, model=model, think=think)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as cli:
        async with cli.stream("POST", f"{config.OLLAMA_HOST}/api/chat", json=body) as r:
            if r.status_code == 400 and _supports_think(name):
                _NO_THINK.add(name)
                body = _payload(messages, stream=True, model=model, think=think)
                async with cli.stream("POST", f"{config.OLLAMA_HOST}/api/chat", json=body) as r2:
                    r2.raise_for_status()
                    async for piece in _iter_sse(r2):
                        yield piece
                    return
            r.raise_for_status()
            async for piece in _iter_sse(r):
                yield piece


async def _iter_sse(r) -> AsyncIterator[str]:
    async for line in r.aiter_lines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        piece = obj.get("message", {}).get("content", "")
        if piece:
            yield piece
        if obj.get("done"):
            return


def chat_stream(messages: list[dict], *, model: str | None = None,
                think: bool = False) -> Iterator[str]:
    body = _payload(messages, stream=True, model=model, think=think)
    with httpx.stream("POST", f"{config.OLLAMA_HOST}/api/chat", json=body,
                      timeout=httpx.Timeout(300.0, connect=5.0)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                return
