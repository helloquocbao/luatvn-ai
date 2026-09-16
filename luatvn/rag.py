"""Ghép ngữ cảnh truy xuất thành prompt và sinh câu trả lời có trích dẫn."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import AsyncIterator

from . import config, llm, retrieve
from .retrieve import Hit

SYSTEM = """Bạn là trợ lý pháp luật Việt Nam. Bạn trả lời dựa TRÊN VÀ CHỈ TRÊN các nguồn được cung cấp.

QUY TẮC BẮT BUỘC:
1. Mỗi khẳng định pháp lý phải kèm số nguồn dạng [1], [2]. Không có nguồn thì không khẳng định.
2. Nếu các nguồn không đủ để trả lời, hãy nói thẳng: "Các nguồn hiện có chưa đủ để trả lời chính xác câu hỏi này" rồi nêu phần nào trả lời được.
3. TUYỆT ĐỐI không bịa số hiệu văn bản, số Điều, số Khoản hay mức phạt. Chỉ dùng đúng những gì có trong nguồn.
4. Nếu nguồn được đánh dấu [ĐÃ HẾT HIỆU LỰC], phải cảnh báo rõ cho người đọc.
5. Khi các nguồn mâu thuẫn nhau, nêu rõ mâu thuẫn và ưu tiên văn bản có hiệu lực pháp lý cao hơn và mới hơn.

CÁCH TRẢ LỜI:
- Viết tiếng Việt, rõ ràng, đi thẳng vào vấn đề.
- Mở đầu bằng câu trả lời ngắn gọn, sau đó mới giải thích chi tiết kèm căn cứ.
- Trích nguyên văn quy định quan trọng, đặt trong dấu ngoặc kép.
- Kết thúc bằng một dòng: "⚖️ Thông tin mang tính tham khảo, không thay thế tư vấn pháp lý chính thức."
"""


def format_source(i: int, h: Hit) -> str:
    """Định dạng một nguồn cho prompt, kèm đủ metadata để LLM trích dẫn đúng."""
    parts: list[str] = []
    if h.kind == "chunk":
        m = h.meta
        head = " ".join(x for x in (m.get("doc_type"), m.get("doc_number")) if x)
        parts.append(f"[{i}] {head} — {h.title}".strip())
        if m.get("chuong"):
            parts.append(f"    {m['chuong']}")
        if m.get("expired"):
            parts.append(f"    ⚠️ [ĐÃ HẾT HIỆU LỰC — {m.get('status') or 'không rõ ngày'}]")
    elif h.kind == "qa":
        parts.append(f"[{i}] Hỏi đáp pháp luật ({h.meta.get('area') or 'chung'}) — {h.title}")
    elif h.kind == "term":
        parts.append(f"[{i}] Thuật ngữ pháp lý: {h.title}")
    else:
        m = h.meta
        head = " ".join(x for x in (h.label, m.get("doc_number")) if x)
        parts.append(f"[{i}] {head} — {h.title}")
        parts.append(f"    (chỉ có thông tin định danh, chưa có toàn văn trong cơ sở dữ liệu)")
    body = re.sub(r"\n{2,}", "\n", h.text.strip())
    parts.append(body)
    return "\n".join(parts)


def build_context(hits: list[Hit], max_chars: int = config.CONTEXT_MAX_CHARS) -> tuple[str, list[Hit]]:
    """Ghép các nguồn lại, cắt khi chạm trần ngữ cảnh của model."""
    blocks, used, total = [], [], 0
    for h in hits:
        block = format_source(len(used) + 1, h)
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(h)
        total += len(block)
    return "\n\n---\n\n".join(blocks), used


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> tuple[list[dict], list[Hit]]:
    context, used = build_context(hits)
    if not used:
        user = (f"Câu hỏi: {question}\n\n"
                "Không tìm thấy nguồn nào liên quan trong cơ sở dữ liệu. "
                "Hãy nói rõ điều đó với người dùng và gợi ý cách đặt câu hỏi cụ thể hơn "
                "(nêu tên luật, lĩnh vực, hoặc tình huống chi tiết). Không được tự suy đoán quy định.")
    else:
        user = (f"NGUỒN THAM KHẢO:\n\n{context}\n\n"
                f"{'=' * 60}\n\nCÂU HỎI: {question}\n\n"
                "Trả lời dựa trên các nguồn trên, kèm số trích dẫn [n].")
    msgs = [{"role": "system", "content": SYSTEM}]
    for turn in (history or [])[-4:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            msgs.append({"role": turn["role"], "content": turn["content"][:2000]})
    msgs.append({"role": "user", "content": user})
    return msgs, used


def answer(
    con: sqlite3.Connection,
    question: str,
    *,
    history: list[dict] | None = None,
    top_k: int = config.CONTEXT_TOPK,
    model: str | None = None,
) -> dict:
    hits = retrieve.retrieve(con, question, top_k=top_k)
    msgs, used = build_messages(question, hits, history)
    text = llm.chat(msgs, model=model)
    return {"answer": text, "sources": used}


async def astream_answer(
    con: sqlite3.Connection,
    question: str,
    *,
    history: list[dict] | None = None,
    top_k: int = config.CONTEXT_TOPK,
    model: str | None = None,
) -> AsyncIterator[dict]:
    """Sinh câu trả lời theo dòng; phát nguồn trước để giao diện hiện ngay."""
    hits = retrieve.retrieve(con, question, top_k=top_k)
    msgs, used = build_messages(question, hits, history)
    yield {"type": "sources", "sources": [serialize(h, i + 1) for i, h in enumerate(used)]}
    async for piece in llm.achat_stream(msgs, model=model):
        yield {"type": "token", "text": piece}
    yield {"type": "done"}


def serialize(h: Hit, index: int) -> dict:
    return {
        "index": index,
        "kind": h.kind,
        "ref": h.ref,
        "title": h.title,
        "label": h.label,
        "citation": h.citation,
        "snippet": h.text[:600],
        "url": h.url,
        "doc_id": h.ref.split("#")[0] if h.kind == "chunk" else (h.ref if h.kind == "doc" else ""),
        "meta": h.meta,
        "score": round(h.score, 4),
        "sources": h.sources,
    }
