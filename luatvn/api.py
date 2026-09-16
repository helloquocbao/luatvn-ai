"""FastAPI: chat RAG có trích dẫn + tra cứu toàn văn văn bản pháp luật.

    uv run uvicorn luatvn.api:app --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, db, llm, rag, retrieve

WEB = Path(__file__).parent / "web"
_con = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _con
    if not config.DB_PATH.exists():
        raise RuntimeError(
            f"Chưa có CSDL tại {config.DB_PATH}. Chạy:\n"
            "  uv run python -m luatvn.download && uv run python -m luatvn.ingest"
        )
    _con = db.connect(readonly=True)
    # Nạp sẵn model ngay lúc khởi động: nạp lazy khiến request ĐẦU TIÊN của
    # người dùng phải chờ thêm ~18 s (embedder 12,4 s + reranker 5,7 s).
    try:
        import asyncio

        from . import models

        await asyncio.gather(
            asyncio.to_thread(models.get_embedder),
            asyncio.to_thread(models.get_reranker),
        )
        # flush: stdout bị đệm theo khối khi ghi ra file, không có flush thì
        # thông báo khởi động (kể cả CẢNH BÁO nạp model hỏng) không bao giờ hiện.
        print("[luatvn] đã nạp sẵn embedder + reranker", flush=True)
    except Exception as e:
        print(f"[luatvn] CẢNH BÁO: không nạp sẵn được model ({e})", flush=True)
    yield
    if _con:
        _con.close()


app = FastAPI(title="Trợ lý pháp luật Việt Nam", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict] = Field(default_factory=list)
    top_k: int = Field(default=config.CONTEXT_TOPK, ge=1, le=20)
    model: str | None = None


@app.get("/api/health")
def health() -> dict:
    q = lambda s: _con.execute(s).fetchone()[0]
    try:
        from .embed import open_table

        n_vec = open_table(create=False).count_rows()
    except Exception:
        n_vec = 0
    return {
        "ok": True,
        "docs": q("SELECT count(*) FROM docs"),
        "docs_full_text": q("SELECT count(*) FROM docs WHERE has_text=1"),
        "articles": q("SELECT count(*) FROM chunks"),
        "qa": q("SELECT count(*) FROM qa"),
        "terms": q("SELECT count(*) FROM terms"),
        "vectors": n_vec,
        "embed_model": config.EMBED_MODEL,
        "llm_model": config.LLM_MODEL,
        # `llm_ready` từng chỉ kiểm tra Ollama có chạy không, nên vẫn báo sẵn
        # sàng khi model chưa tải xong — giao diện hiện xanh rồi mới lỗi lúc chat.
        "llm_ready": llm.available() and llm.model_ready(),
        "ollama_running": llm.available(),
        "llm_available_models": llm.list_models(),
    }


@app.post("/api/search")
def search(req: ChatRequest) -> dict:
    """Chỉ truy xuất, không gọi LLM — dùng để kiểm thử chất lượng tìm kiếm."""
    hits = retrieve.retrieve(_con, req.question, top_k=req.top_k)
    return {"sources": [rag.serialize(h, i + 1) for i, h in enumerate(hits)]}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not llm.available():
        raise HTTPException(
            503,
            f"Không kết nối được Ollama tại {config.OLLAMA_HOST}. "
            "Khởi động Ollama rồi thử lại, hoặc dùng /api/search để tra cứu không cần LLM.",
        )

    async def gen():
        try:
            async for evt in rag.astream_answer(
                _con, req.question, history=req.history, top_k=req.top_k, model=req.model
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as e:  # lỗi giữa dòng vẫn phải tới được giao diện
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: str) -> dict:
    doc = retrieve.get_document(_con, doc_id)
    if not doc:
        raise HTTPException(404, "Không tìm thấy văn bản")
    m = doc["meta"]
    m["display_title"] = m.get("title_vi") or m.get("title")
    m["display_number"] = m.get("number_vi") or m.get("doc_number")
    return doc


@app.get("/api/docs")
def list_docs(
    q: str = Query("", max_length=300),
    doc_type: str = Query(""),
    year: int | None = None,
    limit: int = Query(30, ge=1, le=100),
) -> dict:
    """Duyệt/tìm văn bản theo tiêu đề, số hiệu, loại, năm."""
    where, params = [], []
    if q:
        match = retrieve.fts_queries(_con, q)
        if match:
            rows = retrieve.bm25_docs(_con, match, limit * 3, query_text=q)
            out = [
                {"id": h.ref, "title": h.title, "doc_type": h.label,
                 "doc_number": h.meta.get("doc_number"), "year": h.meta.get("year"),
                 "issuer": h.meta.get("issuer"), "has_text": h.meta.get("has_text"),
                 "expired": h.meta.get("expired"), "url": h.url}
                for h in rows
                if (not doc_type or h.label == doc_type) and (year is None or h.meta.get("year") == year)
            ]
            return {"docs": out[:limit]}
    if doc_type:
        where.append("doc_type = ?"); params.append(doc_type)
    if year is not None:
        where.append("year = ?"); params.append(year)
    sql = ("SELECT id, COALESCE(title_vi,title) title, doc_type, "
           "COALESCE(number_vi,doc_number) doc_number, year, issuer, has_text, expired, url "
           "FROM docs")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY year DESC LIMIT ?"
    params.append(limit)
    return {"docs": [dict(r) for r in _con.execute(sql, params)]}


@app.get("/api/doc-types")
def doc_types() -> dict:
    rows = _con.execute(
        "SELECT doc_type, count(*) n FROM docs WHERE doc_type IS NOT NULL "
        "GROUP BY doc_type ORDER BY n DESC LIMIT 40").fetchall()
    return {"types": [dict(r) for r in rows]}


@app.get("/")
def index() -> FileResponse:
    # no-store: trang được nạp thẳng từ đĩa mỗi lần, nên sửa giao diện là thấy
    # ngay thay vì phải hard-reload để vượt qua cache của trình duyệt.
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})
