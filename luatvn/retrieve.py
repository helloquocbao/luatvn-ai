"""Truy xuất lai: BM25 (FTS5) + vector (LanceDB) -> RRF -> cross-encoder rerank.

Vì sao lai thay vì thuần vector: câu hỏi pháp luật thường chứa định danh chính
xác ("Điều 5 Luật Doanh nghiệp 2020", "Nghị định 100/2019"). Embedding rất kém
với loại chuỗi này, còn BM25 thì bắt chuẩn. Ngược lại BM25 mù với cách diễn đạt
khác từ ("bị sa thải" vs "đơn phương chấm dứt hợp đồng lao động"). Hai lớp bù
nhau, RRF gộp lại mà không cần chỉnh trọng số theo thang điểm khác nhau.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from dataclasses import dataclass, field

from . import config, db

logger = logging.getLogger(__name__)

# phải khớp chính xác tokenizer trong luatvn/db.py
TOKENIZE = "unicode61 remove_diacritics 2"

# Dạng đã qua tokenizer (bỏ dấu, giữ chữ đ) — dùng để lọc token truy vấn.
STOP_TOKENS = {
    "la", "va", "cua", "co", "đuoc", "cho", "trong", "khi", "thi", "nao", "gi",
    "toi", "khong", "mot", "cac", "nhung", "nay", "đo", "voi", "ve", "nhu",
    "đe", "tu", "đen", "bi", "boi", "hay", "hoac", "neu", "ma", "o", "ra",
    "vao", "theo", "se", "đa", "phai", "can", "muon", "hoi", "xin", "a", "vay",
    "the", "sao", "bao", "nhieu", "ai", "đau", "lam", "nhat", "hien",
}

RE_WORD = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)


@dataclass
class Hit:
    """Một kết quả truy xuất, đã đủ thông tin để trích dẫn."""

    kind: str            # 'chunk' | 'qa' | 'term' | 'doc'
    ref: str             # khoá gốc trong SQLite
    title: str           # tiêu đề văn bản / câu hỏi / tên thuật ngữ
    label: str           # nhãn trích dẫn: "Điều 12", "Hỏi đáp", ...
    text: str
    url: str = ""
    meta: dict = field(default_factory=dict)
    score: float = 0.0
    sources: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.label} — {self.title}" if self.label else self.title


# Token có mặt ở hơn ngần này phần trăm số đoạn sẽ bị loại khỏi truy vấn BM25.
# Chúng gần như không mang thông tin phân biệt mà lại quyết định chi phí truy vấn.
MAX_DF_RATIO = 0.35
# Số cụm 2 từ tối đa trong nhánh OR dự phòng.
MAX_OR_PHRASES = 4
_N_CHUNKS: int | None = None


def _corpus_size(con: sqlite3.Connection) -> int:
    global _N_CHUNKS
    if _N_CHUNKS is None:
        _N_CHUNKS = con.execute("SELECT count(*) FROM chunks").fetchone()[0] or 1
    return _N_CHUNKS


def _ensure_tokenizer(con: sqlite3.Connection) -> bool:
    """Tạo bảng tạm dùng ĐÚNG tokenizer của chỉ mục để tách token truy vấn.

    Không thể đoán token bằng Python: tokenizer giữ nguyên chữ "đ" nhưng bỏ dấu
    các nguyên âm, nên "điều" thành `đieu` chứ không phải `dieu`. Đoán sai thì
    tra tần suất trượt và việc lọc mất tác dụng.
    """
    try:
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.qtok USING fts5"
            f"(t, tokenize='{TOKENIZE}')"
        )
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.qtokv USING fts5vocab('qtok','instance')"
        )
        return True
    except sqlite3.OperationalError as e:
        logger.debug("không tạo được bảng tokenize tạm: %s", e)
        return False


def _tokens(con: sqlite3.Connection, text: str) -> list[str]:
    """Tách câu hỏi thành token đúng như chỉ mục đã tách."""
    text = unicodedata.normalize("NFC", text)
    if _ensure_tokenizer(con):
        try:
            con.execute("DELETE FROM temp.qtok")
            con.execute("INSERT INTO temp.qtok VALUES(?)", (text[:2000],))
            toks = [r[0] for r in con.execute("SELECT term FROM temp.qtokv ORDER BY offset")]
            if toks:
                return toks[:24]
        except sqlite3.OperationalError as e:
            logger.debug("tokenize bằng SQLite thất bại: %s", e)
    # dự phòng khi CSDL chỉ đọc không cho tạo bảng tạm
    return [t for t in RE_WORD.findall(text.lower()) if len(t) >= 2][:24]


def _doc_freq(con: sqlite3.Connection, toks: list[str]) -> dict[str, int]:
    """Số đoạn chứa mỗi token. Token vắng mặt trong bảng nghĩa là df < 2000."""
    if not toks:
        return {}
    ph = ",".join("?" * len(toks))
    return {r[0]: r[1] for r in
            con.execute(f"SELECT term, df FROM token_df WHERE term IN ({ph})", toks)}


def _quote(tokens: list[str]) -> list[str]:
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(f'"{t}"')
    return out


def fts_query(con: sqlite3.Connection, text: str) -> str:
    """OR các token — dùng cho bảng nhỏ (terms) nơi posting list ngắn."""
    toks = _quote(_tokens(con, text))
    return " OR ".join(toks) if toks else ""


def fts_queries(con: sqlite3.Connection, text: str) -> list[str]:
    """Sinh biểu thức MATCH theo thứ tự chính xác -> rộng dần.

    Đo trên bảng 1,86 triệu đoạn (giới hạn 40, có JOIN sang docs):
        OR token đơn   : 14,6 s   <- không dùng được, bỏ hẳn
        OR cụm 2 từ    :  1,1 s (tới 13,5 s với cụm phổ biến)
        AND token đơn  :  0,17 s

    Hai nhánh được đối xử KHÁC NHAU, vì chi phí của chúng khác hẳn nhau:
      - AND giữ NGUYÊN mọi token: bản thân phép AND đã đủ chọn lọc nên nhanh
        bất kể token có phổ biến hay không, và giữ đủ token thì BM25 mới bắt
        đúng được định danh ("168", "2024").
      - OR chỉ lấy các cụm có ít nhất một token hiếm. Độ dài posting list của
        một cụm bị chặn bởi token hiếm hơn, nên đây chính là chỗ cần cắt.
    `_fts` dừng sớm khi đã đủ kết quả nên nhánh OR thường không phải chạy.
    """
    toks = [t for t in _tokens(con, text) if t not in STOP_TOKENS] or _tokens(con, text)
    if not toks:
        return []
    queries = [" AND ".join(_quote(toks))]
    if len(toks) < 2:
        return queries

    # Chi phí của một cụm bị chặn bởi token HIẾM hơn trong cụm đó, nên xếp các
    # cụm theo df tăng dần rồi chỉ giữ MAX_OR_PHRASES cụm đầu: vừa rẻ nhất vừa
    # mang nhiều thông tin nhất. Không chặn số cụm thì một truy vấn toàn từ phổ
    # biến ("điều kiện thành lập công ty") ngốn tới 13,5 s.
    df = _doc_freq(con, toks)
    big = sorted(
        ((min(df.get(toks[i], 0), df.get(toks[i + 1], 0)), f'"{toks[i]} {toks[i + 1]}"')
         for i in range(len(toks) - 1)),
        key=lambda x: x[0],
    )
    phrases = [p for _, p in big[:MAX_OR_PHRASES]]
    if phrases:
        queries.append(" OR ".join(phrases))
    return queries


# --------------------------------------------------------------------------- BM25
def _fts(con: sqlite3.Connection, sql: str, match: str | list[str], limit: int) -> list[sqlite3.Row]:
    """Chạy một hoặc nhiều biểu thức MATCH, giữ thứ tự và bỏ trùng."""
    matches = [match] if isinstance(match, str) else match
    seen: set = set()
    out: list[sqlite3.Row] = []
    for m in matches:
        if not m:
            continue
        try:
            rows = con.execute(sql, (m, limit)).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("FTS bỏ qua biểu thức %r: %s", m[:60], e)
            continue
        for r in rows:
            key = r[0]
            if key not in seen:
                seen.add(key)
                out.append(r)
        if len(out) >= limit:
            break
    return out[:limit]


def bm25_chunks(con, q: str, limit: int) -> list[Hit]:
    sql = """
        SELECT c.chunk_id, c.dieu_no, c.dieu_title, c.chuong, c.text,
               COALESCE(d.title_vi, d.title) AS doc_title, d.url, d.doc_type,
               COALESCE(d.number_vi, d.doc_number) AS doc_number, d.year,
               d.expired, d.status, bm25(fts_chunks) AS rank
        FROM fts_chunks JOIN chunks c ON c.rowid_ = fts_chunks.rowid
        JOIN docs d ON d.id = c.doc_id
        WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?
    """
    out = []
    for r in _fts(con, sql, q, limit):
        label = f"Điều {r['dieu_no']}" if r["dieu_no"] else "Phần mở đầu"
        out.append(Hit("chunk", r["chunk_id"], r["doc_title"], label, r["text"],
                       r["url"], {"doc_type": r["doc_type"], "doc_number": r["doc_number"],
                                  "year": r["year"], "expired": r["expired"],
                                  "status": r["status"], "chuong": r["chuong"]},
                       sources=["bm25"]))
    return out


def bm25_qa(con, q: str, limit: int) -> list[Hit]:
    sql = """
        SELECT q.id, q.question, q.answer, q.url, q.area, q.published_date,
               bm25(fts_qa, 3.0, 1.0) AS rank
        FROM fts_qa JOIN qa q ON q.rowid = fts_qa.rowid
        WHERE fts_qa MATCH ? ORDER BY rank LIMIT ?
    """
    return [Hit("qa", r["id"], r["question"], "Hỏi đáp", r["answer"], r["url"],
                {"area": r["area"], "date": r["published_date"]}, sources=["bm25"])
            for r in _fts(con, sql, q, limit)]


def bm25_terms(con, q: str, limit: int) -> list[Hit]:
    sql = """
        SELECT t.term_id, t.term_vi, t.definition_vi, t.url, t.area,
               bm25(fts_terms, 5.0, 1.0) AS rank
        FROM fts_terms JOIN terms t ON t.term_id = fts_terms.rowid
        WHERE fts_terms MATCH ? ORDER BY rank LIMIT ?
    """
    return [Hit("term", str(r["term_id"]), r["term_vi"], "Thuật ngữ",
                r["definition_vi"] or "", r["url"], {"area": r["area"]}, sources=["bm25"])
            for r in _fts(con, sql, q, limit)]


def _strip_marks(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("đ", "d")


def bm25_docs(con, q, limit: int, *, query_text: str = "") -> list[Hit]:
    """Tra cứu theo tiêu đề/số hiệu — phủ cả 573.851 văn bản, kể cả văn bản chưa có toàn văn.

    Hai lớp ưu tiên chồng lên điểm bm25:

    1. Cấp hiệu lực (`tier`) — văn bản cấp cao xếp trước.
    2. Loại văn bản khớp câu hỏi — cần thiết vì `tier` KHÔNG tách được Luật với
       Nghị định (cả hai đều tier 0). Khi tra "Luật Doanh nghiệp", nghị định
       hướng dẫn chứa nguyên cụm đó trong tiêu đề sẽ thắng chính bộ luật, bởi
       tiêu đề trích ra của bộ luật chỉ còn "Doanh nghiệp" — chữ "LUẬT" là dòng
       loại văn bản nên nằm ở cột khác.

    bm25 của FTS5 là số ÂM, nên nhân với hệ số lớn hơn nghĩa là xếp trước.
    """
    sql = """
        SELECT d.id, COALESCE(d.title_vi, d.title) AS title, d.url, d.doc_type,
               COALESCE(d.number_vi, d.doc_number) AS doc_number, d.year, d.issuer,
               d.status, d.expired, d.has_text,
               bm25(fts_docs, 1.0, 8.0, 1.0, 6.0, 8.0, 4.0)
                 * (1.0 + 0.18 * (3 - COALESCE(d.tier, 3))) AS rank
        FROM fts_docs JOIN docs d ON d.rowid = fts_docs.rowid
        WHERE fts_docs MATCH ? ORDER BY rank LIMIT ?
    """
    rows = _fts(con, sql, q, limit * 4)

    qnorm = _strip_marks(query_text)
    # Nếu câu hỏi đã nêu năm thì tôn trọng năm đó, đừng ưu tiên bản mới hơn.
    asked_year = bool(re.search(r"\b(19|20)\d{2}\b", query_text or ""))
    scored = []
    for r in rows:
        boost = 1.0
        dtype = _strip_marks(r["doc_type"])
        if qnorm and dtype and dtype in qnorm:
            # nhỏ thôi: tiêu đề đã chứa sẵn loại văn bản (xem enrich.full_title)
            # nên bm25 tự lo phần lớn; đây chỉ là tín hiệu phụ.
            boost += 0.12
        if not asked_year and r["year"]:
            # "Luật Doanh nghiệp" không nêu năm -> người hỏi gần như chắc chắn
            # muốn bản đang áp dụng, không phải bản 2014 đã bị thay thế.
            boost += min(max(r["year"] - 1990, 0), 40) * 0.006
        if r["expired"]:
            boost -= 0.30
        # "Luật sửa đổi, bổ sung một số điều của Luật Doanh nghiệp" khớp tên
        # mạnh hơn chính bộ luật gốc (tiêu đề gốc chỉ còn "Doanh nghiệp"), nhưng
        # người hỏi tên luật hầu như luôn muốn văn bản gốc.
        tnorm = _strip_marks(r["title"])
        if tnorm.startswith(("sua doi", "bai bo", "dinh chinh", "huong dan")) \
                and "sua doi" not in qnorm:
            boost -= 0.28
        scored.append((r["rank"] * boost, r))
    scored.sort(key=lambda x: x[0])

    return [
        Hit("doc", r["id"], r["title"], r["doc_type"] or "Văn bản", "", r["url"],
            {"doc_number": r["doc_number"], "year": r["year"], "issuer": r["issuer"],
             "status": r["status"], "expired": r["expired"], "has_text": r["has_text"]},
            sources=["bm25"])
        for _, r in scored[:limit]
    ]


# --------------------------------------------------------------------------- vector
def vector_search(con: sqlite3.Connection, query: str, limit: int) -> list[Hit]:
    try:
        from .embed import open_table

        tbl = open_table(create=False)
    except Exception:
        return []  # chưa dựng chỉ mục vector -> chạy thuần BM25

    from . import models

    vec = models.embed_queries([query])[0]
    rows = (tbl.search(vec.tolist())
               .metric("cosine")
               .limit(limit)
               .select(["kind", "ref"])
               .to_list())

    by_kind: dict[str, list[str]] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["ref"])
        order.append((r["kind"], r["ref"]))

    found: dict[tuple[str, str], Hit] = {}
    if by_kind.get("chunk"):
        ph = ",".join("?" * len(by_kind["chunk"]))
        for r in con.execute(f"""
            SELECT c.chunk_id, c.dieu_no, c.chuong, c.text,
                   COALESCE(d.title_vi, d.title) AS doc_title, d.url, d.doc_type,
               COALESCE(d.number_vi, d.doc_number) AS doc_number, d.year,
                   d.expired, d.status
            FROM chunks c JOIN docs d ON d.id = c.doc_id
            WHERE c.chunk_id IN ({ph})""", by_kind["chunk"]):
            label = f"Điều {r['dieu_no']}" if r["dieu_no"] else "Phần mở đầu"
            found[("chunk", r["chunk_id"])] = Hit(
                "chunk", r["chunk_id"], r["doc_title"], label, r["text"], r["url"],
                {"doc_type": r["doc_type"], "doc_number": r["doc_number"],
                 "year": r["year"], "expired": r["expired"], "status": r["status"],
                 "chuong": r["chuong"]}, sources=["vector"])
    if by_kind.get("qa"):
        ph = ",".join("?" * len(by_kind["qa"]))
        for r in con.execute(
            f"SELECT id,question,answer,url,area,published_date FROM qa WHERE id IN ({ph})",
            by_kind["qa"],
        ):
            found[("qa", r["id"])] = Hit("qa", r["id"], r["question"], "Hỏi đáp",
                                         r["answer"], r["url"],
                                         {"area": r["area"], "date": r["published_date"]},
                                         sources=["vector"])
    if by_kind.get("term"):
        ph = ",".join("?" * len(by_kind["term"]))
        for r in con.execute(
            f"SELECT term_id,term_vi,definition_vi,url,area FROM terms WHERE term_id IN ({ph})",
            by_kind["term"],
        ):
            found[("term", str(r["term_id"]))] = Hit(
                "term", str(r["term_id"]), r["term_vi"], "Thuật ngữ",
                r["definition_vi"] or "", r["url"], {"area": r["area"]}, sources=["vector"])

    return [found[k] for k in order if k in found]


# --------------------------------------------------------------------------- fusion
def rrf(lists: list[list[Hit]], k: int = config.RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: gộp nhiều bảng xếp hạng chỉ dựa trên THỨ HẠNG.

    Điểm BM25 và điểm cosine không cùng thang, chuẩn hoá lại luôn phải chỉnh tay
    theo từng truy vấn. RRF chỉ dùng thứ hạng nên miễn nhiễm với vấn đề đó.
    """
    pool: dict[tuple[str, str], Hit] = {}
    scores: dict[tuple[str, str], float] = {}
    for lst in lists:
        for rank, hit in enumerate(lst):
            key = (hit.kind, hit.ref)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key in pool:
                for s in hit.sources:
                    if s not in pool[key].sources:
                        pool[key].sources.append(s)
            else:
                pool[key] = hit
    for key, sc in scores.items():
        pool[key].score = sc
    return sorted(pool.values(), key=lambda h: h.score, reverse=True)


def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Chấm lại bằng cross-encoder tiếng Việt — đắt nhưng chỉ chạy trên ~50 ứng viên."""
    if not hits:
        return []
    from . import models

    try:
        ce = models.get_reranker()
        # 2000 ký tự ~ 512 token/cặp khiến rerank mất 5,2 s cho 50 cặp.
        # 800 ký tự vẫn đủ ngữ cảnh để xếp hạng mà nhanh hơn nhiều lần.
        pairs = [(query, f"{h.citation}\n{h.text}"[:800]) for h in hits]
        scores = ce.predict(pairs, batch_size=32)
    except Exception as e:
        # Trước đây lỗi ở đây bị nuốt im lặng và hệ thống âm thầm chạy không có
        # reranker — nhìn kết quả rất khó nhận ra. Giờ báo rõ rồi mới rơi về RRF.
        logger.warning("Reranker không dùng được (%s) — dùng thứ hạng RRF", e)
        return hits[:top_k]
    for h, s in zip(hits, scores):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]


# --------------------------------------------------------------------------- API
# Thứ tự ưu tiên khi nhiều văn bản cùng chứa một quy định giống hệt nhau.
_TYPE_RANK = {
    "Hiến pháp": 0, "Bộ luật": 1, "Luật": 1, "Pháp lệnh": 2,
    "Nghị định": 3, "Nghị quyết": 4, "Thông tư": 5,
    "Văn bản hợp nhất": 6, "Quyết định": 7,
}


def _authority(h: Hit) -> tuple:
    """Khoá sắp xếp: bản nào đáng trích dẫn hơn thì nhỏ hơn."""
    m = h.meta
    return (
        1 if m.get("expired") else 0,
        _TYPE_RANK.get(m.get("doc_type") or "", 8),
        -(m.get("year") or 0),
    )


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """Gộp các đoạn trùng nội dung, giữ lại BẢN ĐÁNG TRÍCH DẪN NHẤT.

    Corpus có nhiều bản sao của cùng một quy định: văn bản hợp nhất, bản dịch
    tiếng Anh, văn bản đăng lại. Nếu chỉ "giữ bản gặp trước" thì Điều 111 Luật
    Doanh nghiệp 2020 có thể bị thay bằng bản hợp nhất mang tiêu đề slug không
    dấu — cùng nội dung nhưng trích dẫn kém hẳn. Ở đây giữ nguyên VỊ TRÍ của
    lần gặp đầu (tức thứ hạng đã tính) nhưng thay nội dung bằng bản tốt nhất.
    """
    order: list[str] = []
    best: dict[str, Hit] = {}
    loose: list[Hit] = []
    for h in hits:
        key = re.sub(r"[^0-9a-zà-ỹ]+", "", h.text[:220].lower())[:160]
        if not key:
            loose.append(h)
            continue
        if key not in best:
            best[key] = h
            order.append(key)
        elif _authority(h) < _authority(best[key]):
            # giữ điểm của bản đứng trước để không xáo trộn thứ hạng
            h.score = best[key].score
            h.sources = sorted(set(best[key].sources) | set(h.sources))
            best[key] = h
    out = [best[k] for k in order]
    return out + loose


def retrieve(
    con: sqlite3.Connection,
    query: str,
    *,
    top_k: int = config.CONTEXT_TOPK,
    use_rerank: bool = True,
    use_vector: bool = True,
) -> list[Hit]:
    # Bảng lớn (chunks/qa/docs) dùng biểu thức phân tầng; bảng terms nhỏ nên
    # OR đơn giản đã đủ nhanh và cho recall tốt hơn.
    multi = fts_queries(con, query)
    simple = fts_query(con, query)

    # 4 truy vấn BM25 + 1 truy vấn vector độc lập nhau -> chạy song song.
    # Mỗi tác vụ lấy kết nối riêng của luồng mình (xem db.thread_conn), vì dùng
    # chung một kết nối thì SQLite tuần tự hoá và song song thành vô nghĩa.
    tasks = [
        lambda: bm25_chunks(db.thread_conn(), multi, config.BM25_TOPK),
        lambda: bm25_qa(db.thread_conn(), multi, config.BM25_TOPK),
        lambda: bm25_terms(db.thread_conn(), simple, 15),
        lambda: bm25_docs(db.thread_conn(), multi, 15, query_text=query),
    ]
    if use_vector:
        tasks.append(lambda: vector_search(db.thread_conn(), query, config.VECTOR_TOPK))

    lists: list[list[Hit]] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for fut in [pool.submit(t) for t in tasks]:
            try:
                lists.append(fut.result())
            except Exception as e:
                logger.warning("một nguồn truy xuất lỗi, bỏ qua: %s", e)

    fused = _dedupe(rrf(lists))
    # Văn bản hết hiệu lực vẫn giữ (có thể cần tra lịch sử) nhưng đẩy xuống
    # để không lấn chỗ quy định đang áp dụng.
    fused.sort(key=lambda h: (h.meta.get("expired", 0) == 1, -h.score))

    if not use_rerank:
        return fused[:top_k]
    return _dedupe(rerank(query, fused[: config.RERANK_CANDIDATES], top_k))


def get_document(con: sqlite3.Connection, doc_id: str) -> dict | None:
    """Dựng lại toàn văn một văn bản bằng cách nối các chunk theo thứ tự."""
    d = con.execute("SELECT * FROM docs WHERE id = ?", (doc_id,)).fetchone()
    if not d:
        return None
    rows = con.execute(
        "SELECT chunk_id,dieu_no,dieu_title,chuong,text,part,n_parts "
        "FROM chunks WHERE doc_id = ? ORDER BY seq", (doc_id,)).fetchall()
    return {
        "meta": dict(d),
        "articles": [dict(r) for r in rows],
        "has_text": bool(rows),
    }
