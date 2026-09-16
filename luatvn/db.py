"""Lược đồ SQLite và các tiện ích truy cập.

Thiết kế lưu trữ (ràng buộc: ổ cứng còn ~51 GB):
  - `docs` giữ metadata của TOÀN BỘ 573.851 văn bản -> luôn tra cứu được
    số hiệu / cơ quan ban hành / ngày hiệu lực, kể cả văn bản không có toàn văn.
  - `chunks` chỉ giữ toàn văn (tách theo Điều) của văn bản tier <= CHUNK_MAX_TIER.
    Toàn văn KHÔNG lưu lặp: bản đầy đủ được dựng lại bằng cách nối các chunk.
  - FTS5 đánh chỉ mục tiêu đề của mọi văn bản + toàn văn của phần đã chunk.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from . import config

# FTS5 dùng unicode61 + remove_diacritics 2: người dùng gõ "co phan" vẫn khớp
# "cổ phần". Tiếng Việt không dấu là kiểu gõ rất phổ biến, nên đây là lựa chọn
# thiên về recall; độ chính xác do BM25 + reranker xử lý ở lớp trên.
TOKENIZER = "unicode61 remove_diacritics 2"

SCHEMA = f"""
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Metadata toàn bộ văn bản pháp luật (vbpl)
CREATE TABLE IF NOT EXISTS docs (
    id            TEXT PRIMARY KEY,
    doc_type      TEXT,
    doc_number    TEXT,
    year          INTEGER,
    issuer        TEXT,
    tier          INTEGER,
    legal_area    TEXT,
    title         TEXT,
    url           TEXT,
    issue_date    TEXT,
    effective_date TEXT,
    status        TEXT,
    expired       INTEGER DEFAULT 0,  -- 1 nếu status ghi rõ "Hết hiệu lực"
    signer        TEXT,
    num_articles  INTEGER,
    vn_chars      INTEGER,
    has_text      INTEGER DEFAULT 0   -- 1 nếu đã chunk toàn văn vào bảng chunks
);
CREATE INDEX IF NOT EXISTS ix_docs_tier   ON docs(tier);
CREATE INDEX IF NOT EXISTS ix_docs_type   ON docs(doc_type);
CREATE INDEX IF NOT EXISTS ix_docs_year   ON docs(year);
CREATE INDEX IF NOT EXISTS ix_docs_number ON docs(doc_number);

-- Đoạn văn bản theo Điều
CREATE TABLE IF NOT EXISTS chunks (
    rowid_      INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT UNIQUE,
    doc_id      TEXT,
    seq         INTEGER,   -- thứ tự trong văn bản, dùng để dựng lại toàn văn
    book        INTEGER,
    chuong      TEXT,
    dieu_no     TEXT,
    dieu_title  TEXT,
    part        INTEGER,
    n_parts     INTEGER,
    text        TEXT
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id, seq);

-- Hỏi đáp pháp luật (hdpl)
CREATE TABLE IF NOT EXISTS qa (
    id            TEXT PRIMARY KEY,
    url           TEXT,
    question      TEXT,
    answer        TEXT,
    area          TEXT,
    category      TEXT,
    published_date TEXT,
    summary       TEXT,
    keywords      TEXT,
    num_citations INTEGER,
    answer_chars  INTEGER
);

-- Thuật ngữ pháp lý (tnpl)
CREATE TABLE IF NOT EXISTS terms (
    term_id       INTEGER PRIMARY KEY,
    term_vi       TEXT,
    term_en       TEXT,
    definition_vi TEXT,
    area          TEXT,
    status        TEXT,
    url           TEXT,
    related       TEXT
);

-- Tần suất token trong bảng chunks, dùng để cắt bớt token quá phổ biến
-- khỏi truy vấn BM25 (xem luatvn/retrieve.py).
CREATE TABLE IF NOT EXISTS token_df (
    term TEXT PRIMARY KEY,
    df   INTEGER
);

-- Bảng đánh dấu tiến độ, cho phép chạy lại/nối tiếp job dài
CREATE TABLE IF NOT EXISTS build_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ===== FTS5 (external content: không nhân đôi text, vẫn dùng được snippet()) =====
-- LƯU Ý: external-content FTS5 neo theo rowid của bảng gốc -> KHÔNG chạy VACUUM
-- trên CSDL này, vì VACUUM có thể đánh số lại rowid và làm lệch chỉ mục.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    text, dieu_title,
    content = 'chunks', content_rowid = 'rowid_',
    tokenize = '{TOKENIZER}'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(
    title, doc_number, issuer,
    content = 'docs',
    tokenize = '{TOKENIZER}'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_qa USING fts5(
    question, answer,
    content = 'qa',
    tokenize = '{TOKENIZER}'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_terms USING fts5(
    term_vi, definition_vi,
    content = 'terms', content_rowid = 'term_id',
    tokenize = '{TOKENIZER}'
);
"""


def connect(path: Path | None = None, *, readonly: bool = False) -> sqlite3.Connection:
    p = path or config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    else:
        con = sqlite3.connect(p, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 30000")
    return con


def init(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def get_state(con: sqlite3.Connection, key: str, default: str = "") -> str:
    row = con.execute("SELECT value FROM build_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO build_state(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    con.commit()


# Một kết nối SQLite tự tuần tự hoá các câu lệnh, nên chạy 5 nguồn truy xuất
# "song song" trên cùng một kết nối thì không nhanh hơn chút nào. Mỗi luồng
# giữ kết nối chỉ-đọc riêng; chỉ-đọc nên không có rủi ro tranh ghi.
_local = threading.local()


def thread_conn(path: Path | None = None) -> sqlite3.Connection:
    """Kết nối chỉ-đọc riêng cho luồng hiện tại (tạo một lần rồi tái dùng)."""
    con = getattr(_local, "con", None)
    if con is None:
        con = connect(path, readonly=True)
        _local.con = con
    return con
