"""Nạp parquet -> SQLite, tách Điều, rồi dựng chỉ mục FTS5.

Chạy được nhiều lần: mỗi shard parquet đã nạp xong được ghi vào `build_state`,
lần chạy sau sẽ bỏ qua. Cần thiết vì vbpl có 96 shard / 2.96 GB, không thể
nạp hết vào RAM 16 GB một lúc.

    uv run python -m luatvn.ingest            # nạp tất cả
    uv run python -m luatvn.ingest --reset    # xoá CSDL và làm lại
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import polars as pl

from . import config, db
from .chunk import split_document

# Chỉ chunk toàn văn tới tier này (tier 2 = quyết định cấp tỉnh, ~419K văn bản,
# giá trị tra cứu pháp lý phổ thông thấp nhưng chiếm 8.4 GB text).
CHUNK_MAX_TIER = int(__import__("os").getenv("LUATVN_CHUNK_MAX_TIER", "1"))

DOC_COLS = [
    "id", "doc_type", "doc_number", "year", "issuer", "tier", "legal_area",
    "title", "url", "issue_date", "effective_date", "status", "signer",
    "num_articles", "vn_chars",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- vbpl
def ingest_vbpl(con: sqlite3.Connection) -> None:
    files = sorted(config.VBPL_DIR.glob("documents-*.parquet"))
    if not files:
        _log("!! không tìm thấy parquet vbpl — chạy `python -m luatvn.download` trước")
        return
    done = set(json.loads(db.get_state(con, "vbpl_shards", "[]")))
    _log(f"vbpl: {len(files)} shard, đã xong {len(done)}")

    for f in files:
        if f.name in done:
            continue
        t0 = time.time()
        df = pl.read_parquet(f, columns=DOC_COLS + ["vn_text"])

        docs_rows, chunk_rows = [], []
        for r in df.iter_rows(named=True):
            status = r["status"] or ""
            expired = 1 if status.startswith("Hết hiệu lực") else 0
            text = r["vn_text"]
            has_text = 0

            if text and r["tier"] is not None and r["tier"] <= CHUNK_MAX_TIER:
                cs = split_document(r["id"], text)
                if cs:
                    has_text = 1
                    for seq, c in enumerate(cs):
                        chunk_rows.append(
                            (c.chunk_id, c.doc_id, seq, c.book, c.chuong,
                             c.dieu_no, c.dieu_title, c.part, c.n_parts, c.text)
                        )
            docs_rows.append(tuple(r[k] for k in DOC_COLS) + (expired, has_text))

        con.executemany(
            f"INSERT OR REPLACE INTO docs({','.join(DOC_COLS)},expired,has_text) "
            f"VALUES({','.join('?' * (len(DOC_COLS) + 2))})",
            docs_rows,
        )
        con.executemany(
            "INSERT OR IGNORE INTO chunks"
            "(chunk_id,doc_id,seq,book,chuong,dieu_no,dieu_title,part,n_parts,text)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            chunk_rows,
        )
        done.add(f.name)
        db.set_state(con, "vbpl_shards", json.dumps(sorted(done)))
        con.commit()
        _log(f"  {f.name}: {len(docs_rows):,} vb, {len(chunk_rows):,} chunk "
             f"({time.time() - t0:.1f}s) [{len(done)}/{len(files)}]")


# --------------------------------------------------------------------------- hdpl
def ingest_hdpl(con: sqlite3.Connection) -> None:
    files = sorted(config.HDPL_DIR.glob("documents-*.parquet"))
    if not files:
        _log("!! không tìm thấy parquet hdpl")
        return
    done = set(json.loads(db.get_state(con, "hdpl_shards", "[]")))
    for f in files:
        if f.name in done:
            continue
        df = pl.read_parquet(f, columns=[
            "id", "url", "question", "answer", "area", "category",
            "published_date", "summary", "keywords", "num_citations", "answer_chars",
        ])
        rows = [
            (r["id"], r["url"], r["question"], r["answer"], r["area"], r["category"],
             r["published_date"], r["summary"],
             ", ".join(r["keywords"] or []), r["num_citations"], r["answer_chars"])
            for r in df.iter_rows(named=True)
        ]
        con.executemany(
            "INSERT OR REPLACE INTO qa(id,url,question,answer,area,category,"
            "published_date,summary,keywords,num_citations,answer_chars)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
        done.add(f.name)
        db.set_state(con, "hdpl_shards", json.dumps(sorted(done)))
        con.commit()
        _log(f"  hdpl {f.name}: {len(rows):,} hỏi đáp")


# --------------------------------------------------------------------------- tnpl
def ingest_tnpl(con: sqlite3.Connection) -> None:
    f = config.TNPL_DIR / "documents.parquet"
    if not f.exists():
        _log("!! không tìm thấy parquet tnpl")
        return
    df = pl.read_parquet(f, columns=[
        "term_id", "term_name_vi", "term_name_en", "definition_vi",
        "area_name_vi", "status_vi", "source_url", "related_term_names_vi",
    ])
    rows = [
        (r["term_id"], r["term_name_vi"], r["term_name_en"], r["definition_vi"],
         r["area_name_vi"], r["status_vi"], r["source_url"],
         ", ".join(r["related_term_names_vi"] or []))
        for r in df.iter_rows(named=True)
    ]
    con.executemany(
        "INSERT OR REPLACE INTO terms(term_id,term_vi,term_en,definition_vi,"
        "area,status,url,related) VALUES(?,?,?,?,?,?,?,?)", rows)
    con.commit()
    _log(f"  tnpl: {len(rows):,} thuật ngữ")


# --------------------------------------------------------------------------- FTS
def build_fts(con: sqlite3.Connection) -> None:
    """Dựng chỉ mục FTS5 từ các bảng nội dung.

    Dùng lệnh 'rebuild' của FTS5 thay vì DELETE + INSERT...SELECT: với bảng
    external-content, `DELETE FROM fts_x` là thao tác không hợp lệ và SQLite
    báo lại thành "database disk image is malformed" dù CSDL vẫn nguyên vẹn.
    'rebuild' đọc thẳng bảng nội dung nên vừa đúng vừa nhanh hơn.
    """
    for name in ("fts_chunks", "fts_docs", "fts_qa", "fts_terms"):
        t0 = time.time()
        con.execute(f"INSERT INTO {name}({name}) VALUES('rebuild')")
        con.commit()
        n = con.execute(f"SELECT count(*) c FROM {name}").fetchone()["c"]
        _log(f"  {name}: {n:,} dòng ({time.time() - t0:.1f}s)")
    _log("  tối ưu chỉ mục FTS...")
    for name in ("fts_chunks", "fts_docs", "fts_qa", "fts_terms"):
        con.execute(f"INSERT INTO {name}({name}) VALUES('optimize')")
    con.commit()


def build_token_df(con: sqlite3.Connection) -> None:
    """Ghi lại tần suất của các token phổ biến trong bảng chunks.

    FTS5 phải chấm điểm bm25 cho MỌI dòng khớp, nên một token như "điều"
    (tokenizer trả về `đieu`, có mặt ở 97,4% số đoạn) làm truy vấn mất hàng chục
    giây. Bảng này cho phép loại các token như vậy khỏi truy vấn.
    """
    t0 = time.time()
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS v_chunks USING fts5vocab('fts_chunks','row')")
    con.execute("DELETE FROM token_df")
    con.execute("INSERT INTO token_df(term, df) SELECT term, doc FROM v_chunks WHERE doc > 2000")
    con.commit()
    n = con.execute("SELECT count(*) FROM token_df").fetchone()["c" if False else 0]
    _log(f"  token_df: {n:,} token phổ biến ({time.time() - t0:.1f}s)")


def stats(con: sqlite3.Connection) -> None:
    q = lambda s: con.execute(s).fetchone()[0]
    _log("=== THỐNG KÊ ===")
    _log(f"  văn bản pháp luật : {q('SELECT count(*) FROM docs'):,}")
    _log(f"    có toàn văn      : {q('SELECT count(*) FROM docs WHERE has_text=1'):,}")
    _log(f"    đã hết hiệu lực  : {q('SELECT count(*) FROM docs WHERE expired=1'):,}")
    _log(f"  đoạn (Điều)        : {q('SELECT count(*) FROM chunks'):,}")
    _log(f"  hỏi đáp            : {q('SELECT count(*) FROM qa'):,}")
    _log(f"  thuật ngữ          : {q('SELECT count(*) FROM terms'):,}")
    size = config.DB_PATH.stat().st_size / 1e9 if config.DB_PATH.exists() else 0
    _log(f"  kích thước CSDL    : {size:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="xoá CSDL và nạp lại từ đầu")
    ap.add_argument("--skip-fts", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.reset and config.DB_PATH.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(config.DB_PATH) + suffix).unlink(missing_ok=True)
        _log("đã xoá CSDL cũ")

    con = db.connect()
    db.init(con)
    # cache lớn giúp phần chèn hàng triệu dòng nhanh hơn đáng kể
    con.execute("PRAGMA cache_size = -400000")

    _log("--- nạp tnpl (thuật ngữ) ---");   ingest_tnpl(con)
    _log("--- nạp hdpl (hỏi đáp) ---");     ingest_hdpl(con)
    _log("--- nạp vbpl (văn bản) ---");     ingest_vbpl(con)
    if not args.skip_fts:
        _log("--- dựng chỉ mục FTS5 ---");  build_fts(con)
        _log("--- thống kê tần suất token ---"); build_token_df(con)
    stats(con)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
