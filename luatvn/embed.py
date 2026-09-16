"""Dựng chỉ mục vector (LanceDB) cho chunk luật, hỏi đáp và thuật ngữ.

Job này chạy hàng giờ nên được thiết kế để DỪNG GIỮA CHỪNG VÀ CHẠY TIẾP:
tiến độ của từng nguồn được ghi vào bảng `build_state` sau mỗi lô, nên tắt máy
hay Ctrl-C không mất công đã làm.

    uv run python -m luatvn.embed                  # dựng tất cả
    uv run python -m luatvn.embed --only qa,term   # chỉ phần nhanh (~10 phút)
    uv run python -m luatvn.embed --index          # chỉ tạo chỉ mục ANN
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import numpy as np
import pyarrow as pa

from . import config, db, models

TABLE = "vectors"
BATCH_ROWS = 2000  # số dòng đọc từ SQLite mỗi vòng


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("vid", pa.string()),
        pa.field("kind", pa.string()),       # 'chunk' | 'qa' | 'term'
        pa.field("ref", pa.string()),        # chunk_id / qa.id / term_id
        pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
    ])


def open_table(create: bool = True):
    import lancedb

    dbx = lancedb.connect(str(config.INDEX_DIR / "lance"))
    if TABLE in dbx.table_names():
        return dbx.open_table(TABLE)
    if not create:
        raise FileNotFoundError(
            f"Chưa có chỉ mục vector tại {config.INDEX_DIR / 'lance'} — chạy `python -m luatvn.embed`"
        )
    return dbx.create_table(TABLE, schema=_schema())


# --------------------------------------------------------------------------- nguồn
def _chunk_query() -> tuple[str, list]:
    """Chunk luật cần nhúng, kèm tiêu đề văn bản để đoạn tự đứng được về ngữ cảnh."""
    where = ["d.tier <= ?"]
    params: list = [config.VECTOR_MAX_TIER]
    if config.EXCLUDE_EXPIRED:
        where.append("d.expired = 0")
    sql = (
        "SELECT c.rowid_ AS rid, c.chunk_id AS ref, "
        "       COALESCE(d.title_vi, d.title) AS doc_title, "
        "       c.chuong, c.dieu_title, c.text "
        "FROM chunks c JOIN docs d ON d.id = c.doc_id "
        f"WHERE {' AND '.join(where)} AND c.rowid_ > ? "
        "ORDER BY c.rowid_ LIMIT ?"
    )
    return sql, params


def _iter_chunks(con: sqlite3.Connection, after: int):
    sql, params = _chunk_query()
    while True:
        rows = con.execute(sql, (*params, after, BATCH_ROWS)).fetchall()
        if not rows:
            return
        payload = []
        for r in rows:
            head = " — ".join(x for x in (r["doc_title"], r["chuong"]) if x)
            payload.append((str(r["rid"]), r["ref"], f"{head}\n{r['text']}"[:4000]))
            after = r["rid"]
        yield after, payload


def _iter_qa(con: sqlite3.Connection, after: int):
    sql = ("SELECT rowid AS rid, id AS ref, question, summary, answer FROM qa "
           "WHERE rowid > ? ORDER BY rowid LIMIT ?")
    while True:
        rows = con.execute(sql, (after, BATCH_ROWS)).fetchall()
        if not rows:
            return
        payload = []
        for r in rows:
            # Nhúng câu hỏi + phần đầu câu trả lời: câu hỏi mang ý định người dùng,
            # phần đầu trả lời mang thuật ngữ pháp lý để bắt được truy vấn theo khái niệm.
            body = (r["summary"] or r["answer"] or "")[:1200]
            payload.append((str(r["rid"]), r["ref"], f"{r['question']}\n{body}"))
            after = r["rid"]
        yield after, payload


def _iter_terms(con: sqlite3.Connection, after: int):
    sql = ("SELECT term_id AS rid, term_id AS ref, term_vi, definition_vi FROM terms "
           "WHERE term_id > ? ORDER BY term_id LIMIT ?")
    while True:
        rows = con.execute(sql, (after, BATCH_ROWS)).fetchall()
        if not rows:
            return
        payload = []
        for r in rows:
            payload.append((str(r["rid"]), str(r["ref"]),
                            f"{r['term_vi']}: {r['definition_vi'] or ''}"[:2000]))
            after = r["rid"]
        yield after, payload


SOURCES = {"chunk": _iter_chunks, "qa": _iter_qa, "term": _iter_terms}


def _total(con: sqlite3.Connection, kind: str) -> int:
    if kind == "chunk":
        where = "d.tier <= ?" + (" AND d.expired = 0" if config.EXCLUDE_EXPIRED else "")
        return con.execute(
            f"SELECT count(*) FROM chunks c JOIN docs d ON d.id=c.doc_id WHERE {where}",
            (config.VECTOR_MAX_TIER,),
        ).fetchone()[0]
    return con.execute(f"SELECT count(*) FROM {'qa' if kind == 'qa' else 'terms'}").fetchone()[0]


def build(kinds: list[str]) -> None:
    con = db.connect(readonly=True)
    tbl = open_table()

    for kind in kinds:
        state_key = f"embed_cursor_{kind}_{config.EMBED_MODEL}"
        con_w = db.connect()
        after = int(db.get_state(con_w, state_key, "0") or 0)
        total = _total(con, kind)
        if after:
            _log(f"[{kind}] chạy tiếp từ con trỏ {after:,}")
        _log(f"[{kind}] tổng {total:,} mục cần nhúng")

        n_done, t_start, last_report = 0, time.time(), time.time()
        for cursor, payload in SOURCES[kind](con, after):
            vecs = models.embed_passages([p[2] for p in payload])
            vecs = np.asarray(vecs, dtype=np.float32)
            tbl.add(pa.table(
                {
                    "vid": pa.array([f"{kind}:{p[0]}" for p in payload]),
                    "kind": pa.array([kind] * len(payload)),
                    "ref": pa.array([p[1] for p in payload]),
                    "vector": pa.FixedSizeListArray.from_arrays(
                        pa.array(vecs.reshape(-1), type=pa.float32()), config.EMBED_DIM),
                },
                schema=_schema(),
            ))
            db.set_state(con_w, state_key, str(cursor))
            n_done += len(payload)
            if time.time() - last_report > 30:
                rate = n_done / (time.time() - t_start)
                left = max(total - n_done, 0) / rate / 3600 if rate else 0
                _log(f"[{kind}] {n_done:,} mục | {rate:.0f}/s | còn ~{left:.1f} giờ")
                last_report = time.time()
        _log(f"[{kind}] xong {n_done:,} mục trong {(time.time() - t_start) / 60:.1f} phút")
        con_w.close()
    con.close()


def build_ann_index() -> None:
    """Tạo chỉ mục IVF_PQ. Không có nó, mỗi truy vấn phải quét toàn bộ vector."""
    tbl = open_table(create=False)
    n = tbl.count_rows()
    if n < 10_000:
        _log(f"chỉ {n:,} vector — quét tuần tự đã đủ nhanh, bỏ qua chỉ mục ANN")
        return
    # quy tắc thông dụng: số phân vùng ~ sqrt(N)
    parts = max(64, min(4096, int(n ** 0.5)))
    _log(f"tạo chỉ mục IVF_PQ trên {n:,} vector (num_partitions={parts})...")
    t0 = time.time()
    tbl.create_index(
        metric="cosine",
        num_partitions=parts,
        num_sub_vectors=config.EMBED_DIM // 8,
        replace=True,
    )
    _log(f"xong sau {(time.time() - t0) / 60:.1f} phút")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="term,qa,chunk",
                    help="danh sách nguồn, cách nhau bởi dấu phẩy: term,qa,chunk")
    ap.add_argument("--index", action="store_true", help="chỉ tạo chỉ mục ANN rồi thoát")
    ap.add_argument("--no-index", action="store_true", help="bỏ qua bước tạo chỉ mục ANN")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.index:
        build_ann_index()
        return 0

    kinds = [k.strip() for k in args.only.split(",") if k.strip() in SOURCES]
    _log(f"model={config.EMBED_MODEL} dim={config.EMBED_DIM} nguồn={kinds}")
    build(kinds)
    if not args.no_index:
        build_ann_index()
    return 0


if __name__ == "__main__":
    sys.exit(main())
