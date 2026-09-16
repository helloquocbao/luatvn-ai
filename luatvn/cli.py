"""Giao diện dòng lệnh để kiểm thử nhanh chất lượng truy xuất và trả lời.

    uv run python -m luatvn.cli "điều kiện thành lập công ty cổ phần"
    uv run python -m luatvn.cli --search "nồng độ cồn xe máy"   # không gọi LLM
    uv run python -m luatvn.cli                                  # chế độ hỏi liên tục
"""

from __future__ import annotations

import argparse
import sys
import time

from . import db, llm, rag, retrieve


def show_sources(hits) -> None:
    for i, h in enumerate(hits, 1):
        m = h.meta
        head = " ".join(str(x) for x in (m.get("doc_type"), m.get("doc_number")) if x)
        flag = "  ⚠️ HẾT HIỆU LỰC" if m.get("expired") else ""
        print(f"\n  [{i}] {h.label} · {head}{flag}")
        print(f"      {h.title[:95]}")
        print(f"      điểm {h.score:.2f} · {'+'.join(h.sources)}")
        body = " ".join(h.text.split())
        print(f"      {body[:200]}…")


def run_once(con, question: str, *, search_only: bool) -> None:
    t0 = time.time()
    hits = retrieve.retrieve(con, question)
    print(f"\n── {len(hits)} nguồn ({time.time() - t0:.2f}s) " + "─" * 40)
    show_sources(hits)
    if search_only:
        return
    if not llm.available():
        print("\n⚠️  Ollama chưa chạy — bỏ qua phần sinh câu trả lời.")
        print("   Khởi động bằng: ollama serve")
        return
    msgs, used = rag.build_messages(question, hits)
    print("\n── Trả lời " + "─" * 46 + "\n")
    for piece in llm.chat_stream(msgs):
        sys.stdout.write(piece)
        sys.stdout.flush()
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Trợ lý pháp luật Việt Nam (local)")
    ap.add_argument("question", nargs="*", help="câu hỏi; bỏ trống để vào chế độ tương tác")
    ap.add_argument("--search", action="store_true", help="chỉ tra cứu, không gọi LLM")
    args = ap.parse_args()

    con = db.connect(readonly=True)
    if args.question:
        run_once(con, " ".join(args.question), search_only=args.search)
        return 0

    print("Trợ lý pháp luật Việt Nam — gõ câu hỏi, Ctrl-C để thoát.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q:
            run_once(con, q, search_only=args.search)


if __name__ == "__main__":
    sys.exit(main())
