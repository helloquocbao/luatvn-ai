"""Đo chất lượng truy xuất trên một bộ câu hỏi có đáp án biết trước.

Mỗi câu hỏi gắn với một Điều cụ thể của một văn bản cụ thể. Ta đo:
  - Recall@k : đáp án có nằm trong k kết quả đầu không
  - MRR      : nghịch đảo thứ hạng của đáp án đầu tiên (1.0 = luôn đứng đầu)
Chạy cả khi bật và tắt từng thành phần để thấy mỗi phần đóng góp bao nhiêu.

    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --ablate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luatvn import db, retrieve  # noqa: E402

GOLD = json.loads((Path(__file__).parent / "gold.json").read_text(encoding="utf-8"))


def _dieu_match(h, expect: str) -> bool:
    return h.label.replace("Điều ", "").split(" ")[0] == expect


def evaluate(con, *, use_vector: bool, use_rerank: bool, k: int = 10, verbose: bool = False) -> dict:
    hit_at = {1: 0, 3: 0, 5: 0, 10: 0}
    rr_total, elapsed = 0.0, 0.0
    for g in GOLD:
        t0 = time.time()
        hits = retrieve.retrieve(con, g["q"], top_k=k, use_vector=use_vector, use_rerank=use_rerank)
        elapsed += time.time() - t0
        rank = None
        for i, h in enumerate(hits, 1):
            if h.kind == "chunk" and (h.meta.get("doc_number") or "") == g["expect_doc"] \
                    and _dieu_match(h, g["expect_dieu"]):
                rank = i
                break
        if rank:
            rr_total += 1 / rank
            for kk in hit_at:
                if rank <= kk:
                    hit_at[kk] += 1
        if verbose:
            mark = f"#{rank}" if rank else "TRƯỢT"
            print(f"   {mark:>6}  {g['q'][:62]}")
            if not rank and hits:
                top = hits[0]
                print(f"           thay vào đó: {top.label} · "
                      f"{top.meta.get('doc_number') or top.kind} | {top.title[:44]}")
    n = len(GOLD)
    return {
        "recall@1": hit_at[1] / n, "recall@3": hit_at[3] / n,
        "recall@5": hit_at[5] / n, "recall@10": hit_at[10] / n,
        "mrr": rr_total / n, "giây/truy vấn": elapsed / n,
    }


def show(name: str, m: dict) -> None:
    print(f"  {name:28} R@1={m['recall@1']:.2f}  R@3={m['recall@3']:.2f}  "
          f"R@5={m['recall@5']:.2f}  R@10={m['recall@10']:.2f}  "
          f"MRR={m['mrr']:.3f}  {m['giây/truy vấn']:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true", help="đo đóng góp của từng thành phần")
    args = ap.parse_args()

    con = db.connect(readonly=True)
    print(f"Bộ đánh giá: {len(GOLD)} câu hỏi có đáp án Điều cụ thể\n")

    print("Đầy đủ (BM25 + vector + rerank):")
    full = evaluate(con, use_vector=True, use_rerank=True, verbose=True)
    print()
    show("đầy đủ", full)

    if args.ablate:
        show("bỏ rerank", evaluate(con, use_vector=True, use_rerank=False))
        show("bỏ vector (chỉ BM25)", evaluate(con, use_vector=False, use_rerank=True))
        show("chỉ BM25, không rerank", evaluate(con, use_vector=False, use_rerank=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
