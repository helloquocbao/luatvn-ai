"""Cấu hình tập trung cho toàn bộ pipeline.

Các con số ở đây đến từ đo đạc thực tế trên máy M4/16GB, không phải ước lượng:
  - AITeamVN/Vietnamese_Embedding (bge-m3, 568M, 1024-d):  20.6 chunk/s -> 13.5h/1M
  - intfloat/multilingual-e5-base (278M, 768-d):           67.9 chunk/s ->  4.1h/1M
  - intfloat/multilingual-e5-small (118M, 384-d):         217.1 chunk/s ->  1.3h/1M
Vì vậy vòng 1 dùng bi-encoder nhanh (e5-small), vòng 2 dùng cross-encoder
tiếng Việt mạnh (Vietnamese_Reranker) chỉ trên ~50 ứng viên mỗi truy vấn.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = DATA / "db" / "luatvn.sqlite"
INDEX_DIR = DATA / "index"

VBPL_DIR = RAW / "thuvienphapluat-vn-vbpl"
HDPL_DIR = RAW / "thuvienphapluat-vn-hdpl"
TNPL_DIR = RAW / "thuvienphapluat-vn-tnpl"

# --- Phạm vi corpus ---------------------------------------------------------
# BM25/FTS5 phủ TOÀN BỘ 573.851 văn bản. Chỉ index vector mới bị giới hạn.
#
# tier trong bộ dữ liệu (đã kiểm chứng trên mẫu 72.000 dòng):
#   0 = primary legislation        ~61.800 vb  (Luật, Bộ luật, Pháp lệnh, Nghị định, Thông tư)
#   1 = central decisions          ~54.200 vb  (Quyết định/Chỉ thị trung ương)
#   2 = provincial & administrative ~418.700 vb (Quyết định địa phương — giá trị tra cứu thấp)
#   3 = official letters & standards ~39.200 vb (Công văn, TCVN)
VECTOR_MAX_TIER = int(os.getenv("LUATVN_MAX_TIER", "0"))

# status null ở ~92% số dòng nên KHÔNG dùng được như cờ "còn hiệu lực".
# Chỉ loại những văn bản ghi rõ đã hết hiệu lực; null = không rõ = vẫn giữ.
EXCLUDE_EXPIRED = os.getenv("LUATVN_EXCLUDE_EXPIRED", "1") == "1"

# --- Chunking ---------------------------------------------------------------
# "Điều" là đơn vị trích dẫn tự nhiên của luật Việt Nam -> chunk theo Điều.
# Điều quá dài thì cắt tiếp theo Khoản, mỗi phần được gắn lại tiêu đề Điều.
MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 50
# Cắt bỏ phần đuôi của văn bản dài bất thường (phụ lục, biểu mẫu hàng nghìn dòng)
MAX_DOC_CHARS = 400_000
# Số ký tự tối đa giữ lại cho phần mở đầu (căn cứ pháp lý) của mỗi văn bản
PREAMBLE_CHARS = 1200

# --- Model ------------------------------------------------------------------
# Mặc định e5-small: đo trên tier 0 thật (2.90M chunk) -> 3.7h build, index 4.5GB.
# e5-base cho recall tốt hơn nhưng ~4.5h build, index 8.9GB:
#   LUATVN_EMBED_MODEL=intfloat/multilingual-e5-base LUATVN_EMBED_DIM=768
# Recall vòng 1 yếu hơn được bù bởi BM25 (bắt đúng thuật ngữ pháp lý) + reranker.
EMBED_MODEL = os.getenv("LUATVN_EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("LUATVN_EMBED_DIM", "384"))
EMBED_MAX_TOKENS = 512
EMBED_BATCH = int(os.getenv("LUATVN_EMBED_BATCH", "64"))
# e5 yêu cầu tiền tố "query: " / "passage: " — sai tiền tố là mất ~10% chất lượng
EMBED_QUERY_PREFIX = "query: "
EMBED_DOC_PREFIX = "passage: "

RERANK_MODEL = os.getenv("LUATVN_RERANK_MODEL", "AITeamVN/Vietnamese_Reranker")
RERANK_MAX_TOKENS = 512

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LUATVN_LLM_MODEL", "qwen3:8b")
LLM_NUM_CTX = int(os.getenv("LUATVN_NUM_CTX", "16384"))

# --- Retrieval --------------------------------------------------------------
BM25_TOPK = 40      # ứng viên lấy từ FTS5 mỗi nguồn
VECTOR_TOPK = 40    # ứng viên lấy từ index vector mỗi nguồn
RRF_K = 60          # hằng số Reciprocal Rank Fusion
RERANK_CANDIDATES = 32   # số ứng viên đưa vào cross-encoder
CONTEXT_TOPK = 8         # số đoạn cuối cùng đưa vào prompt LLM
CONTEXT_MAX_CHARS = 24_000


def ensure_dirs() -> None:
    for d in (DATA, RAW, DB_PATH.parent, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
