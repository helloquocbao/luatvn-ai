"""Tải các config `documents` của 3 bộ dữ liệu thuvienphapluat từ HuggingFace.

Cố tình BỎ QUA config `embeddings` (28.4 GB, Nemotron-8B 4096-d) và `pages`
(3.9 GB HTML thô): ta tự sinh embedding 384 chiều để tiết kiệm ổ cứng và
để vector khớp đúng với model dùng lúc query.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# repo_id -> các pattern file cần lấy
SPECS: dict[str, list[str]] = {
    "tmquan/thuvienphapluat-vn-tnpl": ["documents.parquet", "README.md"],
    "tmquan/thuvienphapluat-vn-vbpl": ["documents-*.parquet", "README.md"],
    "tmquan/thuvienphapluat-vn-hdpl": ["documents-*.parquet", "README.md"],
}


def fetch(repo_id: str, patterns: list[str]) -> Path:
    local = RAW / repo_id.split("/")[-1]
    local.mkdir(parents=True, exist_ok=True)
    print(f"[tải] {repo_id} -> {local}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=patterns,
        max_workers=4,
    )
    return local


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for repo_id, patterns in SPECS.items():
        path = fetch(repo_id, patterns)
        size = sum(f.stat().st_size for f in path.rglob("*.parquet"))
        print(f"[xong] {repo_id}: {size / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
