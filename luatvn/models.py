"""Nạp model theo kiểu lazy + singleton.

API phục vụ nhiều request đồng thời nhưng chỉ có 16 GB RAM, nên mỗi model chỉ
được nạp một lần và dùng chung. Model chạy trên MPS (GPU tích hợp của Apple)
với float16 — đo thực tế nhanh hơn float32 khoảng 20% và giảm nửa bộ nhớ.
"""

from __future__ import annotations

import threading

from . import config

_lock = threading.Lock()
_embedder = None
_reranker = None


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedder():
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                import torch
                from sentence_transformers import SentenceTransformer

                dev = _device()
                m = SentenceTransformer(
                    config.EMBED_MODEL,
                    device=dev,
                    model_kwargs={"torch_dtype": torch.float16 if dev == "mps" else torch.float32},
                )
                m.max_seq_length = config.EMBED_MAX_TOKENS
                _embedder = m
    return _embedder


class _Reranker:
    """Cross-encoder nạp thẳng bằng transformers.

    Không dùng `sentence_transformers.CrossEncoder`: với transformers 5.x nó đi
    qua AutoProcessor và văng `Unrecognized processing class` trên các repo
    reranker chỉ có tokenizer SentencePiece. Nạp tay vừa ít phụ thuộc vừa ổn định.
    """

    def __init__(self, name: str, device: str, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(name, dtype=dtype).to(device).eval())

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        """Chấm điểm từng cặp (câu hỏi, đoạn văn bản).

        Các cặp được xếp theo độ dài trước khi chia lô: đệm (padding) tính theo
        cặp DÀI NHẤT trong lô, nên trộn lẫn đoạn ngắn với đoạn dài khiến phần
        lớn phép tính chạy trên token đệm vô nghĩa. Xếp theo độ dài rồi trả kết
        quả về đúng thứ tự ban đầu.
        """
        if not pairs:
            return []
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        scores = [0.0] * len(pairs)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = [pairs[j] for j in idx]
            enc = self.tok([b[0] for b in batch], [b[1] for b in batch],
                           padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            for j, v in zip(idx, logits.view(-1).float().cpu().tolist()):
                scores[j] = v
        return scores


def get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = _Reranker(
                    config.RERANK_MODEL, _device(), config.RERANK_MAX_TOKENS
                )
    return _reranker


def embed_queries(texts: list[str]):
    """Nhúng câu truy vấn (e5 bắt buộc tiền tố 'query: ')."""
    m = get_embedder()
    return m.encode(
        [config.EMBED_QUERY_PREFIX + t for t in texts],
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_passages(texts: list[str], *, progress: bool = False):
    """Nhúng đoạn văn bản (e5 bắt buộc tiền tố 'passage: ')."""
    m = get_embedder()
    return m.encode(
        [config.EMBED_DOC_PREFIX + t for t in texts],
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=progress,
    )
