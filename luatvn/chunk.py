"""Tách toàn văn văn bản pháp luật thành các đoạn theo cấu trúc Điều / Khoản.

Vì sao không cắt theo độ dài cố định: câu trả lời pháp lý phải trích dẫn được
"Điều x Khoản y" — cắt mù theo ký tự sẽ làm đứt đôi một Điều và khiến trích dẫn
sai. Ở đây mỗi đoạn luôn nằm gọn trong một Điều và mang theo số Điều của nó.

Đặc điểm dữ liệu thật (đã kiểm chứng): `vn_text` là text thuần, xuống dòng cứng
ở giữa câu, nhưng tiêu đề "Điều N." LUÔN đứng đầu dòng. Do đó bắt tiêu đề bằng
regex neo đầu dòng (MULTILINE) thay vì trên text đã làm phẳng — tránh bắt nhầm
các tham chiếu giữa câu như "Căn cứ Điều 5 Luật Doanh nghiệp".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from . import config

# "Điều 5.", "Điều 12a:", "Ðiều 3 ." — chấp nhận cả biến thể Đ/Ð của Unicode.
# BẮT BUỘC có dấu chấm/hai chấm ngay sau số: đo trên 1.500 văn bản tier<=1 với
# `num_articles` làm ground truth, ràng buộc này nâng tỉ lệ khớp chính xác từ
# 53.3% -> 58.1% và giảm bắt thừa 24.0% -> 18.7% (tham chiếu giữa câu như
# "quy định tại\nĐiều 5 của Luật này" bị xuống dòng cứng rơi vào đầu dòng).
RE_DIEU = re.compile(r"^[ \t]*(?:Điều|Ðiều|ĐIỀU)[ \t]+(\d+[a-zA-Z]?)[ \t]*[.．:·]", re.MULTILINE)
RE_CHUONG = re.compile(r"^[ \t]*(Chương|CHƯƠNG|Phần|PHẦN|Mục|MỤC)[ \t]+([IVXLCDM\d]+[a-zA-Z]?)\b.*$", re.MULTILINE)
# Đầu Khoản: "1.", "2 ." ở đầu dòng (không phải "1.2." hay ngày tháng)
RE_KHOAN = re.compile(r"^[ \t]*(\d{1,2})[ \t]*[.)][ \t]+(?=[^\d])", re.MULTILINE)
# Phụ lục / biểu mẫu: phần đuôi thường là bảng biểu dài, giá trị tra cứu thấp
RE_PHULUC = re.compile(r"^[ \t]*(PHỤ LỤC|Phụ lục)\b", re.MULTILINE)
RE_NOINHAN = re.compile(r"^[ \t]*Nơi nhận[ \t]*:", re.MULTILINE)


@dataclass(slots=True)
class Chunk:
    """Một đoạn văn bản có thể trích dẫn được."""

    chunk_id: str
    doc_id: str
    book: int          # 0 = thân văn bản chính; >=1 = quy chế/điều lệ ban hành kèm theo
    chuong: str        # tiêu đề Chương/Mục gần nhất phía trên
    dieu_no: str       # "5", "12a", hoặc "" nếu là phần mở đầu
    dieu_title: str    # câu tiêu đề của Điều (dòng đầu)
    part: int          # chỉ số phần khi một Điều bị cắt nhỏ
    n_parts: int
    text: str

    @property
    def label(self) -> str:
        """Nhãn trích dẫn ngắn hiển thị cho người dùng."""
        if not self.dieu_no:
            return "Phần mở đầu"
        s = f"Điều {self.dieu_no}"
        if self.n_parts > 1:
            s += f" (phần {self.part + 1}/{self.n_parts})"
        return s


def normalize(text: str) -> str:
    """Chuẩn hoá Unicode và khoảng trắng, GIỮ NGUYÊN cấu trúc dòng."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_tail(text: str) -> str:
    """Bỏ phần phụ lục/biểu mẫu ở đuôi và giới hạn độ dài tổng."""
    m = RE_PHULUC.search(text)
    # chỉ cắt nếu phụ lục nằm ở nửa sau và phần còn lại vẫn đủ dài
    if m and m.start() > 2000 and m.start() > len(text) * 0.4:
        text = text[: m.start()]
    if len(text) > config.MAX_DOC_CHARS:
        text = text[: config.MAX_DOC_CHARS]
    return text


def _split_long(body: str, limit: int) -> list[str]:
    """Cắt một Điều quá dài, ưu tiên ranh giới Khoản, sau đó tới ranh giới dòng."""
    if len(body) <= limit:
        return [body]

    bounds = [m.start() for m in RE_KHOAN.finditer(body)]
    if not bounds or bounds[0] > limit:
        bounds = [m.start() for m in re.finditer(r"\n", body)]
    bounds = [b for b in bounds if b > 0]
    if not bounds:
        return [body[i : i + limit] for i in range(0, len(body), limit)]

    parts: list[str] = []
    start = 0
    while start < len(body):
        if len(body) - start <= limit:
            parts.append(body[start:])
            break
        # điểm cắt xa nhất còn nằm trong giới hạn
        cand = [b for b in bounds if start < b <= start + limit]
        cut = cand[-1] if cand else start + limit
        parts.append(body[start:cut])
        start = cut
    return [p for p in parts if p.strip()]


def split_document(doc_id: str, vn_text: str) -> list[Chunk]:
    """Tách toàn văn một văn bản thành danh sách Chunk."""
    if not vn_text:
        return []
    text = _truncate_tail(normalize(vn_text))
    if len(text) < config.MIN_CHUNK_CHARS:
        return []

    heads = list(RE_DIEU.finditer(text))
    chuongs = [(m.start(), " ".join(m.group(0).split())) for m in RE_CHUONG.finditer(text)]

    def chuong_at(pos: int) -> str:
        found = ""
        for start, title in chuongs:
            if start <= pos:
                found = title
            else:
                break
        return found

    chunks: list[Chunk] = []

    # --- Phần mở đầu: tiêu đề + "Căn cứ ..." trước Điều đầu tiên ---
    pre_end = heads[0].start() if heads else min(len(text), config.PREAMBLE_CHARS)
    preamble = text[:pre_end].strip()
    if len(preamble) >= config.MIN_CHUNK_CHARS:
        preamble = preamble[: config.PREAMBLE_CHARS]
        chunks.append(Chunk(f"{doc_id}#pre", doc_id, 0, "", "", "", 0, 1, preamble))

    if not heads:
        return chunks

    # --- Từng Điều ---
    # Số Điều quay về 1-2 nghĩa là bắt đầu một văn bản ban hành kèm theo
    # (Quy chế, Điều lệ...). Chỉ tăng `book` khi thực sự reset về đầu, không tăng
    # với mọi lần tụt số — một tham chiếu bắt nhầm sẽ làm vỡ toàn bộ đánh số.
    book = 0
    prev_num = -1

    for i, m in enumerate(heads):
        raw_no = m.group(1)
        try:
            num = int(re.sub(r"[a-zA-Z]", "", raw_no))
        except ValueError:
            num = prev_num
        if prev_num >= 1 and num <= 2 and num < prev_num:
            book += 1
        prev_num = max(num, prev_num) if num < prev_num and not (num <= 2) else num

        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.start() : end].strip()
        if len(body) < config.MIN_CHUNK_CHARS:
            continue

        first_line = body.split("\n", 1)[0].strip()
        dieu_title = first_line[:250]
        chuong = chuong_at(m.start())

        parts = _split_long(body, config.MAX_CHUNK_CHARS)
        n = len(parts)
        for p_idx, part_text in enumerate(parts):
            part_text = part_text.strip()
            if len(part_text) < config.MIN_CHUNK_CHARS and n > 1:
                continue
            # phần 2 trở đi mất tiêu đề Điều -> gắn lại để đoạn tự đứng được
            if p_idx > 0:
                part_text = f"[tiếp] {dieu_title}\n{part_text}"
            suffix = f".{p_idx}" if n > 1 else ""
            bk = f"b{book}" if book else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{bk}d{raw_no}{suffix}",
                    doc_id=doc_id,
                    book=book,
                    chuong=chuong,
                    dieu_no=raw_no,
                    dieu_title=dieu_title,
                    part=p_idx,
                    n_parts=n,
                    text=part_text,
                )
            )
    return chunks
