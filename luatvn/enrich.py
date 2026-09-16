"""Trích tiêu đề và số hiệu THẬT của văn bản từ phần mở đầu.

Vì sao cần: cột `title` trong dataset là slug ASCII không dấu suy ra từ URL
("Nghi dinh 73 2016 ND CP huong dan Luat Kinh doanh bao hiem") và `doc_number`
bị nối thêm cả slug ("20/2017/TT-BYT-HUONG-DAN-LUAT-DUOC-54-2017-ND-CP...").
Dùng trực tiếp thì trích dẫn xấu và — quan trọng hơn — tiêu đề không dấu bị
nhúng kèm mỗi chunk sẽ làm nhiễu vector.

Phần mở đầu văn bản có cấu trúc rất ổn định:
    <cơ quan> ... Số: 20/2017/TT-BYT ... Hà Nội, ngày ...
    THÔNG TƯ
    QUY ĐỊNH CHI TIẾT MỘT SỐ ĐIỀU CỦA LUẬT DƯỢC ...      <- tiêu đề thật
    Căn cứ ...
nên lấy khối in hoa nằm giữa dòng ngày tháng và chữ "Căn cứ".

    uv run python -m luatvn.enrich
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata

from . import db

RE_SO = re.compile(r"\bSố:?\s*([0-9][^\n]{0,60})", re.IGNORECASE)
# "24/2007/ND-CP", "59/2020/QH14", "01/2020/TTLT-BCA-BQP"
RE_NUM_FULL = re.compile(r"^\s*([0-9]+[a-zA-Z]?)\s*/\s*([0-9]{2,4})\s*/\s*(\S.*)$")
# "1688/QĐ-TTg", "61-QĐ/TW"
RE_NUM_SHORT = re.compile(r"^\s*([0-9]+[a-zA-Z]?)\s*([/-])\s*([A-ZĐ]\S*)$")
# điểm kết thúc của khối tiêu đề
RE_TITLE_END = re.compile(
    r"\b(Căn\s+cứ|Theo\s+đề\s+nghị|Xét\s+đề\s+nghị|Để\s+thực\s+hiện|QUYẾT\s+ĐỊNH:|Điều\s+1\s*\.)",
    re.IGNORECASE,
)
RE_DATE_LINE = re.compile(r"ngày\s+\d{1,2}\s*\n?\s*tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE)
# các dòng loại văn bản in hoa đứng ngay trước tiêu đề
RE_KIND_HEAD = re.compile(
    r"^\s*(LUẬT|BỘ\s*LUẬT|PHÁP\s*LỆNH|NGHỊ\s*ĐỊNH|NGHỊ\s*QUYẾT|THÔNG\s*TƯ(?:\s*LIÊN\s*TỊCH)?|"
    r"QUYẾT\s*ĐỊNH|CHỈ\s*THỊ|HIẾN\s*PHÁP|SẮC\s*LỆNH|VĂN\s*BẢN\s*HỢP\s*NHẤT|LỆNH|"
    r"THÔNG\s*BÁO|CÔNG\s*VĂN|QUY\s*CHẾ|QUY\s*ĐỊNH)\s*$",
    re.IGNORECASE,
)


def _trim_suffix(rest: str) -> str:
    """Giữ lại phần mã cơ quan, cắt bỏ slug tên văn bản dính phía sau.

    Slug nguồn nối cả tên văn bản vào số hiệu bằng gạch ngang viết hoa
    ("24/2007/ND-CP-HUONG-DAN-LUAT-THUE-THU-NHAP-DOANH-NGHIEP") — về mặt ký tự
    nó giống hệt mã cơ quan nên không regex nào tách được. Chặn theo SỐ NHÓM và
    ĐỘ DÀI: mã cơ quan thật nhiều nhất 3 nhóm và hiếm khi quá 20 ký tự
    ("NĐ-CP", "TT-BLDTBXH", "TTLT-BCA-BQP").
    """
    # Số hiệu lấy từ phần mở đầu thường dính cả phần sau ("NĐ-CP ngày 26 tháng
    # 12 năm 2024") -> cắt tại khoảng trắng đầu tiên trước khi xử lý nhóm.
    rest = rest.split()[0] if rest.split() else rest
    groups = [g for g in rest.split("-") if g]
    if not groups:
        return rest
    # Mã cơ quan hầu như luôn có 2 nhóm ("NĐ-CP", "TT-BLDTBXH"). Cho phép nhóm
    # thứ 3 thì "24/2007/ND-CP-HUONG-DAN-..." lại lọt thành "ND-CP-HUONG".
    # Chỉ văn bản LIÊN TỊCH mới thật sự có 3+ nhóm ("TTLT-BCA-BQP-BTC").
    limit = 4 if groups[0].upper().startswith(("TTLT", "TTLB", "NQLT")) else 2
    out: list[str] = []
    for g in groups[:limit]:
        cand = "-".join([*out, g])
        if len(cand) > 24:
            break
        out.append(g)
    return "-".join(out) if out else rest


def clean_number(raw: str | None, preamble: str) -> str:
    """Ưu tiên số hiệu in trong phần mở đầu; nếu không có thì làm sạch metadata."""
    for cand in (_from_preamble(preamble), (raw or "").strip()):
        if not cand:
            continue
        m = RE_NUM_FULL.match(cand)
        if m:
            num, year, rest = m.groups()
            return f"{num}/{year}/{_trim_suffix(rest)}"
        m = RE_NUM_SHORT.match(cand)
        if m:
            num, sep, rest = m.groups()
            return f"{num}{sep}{_trim_suffix(rest)}"
    return (raw or "").split("-")[0] if raw else ""


def _from_preamble(preamble: str) -> str:
    m = RE_SO.search(preamble or "")
    return " ".join(m.group(1).split()) if m else ""


# Danh từ riêng cần viết hoa lại sau khi hạ ALL-CAPS về dạng câu
PROPER = [
    "Việt Nam", "Chính phủ", "Quốc hội", "Thủ tướng", "Nhà nước", "Hiến pháp",
    "Bộ luật", "Luật", "Nghị định", "Thông tư", "Nghị quyết", "Pháp lệnh",
    "Quyết định", "Chỉ thị", "Hà Nội", "Trung ương",
]


def _sentence_case(title: str) -> str:
    """Hạ tiêu đề ALL-CAPS về dạng câu rồi phục hồi danh từ riêng."""
    title = title.capitalize()
    for word in PROPER:
        title = re.sub(rf"\b{re.escape(word.lower())}\b", word, title)
    return title[:1].upper() + title[1:]


def extract_title(preamble: str) -> str:
    """Lấy khối tiêu đề in hoa nằm giữa dòng loại văn bản và chữ 'Căn cứ'.

    Neo theo DÒNG LOẠI VĂN BẢN ("THÔNG TƯ", "NGHỊ ĐỊNH") chứ không theo dòng
    ngày tháng: nhiều tiêu đề có ngày tháng nằm bên trong ("... NGHỊ ĐỊNH SỐ
    54/2017/NĐ-CP NGÀY 08 THÁNG 5 NĂM 2017 CỦA CHÍNH PHỦ ...") và neo theo ngày
    sẽ cắt mất nửa đầu tiêu đề.
    """
    if not preamble:
        return ""
    end = RE_TITLE_END.search(preamble)
    seg = preamble[: end.start()] if end else preamble

    lines = [ln.strip() for ln in seg.split("\n")]
    # tìm dòng loại văn bản cuối cùng trong phần đầu -> tiêu đề bắt đầu ngay sau đó
    kind_at = -1
    for i, ln in enumerate(lines):
        if RE_KIND_HEAD.match(ln):
            kind_at = i
    if kind_at >= 0:
        lines = lines[kind_at + 1:]
        # loại văn bản có thể bị xuống dòng giữa chừng: "THÔNG\nTƯ"
        if lines and lines[0].upper() in {
            "TƯ", "ĐỊNH", "QUYẾT", "LỆNH", "PHÁP", "LUẬT", "THỊ", "NHẤT", "SỰ",
        }:
            lines.pop(0)
    else:
        dm = list(RE_DATE_LINE.finditer(seg))
        if dm:
            lines = seg[dm[0].end():].split("\n")

    title = " ".join(w for w in " ".join(lines).split() if w)
    title = re.sub(r"^[-–—:.\s]+", "", title)
    if len(title) < 6 or len(title) > 400:
        return ""
    # Khi không tìm được dòng loại văn bản, phần cắt ra có thể là đoạn đầu trang
    # ("Số: 10/2012/QH13 Hà Nội, ngày 18 tháng 6 năm 2012 BỘ LUẬT..."). Thà trả
    # về rỗng để rơi về slug còn hơn hiển thị rác như một tiêu đề.
    low = title.lower()
    if low.startswith(("số:", "so:", "căn cứ", "quốc hội", "chính phủ")) or (
        "hà nội, ngày" in low[:120] or "độc lập - tự do" in low[:120]
    ):
        return ""
    letters = [c for c in title if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        title = _sentence_case(title)
    return title


def full_title(doc_type: str | None, title: str) -> str:
    """Ghép loại văn bản vào trước tiêu đề để ra ĐÚNG tên gọi thông dụng.

    Bộ trích tiêu đề cố tình bỏ dòng loại văn bản ("LUẬT", "NGHỊ ĐỊNH") vì nó
    nằm riêng một dòng phía trên. Nhưng như vậy bộ luật lại mang tên "Doanh
    nghiệp" thay vì "Luật Doanh nghiệp" — vừa xấu khi trích dẫn, vừa khiến tra
    "Luật Doanh nghiệp" thua nghị định hướng dẫn (nghị định có nguyên cụm đó
    trong tiêu đề, còn bộ luật thì không).
    """
    if not title:
        return ""
    if not doc_type:
        return title
    if _strip(title).startswith(_strip(doc_type)):
        return title
    return f"{doc_type} {title[0].lower() + title[1:]}" if len(title) > 1 else title


def _strip(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("đ", "d")


RE_YEAR_IN_NUM = re.compile(r"^[0-9]+[a-zA-Z]?/((?:19|20)[0-9]{2})/")


def infer_year(number: str, issue_date: str | None, current: int | None) -> int | None:
    """Suy ra năm ban hành khi metadata để trống.

    `year` rỗng ở khá nhiều văn bản quan trọng (Bộ luật Lao động 45/2019/QH14 là
    một ví dụ), mà xếp hạng lại dùng năm để ưu tiên bản đang áp dụng — thiếu năm
    thì chính bộ luật thua nghị định hướng dẫn nó. Số hiệu đã chứa sẵn năm.
    """
    if current:
        return current
    m = RE_YEAR_IN_NUM.match(number or "")
    if m:
        return int(m.group(1))
    m = re.search(r"\b((?:19|20)[0-9]{2})\b", issue_date or "")
    return int(m.group(1)) if m else None


def main() -> int:
    con = db.connect()
    cols = {r["name"] for r in con.execute("PRAGMA table_info(docs)")}
    if "title_vi" not in cols:
        con.execute("ALTER TABLE docs ADD COLUMN title_vi TEXT")
        con.execute("ALTER TABLE docs ADD COLUMN number_vi TEXT")
        con.commit()

    rows = con.execute(
        "SELECT d.id, d.doc_number, d.doc_type, d.year, d.issue_date, c.text FROM docs d "
        "JOIN chunks c ON c.doc_id = d.id AND c.seq = 0 "
        "WHERE d.has_text = 1"
    ).fetchall()
    print(f"xử lý {len(rows):,} văn bản có toàn văn...", flush=True)

    updates, n_title, n_year, t0 = [], 0, 0, time.time()
    for r in rows:
        pre = r["text"]
        title = full_title(r["doc_type"], extract_title(pre))
        num = clean_number(r["doc_number"], pre)
        year = infer_year(num, r["issue_date"], r["year"])
        n_title += bool(title)
        n_year += year is not None and r["year"] is None
        updates.append((title or None, num or None, year, r["id"]))

    con.executemany("UPDATE docs SET title_vi=?, number_vi=?, year=? WHERE id=?", updates)
    con.commit()
    print(f"  trích được tiêu đề: {n_title:,}/{len(rows):,} "
          f"({n_title / max(len(rows), 1) * 100:.1f}%) trong {time.time() - t0:.1f}s", flush=True)
    print(f"  bổ sung năm ban hành còn thiếu: {n_year:,}", flush=True)

    # FTS phải được dựng lại để tìm được theo tiêu đề tiếng Việt có dấu
    print("dựng lại fts_docs với tiêu đề tiếng Việt...", flush=True)
    con.execute("DROP TABLE IF EXISTS fts_docs")
    # doc_type PHẢI nằm trong chỉ mục: tiêu đề trích ra đã bị bỏ phần loại văn
    # bản ở đầu (Luật Doanh nghiệp -> title_vi "Doanh nghiệp"), nên không có cột
    # này thì truy vấn "Luật Doanh nghiệp" không bao giờ ra đúng bộ luật đó.
    con.execute(
        "CREATE VIRTUAL TABLE fts_docs USING fts5("
        "title, doc_number, issuer, title_vi, number_vi, doc_type,"
        " content='docs', tokenize='unicode61 remove_diacritics 2')"
    )
    con.execute("INSERT INTO fts_docs(fts_docs) VALUES('rebuild')")
    con.execute("INSERT INTO fts_docs(fts_docs) VALUES('optimize')")
    con.commit()
    print("xong", flush=True)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
