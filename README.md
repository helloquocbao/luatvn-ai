# Trợ lý pháp luật Việt Nam — chạy hoàn toàn local

Chatbot hỏi–đáp pháp luật Việt Nam có trích dẫn, dựng trên corpus công khai của
`thuvienphapluat.vn`. Không gọi API bên ngoài: dữ liệu, mô hình nhúng, mô hình
xếp hạng và mô hình sinh câu trả lời đều chạy trên máy bạn.

> ## ⚠️ Mục đích sử dụng
>
> **Dự án này được thực hiện cho mục đích học tập và nghiên cứu cá nhân.**
>
> - **Không phải tư vấn pháp lý.** Câu trả lời do mô hình ngôn ngữ sinh ra nên
>   có thể sai, thiếu, hoặc dẫn văn bản đã hết hiệu lực. Không dùng làm căn cứ
>   cho quyết định pháp lý, kinh doanh hay tố tụng. Hãy hỏi luật sư có chứng
>   chỉ hành nghề.
> - **Luôn đối chiếu văn bản gốc.** Mỗi trích dẫn đều kèm liên kết tới nguồn —
>   hãy mở và tự kiểm chứng trước khi tin.
> - **Không nhằm mục đích thương mại.** Dự án không thay thế, không cạnh tranh
>   và không phân phối lại dịch vụ của thuvienphapluat.vn.
> - **Repo không kèm dữ liệu.** Chỉ có mã nguồn. Người dùng tự tải corpus từ
>   nguồn công khai và tự chịu trách nhiệm tuân thủ điều khoản của nguồn đó.
> - Phần mềm được cung cấp "nguyên trạng", không kèm bảo đảm dưới bất kỳ hình
>   thức nào. Tác giả không chịu trách nhiệm cho bất kỳ thiệt hại nào phát sinh
>   từ việc sử dụng.

## Dữ liệu

| Nguồn | Nội dung | Số lượng |
|---|---|---|
| `tmquan/thuvienphapluat-vn-vbpl` | Văn bản pháp luật + toàn văn | **573.851** văn bản |
| `tmquan/thuvienphapluat-vn-hdpl` | Hỏi đáp pháp luật có trích dẫn | **109.543** bài |
| `tmquan/thuvienphapluat-vn-tnpl` | Thuật ngữ pháp lý + định nghĩa | **17.091** thuật ngữ |

Chỉ tải config `documents` (~3,1 GB). Config `embeddings` của ba bộ cộng lại
**28,4 GB** (Nemotron-8B, 4096 chiều) nên bị bỏ qua — ta tự sinh vector 384
chiều vừa nhẹ hơn 17 lần vừa khớp đúng mô hình dùng lúc truy vấn.

Sau khi nạp: **1.859.336 đoạn theo Điều**, **1.112.926 vector**, CSDL 6,8 GB +
chỉ mục 1,7 GB.

## Kiến trúc

```
câu hỏi
   │
   ├─ BM25 / FTS5 ──┬─ chunks (1,86 M đoạn, phủ 117.257 văn bản có toàn văn)
   │                ├─ qa     (109.543 hỏi đáp)
   │                ├─ terms  (17.091 thuật ngữ)
   │                └─ docs   (573.851 tiêu đề — phủ TOÀN BỘ corpus)
   │                                                    5 nguồn chạy song song
   └─ vector / LanceDB IVF_PQ (1,11 M vector, e5-small)
                    │
              RRF (hợp nhất theo thứ hạng)
                    │
              khử trùng lặp  →  cross-encoder tiếng Việt xếp hạng lại
                    │
              8 đoạn tốt nhất  →  Qwen3-8B (Ollama)  →  trả lời kèm [1][2]
```

**Vì sao lai BM25 + vector.** Câu hỏi pháp luật hay chứa định danh chính xác
("Điều 5 Luật Doanh nghiệp 2020", "Nghị định 100/2019"). Embedding rất kém với
loại chuỗi này, BM25 thì bắt chuẩn. Ngược lại BM25 mù với cách diễn đạt khác từ
("bị sa thải" ↔ "đơn phương chấm dứt hợp đồng lao động"). RRF gộp hai bảng xếp
hạng chỉ dựa trên thứ hạng nên không phải chuẩn hoá hai thang điểm khác nhau.

**Vì sao bi-encoder nhỏ + cross-encoder lớn.** Đo thực tế trên M4/16GB:

| Mô hình | Kích thước | Tốc độ | Thời gian nhúng 1,11 M đoạn |
|---|---|---|---|
| `AITeamVN/Vietnamese_Embedding` (bge-m3) | 568 M | 20,6/s | ~15 giờ |
| `intfloat/multilingual-e5-base` | 278 M | 67,9/s | ~4,5 giờ |
| **`intfloat/multilingual-e5-small`** ✔ | 118 M | 217/s | **~2,2 giờ** |

Mô hình nhỏ chỉ cần đủ tốt để đưa đáp án đúng vào top-40; chất lượng thật do
`AITeamVN/Vietnamese_Reranker` quyết định, và nó chỉ chấm ~32 cặp mỗi truy vấn
nên chi phí không đáng kể.

## Cài đặt

```bash
pip3 install --user uv          # nếu chưa có
uv sync                          # tạo môi trường Python 3.12
```

Cần Ollama cho phần sinh câu trả lời:

```bash
brew install ollama && ollama serve && ollama pull qwen3:8b
```

## Dựng dữ liệu (một lần)

```bash
uv run python -m luatvn.download   # tải 3,1 GB parquet          (~5 phút)
uv run python -m luatvn.ingest     # parquet -> SQLite + FTS5     (~7 phút)
uv run python -m luatvn.enrich     # trích tiêu đề/số hiệu thật   (~10 giây)
uv run python -m luatvn.embed      # dựng chỉ mục vector          (~2,2 giờ)
```

`ingest` và `embed` ghi tiến độ vào bảng `build_state`, dừng giữa chừng rồi
chạy lại sẽ tiếp tục từ chỗ cũ chứ không làm lại từ đầu.

## Chạy

```bash
./run.sh                                      # khởi động Ollama + web: http://localhost:8000
uv run uvicorn luatvn.api:app --port 8000     # hoặc chạy riêng máy chủ web
uv run python -m luatvn.cli "nồng độ cồn xe máy bị phạt bao nhiêu"
uv run python -m luatvn.cli --search "..."    # chỉ tra cứu, không cần LLM
uv run python eval/run_eval.py --ablate       # đo chất lượng truy xuất
```

## API

| Endpoint | Công dụng |
|---|---|
| `GET /api/health` | thống kê corpus, trạng thái mô hình |
| `POST /api/search` | chỉ truy xuất (không cần Ollama) |
| `POST /api/chat` | trả lời dạng stream (SSE) kèm trích dẫn |
| `GET /api/doc/{id}` | toàn văn một văn bản, tách theo Điều |
| `GET /api/docs?q=&doc_type=&year=` | duyệt/tìm văn bản |

## Tinh chỉnh

Đổi qua biến môi trường (xem `luatvn/config.py`):

```bash
LUATVN_EMBED_MODEL=intfloat/multilingual-e5-base LUATVN_EMBED_DIM=768 \
  uv run python -m luatvn.embed       # recall tốt hơn, đổi lại ~4,5 giờ dựng

LUATVN_MAX_TIER=1 uv run python -m luatvn.embed   # phủ thêm ~54.000 văn bản cấp trung ương
LUATVN_LLM_MODEL=qwen3:14b uv run uvicorn luatvn.api:app
```

## Chất lượng đo được

`eval/gold.json` gồm 10 câu hỏi pháp luật thực tế, mỗi câu gắn với **đúng một
Điều của đúng một văn bản**. Chạy `uv run python eval/run_eval.py --ablate`:

| Cấu hình | R@1 | R@3 | MRR | giây/truy vấn |
|---|---|---|---|---|
| **Đầy đủ (BM25 + vector + rerank)** | **0,60** | **0,70** | **0,650** | 4,2 |
| Bỏ rerank | 0,10 | 0,20 | 0,195 | 1,2 |
| Bỏ vector (chỉ BM25) | 0,10 | 0,20 | 0,150 | 2,2 |
| Chỉ BM25, không rerank | 0,10 | 0,10 | 0,100 | 0,6 |

Hai con số đáng chú ý: bỏ **reranker** làm MRR rơi từ 0,650 xuống 0,195, và bỏ
**vector** cũng rơi tương tự. Tức là kiến trúc "bi-encoder nhỏ + cross-encoder
mạnh" không phải lựa chọn tiết kiệm — nó là thứ tạo ra phần lớn chất lượng.
Ba câu trượt đều là trượt sát (ví dụ trả về Điều 135 thay vì Điều 134 Bộ luật
Hình sự), nên R@3 = 0,70 phản ánh đúng mức dùng được hơn.

## Phạm vi đang phủ

- **Tra cứu từ khoá (BM25): toàn bộ 573.851 văn bản** theo tiêu đề và số hiệu;
  toàn văn cho 117.257 văn bản (tier 0–1).
- **Tra cứu ngữ nghĩa (vector): 986.292 đoạn** thuộc văn bản tier 0 còn hiệu lực
  (Luật, Bộ luật, Pháp lệnh, Nghị định, Thông tư), cộng toàn bộ hỏi đáp và thuật ngữ.
- Văn bản cấp tỉnh (tier 2, ~419.000) hiện chỉ tra được theo metadata.

## Nên làm tiếp

- **Dựng lại chỉ mục vector sau khi có tiêu đề chuẩn.** Vector hiện tại được
  nhúng trước khi `enrich` ghép loại văn bản vào tiêu đề, nên tiền tố của mỗi
  đoạn còn là "Doanh nghiệp" thay vì "Luật doanh nghiệp". Chạy lại
  `uv run python -m luatvn.embed` (~2,2 giờ) sẽ khớp lại; hãy xoá
  `data/index/lance` và các dòng `embed_cursor_*` trong bảng `build_state` trước.
- **Mở rộng bộ đánh giá.** 10 câu là đủ để bắt lỗi lớn nhưng quá ít để so sánh
  các thay đổi nhỏ. Thêm 50–100 câu sẽ giúp tinh chỉnh đáng tin cậy hơn.
- **Phủ vector cho tier 1** (`LUATVN_MAX_TIER=1`) nếu cần tra cả quyết định,
  chỉ thị cấp trung ương.

## Hạn chế cần biết

- Bộ tách Điều khớp chính xác `num_articles` trong metadata ở **58%** số văn bản,
  lệch tuyệt đối trung vị bằng 0. Phần sai chủ yếu là văn bản có quy chế ban hành
  kèm theo với hệ thống đánh số riêng.
- Trích được tiêu đề tiếng Việt cho **86,8%** văn bản có toàn văn; phần còn lại
  hiển thị theo slug không dấu của nguồn.
- Cột `status` rỗng ở ~92% số dòng nên **không thể coi là cờ "còn hiệu lực"**.
  Hệ thống chỉ loại văn bản ghi rõ "Hết hiệu lực" (5.836 văn bản) và đẩy chúng
  xuống cuối; văn bản không rõ trạng thái vẫn được giữ.
- Vì vậy: **luôn đối chiếu lại với văn bản gốc trước khi sử dụng.** Mỗi nguồn
  trong giao diện đều có liên kết tới trang gốc trên thuvienphapluat.vn.
- Đây là công cụ tra cứu, không phải tư vấn pháp lý.

## Giấy phép

**Mã nguồn** trong repo này dùng cho mục đích học tập và nghiên cứu.

**Dữ liệu không thuộc repo và không được phát hành lại kèm theo.** `vbpl`
CC-BY-4.0 (phần metadata), `tnpl`/`hdpl` theo giấy phép riêng của nguồn; nội
dung toàn văn văn bản thuộc © thuvienphapluat.vn. Repo chỉ chứa mã để *xử lý*
dữ liệu đã công khai, không chứa và không phân phối bản sao dữ liệu đó.

Văn bản quy phạm pháp luật do Nhà nước ban hành không phải đối tượng bảo hộ
quyền tác giả theo Điều 15 Luật Sở hữu trí tuệ, nhưng phần biên tập, chuẩn hoá
và cơ sở dữ liệu của thuvienphapluat.vn thì có. Nếu bạn định dùng ngoài phạm vi
học tập cá nhân, hãy tự kiểm tra điều khoản sử dụng của nguồn.
