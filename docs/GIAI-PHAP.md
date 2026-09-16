# Giải pháp trợ lý pháp luật Việt Nam — tài liệu kỹ thuật

> Tài liệu này mô tả **đã làm gì với dữ liệu, tại sao làm như vậy, và kết quả đo được ra sao**.
> Viết cho chủ dự án, không giả định người đọc là kỹ sư machine learning.
> Mọi con số trong tài liệu đều lấy từ README, docstring trong mã nguồn, hoặc log build thật
> (`data/ingest2.log`, `data/embed.log`, `data/download.log`). Chỗ nào chưa đo, tài liệu ghi rõ "chưa đo".

---

## 1. Tóm tắt

**Đã làm gì.** Lấy 3 bộ dữ liệu pháp luật công khai trên HuggingFace (tổng hơn **700.000 mục**),
biến chúng thành một **cơ sở tra cứu có trích dẫn** chạy hoàn toàn trên một chiếc MacBook Apple M4 /
16 GB RAM, rồi gắn một mô hình ngôn ngữ local lên trên để trả lời câu hỏi pháp luật **kèm số điều
khoản và liên kết tới văn bản gốc**.

**Không có mô hình nào được huấn luyện lại.** Đây là hệ **RAG** (Retrieval-Augmented Generation —
"sinh câu trả lời có tăng cường bằng truy xuất"): máy đi tìm đúng điều luật trước, rồi mới yêu cầu
mô hình ngôn ngữ viết câu trả lời **dựa trên và chỉ dựa trên** những điều luật vừa tìm được.
Mục 2 giải thích vì sao đây là lựa chọn đúng cho bài toán pháp luật.

**Quy mô đã dựng xong:**

| Hạng mục | Con số |
|---|---|
| Văn bản pháp luật (metadata, tra cứu được 100%) | **573.851** |
| Văn bản có toàn văn đã tách theo Điều | **117.257** |
| Đoạn văn bản theo Điều (chunk) | **1.859.336** |
| Bài hỏi đáp pháp luật | **109.543** |
| Thuật ngữ pháp lý | **17.091** |
| Vector ngữ nghĩa đã sinh | **1.112.926** |
| Dung lượng CSDL SQLite | 7,3 GB |
| Dung lượng chỉ mục vector | 1,7 GB |

**Mất bao lâu.** Toàn bộ quá trình dựng dữ liệu (chạy một lần duy nhất) khoảng **3 giờ**:

| Bước | Thời gian (đo thật) |
|---|---|
| `download` — tải 3,1 GB parquet | ~2 phút (log: vbpl 1 phút 35 giây) |
| `ingest` — parquet → SQLite + tách Điều + chỉ mục BM25 | ~7 phút (riêng dựng FTS5 cho 1,86 triệu đoạn: 72,1 giây) |
| `enrich` — trích tiêu đề/số hiệu thật | ~10 giây |
| `embed` — sinh 1,11 triệu vector + chỉ mục ANN | **2 giờ 06 phút** (23:15:46 → 01:21:40) |

**Chất lượng.** Trên bộ 10 câu hỏi có đáp án biết trước: **R@1 = 0,60 · R@3 = 0,70 · MRR = 0,650**,
4,2 giây mỗi truy vấn. Chi tiết và ý nghĩa ở mục 5.

---

## 2. RAG vs Fine-tuning — vì sao chọn RAG

### 2.1 Hai cách làm, nói cho dễ hiểu

Hãy hình dung mô hình ngôn ngữ (LLM) như một luật sư mới ra trường, trí nhớ tốt nhưng hay nhớ nhầm.

**Fine-tuning** (huấn luyện tinh chỉnh) = bắt anh ta học thuộc lòng toàn bộ thư viện luật.
Kiến thức nằm *trong đầu* anh ta, dưới dạng những con số trong mạng nơ-ron. Hỏi là trả lời ngay,
không cần tra cứu. Vấn đề: khi nhớ nhầm, anh ta vẫn trả lời rất tự tin, và **không có cách nào biết
anh ta đang nhớ nhầm**.

**RAG** = đưa cho anh ta cả thư viện, nhưng bắt buộc: *"Trước khi trả lời, phải mở đúng trang, đọc
đúng điều, rồi trích dẫn ra."* Kiến thức nằm *ngoài* mô hình, trong cơ sở dữ liệu. Câu trả lời nào
cũng kèm nguồn — người đọc mở link ra là đối chiếu được.

```
FINE-TUNING                         RAG (cách dự án này làm)
─────────────                       ────────────────────────
câu hỏi                             câu hỏi
   │                                   │
   ▼                                   ▼
[mô hình đã học thuộc luật]         [tìm trong 1,86 triệu đoạn luật]
   │                                   │
   ▼                                   ▼
câu trả lời                         8 đoạn luật liên quan nhất
(không biết lấy từ đâu)                │
                                       ▼
                                    [mô hình viết câu trả lời DỰA TRÊN 8 đoạn đó]
                                       │
                                       ▼
                                    câu trả lời + [1][2][3] + link văn bản gốc
```

### 2.2 Ba lý do RAG là lựa chọn đúng cho pháp luật

**(a) Phải trích dẫn chính xác được.** Một câu trả lời pháp lý không kèm "Điều 111 Luật Doanh nghiệp
59/2020/QH14" thì gần như vô dụng — người dùng không kiểm chứng được, không mang đi dùng được.
RAG cho trích dẫn *theo cấu trúc*: hệ thống biết chắc đoạn văn bản đó đến từ chunk nào, thuộc văn bản
nào, và trả kèm URL gốc trên thuvienphapluat.vn. Mô hình fine-tune chỉ "nhớ" ra một chuỗi trông giống
số hiệu văn bản — và đó chính là kiểu nhớ dễ sai nhất.

**(b) Luật thay đổi liên tục.** Luật Đất đai 2024 thay Luật Đất đai 2013; nghị định sửa đổi ra hàng
tháng. Với RAG, cập nhật = chạy lại `ingest` + `embed`, vài giờ là xong. Với fine-tuning, cập nhật =
huấn luyện lại toàn bộ mô hình, và kiến thức cũ **không tự biến mất** — mô hình vẫn có thể trả lời
theo bản luật đã hết hiệu lực.

**(c) Không được bịa.** Đây là lý do quan trọng nhất, và mục 2.4 nói riêng về nó.

### 2.3 Nếu muốn fine-tune thật thì cần gì

Để so sánh cho công bằng, đây là những gì một dự án fine-tuning thực sự sẽ cần — và những thứ này
dự án **chưa** có:

| Cần gì | Thực tế |
|---|---|
| **Dữ liệu gán nhãn** — hàng chục nghìn cặp (câu hỏi, câu trả lời chuẩn có trích dẫn đúng), do người am hiểu luật soạn hoặc kiểm duyệt | Corpus hiện có là văn bản luật thô + 109.543 bài hỏi đáp chưa qua kiểm duyệt cho mục đích huấn luyện. Chi phí soạn bộ này chưa được ước tính trong dự án. |
| **GPU** — fine-tune một mô hình 8B cỡ LoRA cần GPU nhiều VRAM; full fine-tune cần hạ tầng nhiều GPU | Máy đang dùng là MacBook M4 / 16 GB RAM dùng chung cho cả hệ thống. Chưa đo được thời gian/chi phí fine-tune trên cấu hình này. |
| **Chi phí** — thuê GPU cloud theo giờ, cộng nhiều vòng thử-sai vì lần đầu hiếm khi ra kết quả tốt | Chưa ước tính. |
| **Bộ đánh giá đủ lớn** để biết bản fine-tune có tốt hơn bản gốc không | Hiện chỉ có 10 câu (xem mục 5 và 7). |

### 2.4 Vì sao fine-tuning KHÔNG giải quyết được việc bịa số điều luật

Đây là điểm dễ hiểu nhầm nhất, nên nói kỹ.

Mô hình ngôn ngữ về bản chất là bộ máy **đoán chữ tiếp theo dựa trên xác suất**. Khi nó viết
"theo Điều …", phần tiếp theo được chọn vì *nghe có vẻ đúng trong ngữ cảnh*, chứ không vì mô hình
có cơ chế kiểm tra "Điều này có thật không". Fine-tuning chỉ làm cho các xác suất đó *nghiêng* về
phía văn phong và nội dung pháp luật — nó **làm cho lỗi sai trông thuyết phục hơn, chứ không làm
lỗi sai biến mất**.

Cụ thể hơn:

- Số hiệu văn bản ("100/2015/QH13") và số Điều là những chuỗi **tuỳ ý, không có quy luật ngữ nghĩa**.
  Không có gì trong cách hành văn giúp mô hình suy ra được Điều 134 hay Điều 135 mới là điều đúng.
  Chính bộ đánh giá của dự án cũng cho thấy đây là kiểu nhầm phổ biến nhất (mục 5).
- Fine-tuning nén hàng triệu đoạn văn bản vào một lượng tham số cố định — **quá trình nén đó
  làm mất chi tiết**, và chi tiết bị mất đầu tiên luôn là những con số hiếm gặp.
- Quan trọng nhất: sau khi fine-tune, **vẫn không có cách nào kiểm tra câu trả lời**. Còn với RAG,
  mỗi khẳng định đều gắn với một đoạn văn bản cụ thể — sai hay đúng đều **nhìn thấy được ngay**.

Vì vậy dự án chọn hướng ngược lại: thay vì cố làm mô hình nhớ giỏi hơn, **cấm mô hình dựa vào trí
nhớ**. Prompt hệ thống trong `luatvn/rag.py` viết rõ:

> *"Mỗi khẳng định pháp lý phải kèm số nguồn dạng [1], [2]. Không có nguồn thì không khẳng định."*
> *"TUYỆT ĐỐI không bịa số hiệu văn bản, số Điều, số Khoản hay mức phạt. Chỉ dùng đúng những gì có
> trong nguồn."*
> *"Nếu các nguồn không đủ để trả lời, hãy nói thẳng […]"*

Cộng thêm `temperature = 0.2` (mức "sáng tạo" thấp, bám sát văn bản) trong `luatvn/llm.py`.

**Khi nào fine-tuning mới đáng làm** — xem mục 7.

---

## 2bis. Ba mô hình local đang dùng

Hệ thống không dùng một mô hình mà **ba**, mỗi cái làm một việc khác nhau. Tất cả đều chạy
trên máy, không gọi API bên ngoài. Số liệu dưới đây đo trực tiếp trên MacBook M4 / 16 GB.

| | **qwen3:8b**<br>viết câu trả lời | **multilingual-e5-small**<br>tìm theo ngữ nghĩa | **Vietnamese_Reranker**<br>chấm lại độ liên quan |
|---|---|---|---|
| Kiến trúc | Qwen3 | BERT | XLM-RoBERTa large |
| Tham số | 8,2 tỷ | 118 triệu | ~560 triệu |
| Lượng tử hoá | Q4_K_M | fp16 | fp16 |
| Ngữ cảnh tối đa | 40.960 token | 512 token | 8.194 token |
| Đang cấu hình dùng | 16.384 token | 512 token | 512 token |
| Chiều vector xuất ra | — | **384** | — (chỉ xuất 1 điểm số) |
| Dung lượng đĩa | 5,2 GB | 470 MB | 2,1 GB |
| RAM khi chạy | **6,3 GB** (100% GPU) | ~0,24 GB | ~1,1 GB |
| Giấy phép | Apache 2.0 | MIT | Apache 2.0 |
| Chạy qua | Ollama | sentence-transformers / MPS | transformers / MPS |

**Vì sao ba mô hình chứ không phải một.** Mỗi việc có một đòi hỏi khác nhau:

- `e5-small` phải chạy **1,11 triệu lần** lúc dựng chỉ mục, nên bắt buộc phải nhẹ.
- `Vietnamese_Reranker` chỉ chạy **32 lần mỗi câu hỏi**, nên nặng cũng không sao —
  và chính nó tạo ra phần lớn chất lượng (mục 4.4).
- `qwen3:8b` không đi tìm gì cả, nó chỉ đọc những đoạn đã được chọn sẵn rồi viết lại
  thành câu trả lời tiếng Việt có trích dẫn.

### Yêu cầu phần cứng

| Hạng mục | Cần | Máy đang dùng |
|---|---|---|
| RAM | 16 GB (mức tối thiểu thực tế) | 16 GB — vừa đủ, không dư |
| Đĩa trống | ~18 GB | đã dùng hết |
| GPU | Apple Silicon (MPS) hoặc NVIDIA CUDA | M4 |
| Phần mềm | Python 3.12, Ollama, SQLite có FTS5 | đã cài |

Chia nhỏ 18 GB: CSDL 7,3 + chỉ mục vector 1,7 + `qwen3:8b` 5,2 + hai mô hình nhúng/rerank 2,6
+ thư viện Python 1,3.

Chạy trên CPU không GPU **vẫn được**, nhưng bước sinh vector sẽ chậm khoảng 10 lần và tốc độ
viết câu trả lời rơi xuống 3–5 token/giây — dùng thực tế rất khó chịu.

### Tốc độ một lượt hỏi (đo thật, câu hỏi mới để không trúng cache)

| Giai đoạn | Thời gian |
|---|---|
| Truy xuất (BM25 + vector + rerank) | 2,7 – 6,9 giây |
| Viết câu trả lời | 57 – 78 giây |
| **Tổng** | **59 – 85 giây** |

Nút thắt là bước viết câu trả lời: `qwen3:8b` sinh **15,8 token/giây** trên M4, nhân với câu
trả lời dài 400–900 token. Ba cách rút ngắn, theo thứ tự hiệu quả:

1. **Đổi sang `qwen3:4b`** — nhanh khoảng gấp đôi. Mô hình nhỏ vẫn đủ dùng vì nó không cần
   *biết* luật, chỉ cần đọc lại đoạn luật đã được đưa sẵn vào ngữ cảnh.
2. **Giới hạn độ dài câu trả lời trong prompt** — hiện chưa giới hạn nên mô hình viết khá dài.
3. **Giảm `num_ctx` 16.384 → 8.192** — đỡ tốn bộ nhớ KV-cache.

Lúc đo, hệ thống còn khoảng 20% bộ nhớ trống và đã có hiện tượng ghi bộ nhớ ra đĩa (pageout).
16 GB đủ chạy nhưng không dư; mở thêm nhiều ứng dụng nặng cùng lúc sẽ làm tốc độ tụt.

---

## 3. Đường đi của dữ liệu

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  HuggingFace — 3 bộ dữ liệu của tmquan                                   │
 │    vbpl  573.851 văn bản + toàn văn      (2,96 GB parquet, 96 shard)     │
 │    hdpl  109.543 bài hỏi đáp             (0,35 GB)                        │
 │    tnpl   17.091 thuật ngữ               (~0,00 GB)                       │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  │  ① download.py  (~2 phút)
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ② ingest.py — nạp SQLite + tách theo Điều  (~7 phút)                    │
 │     docs   573.851 dòng metadata (TOÀN BỘ corpus)                        │
 │     chunks 1.859.336 đoạn theo Điều (117.257 văn bản có toàn văn)        │
 │     qa / terms                                                            │
 │     ③ chỉ mục BM25/FTS5 trên cả 4 bảng (fts_chunks: 72,1 giây)           │
 │     ④ bảng token_df — tần suất token, để lọc từ quá phổ biến            │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  │  ⑤ enrich.py  (~10 giây)
                                  │     trích tiêu đề + số hiệu THẬT từ phần mở đầu
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⑥ embed.py — sinh vector  (2 giờ 06 phút)                               │
 │     terms   17.091 vector   (0,7 phút  · 371 mục/giây)                   │
 │     qa     109.543 vector   (2,7 phút  · ~680 mục/giây)                  │
 │     chunks 986.292 vector   (121,5 phút · 135 mục/giây)                  │
 │     ⑦ chỉ mục IVF_PQ trên 1.112.926 vector, 1.054 phân vùng (0,9 phút)   │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
                            [ SẴN SÀNG PHỤC VỤ ]
```

### Lúc người dùng hỏi một câu

```
 "Người lao động đơn phương chấm dứt HĐLĐ phải báo trước bao nhiêu ngày?"
                                  │
          ┌───────────────────────┴───────────────────────┐
          │  5 nguồn truy xuất chạy SONG SONG (5 luồng)    │
          ├───────────────────────────────────────────────┤
          │  BM25 → chunks   (top 40)   ← khớp từ khoá     │
          │  BM25 → qa       (top 40)                      │
          │  BM25 → terms    (top 15)                      │
          │  BM25 → docs     (top 15)   ← phủ CẢ 573.851   │
          │  Vector → LanceDB(top 40)   ← khớp ý nghĩa     │
          └───────────────────────┬───────────────────────┘
                                  ▼
                  ⑧ RRF — hợp nhất 5 bảng xếp hạng theo THỨ HẠNG
                                  ▼
                  ⑨ khử trùng lặp — giữ bản đáng trích dẫn nhất
                                  ▼
                  ⑩ cross-encoder tiếng Việt chấm lại 32 ứng viên
                                  ▼
                  ⑪ 8 đoạn tốt nhất → prompt (tối đa 24.000 ký tự)
                                  ▼
                  ⑫ Qwen3-8B qua Ollama, num_ctx 16.384, temperature 0,2
                                  ▼
          câu trả lời + [1][2][3] + link tới thuvienphapluat.vn
```

Toàn bộ chuỗi này mất **4,2 giây** mỗi truy vấn (đo trên bộ eval, mục 5).

**Giải nghĩa nhanh các thuật ngữ:**

| Thuật ngữ | Nghĩa trong dự án này |
|---|---|
| **chunk** | một mẩu văn bản đủ nhỏ để đưa vào mô hình — ở đây mỗi chunk là **một Điều luật** |
| **BM25 / FTS5** | tìm kiếm theo **từ khoá** (giống Google thời đầu). FTS5 là bộ tìm kiếm toàn văn có sẵn trong SQLite |
| **embedding / vector** | biến một đoạn văn thành dãy 384 con số; hai đoạn ý nghĩa gần nhau thì hai dãy số gần nhau → tìm theo **ý nghĩa**, không cần trùng từ |
| **bi-encoder** | mô hình mã hoá câu hỏi và đoạn văn **riêng rẽ** thành vector → rất nhanh, có thể chạy trước cho cả triệu đoạn |
| **cross-encoder / rerank** | mô hình đọc **cặp** (câu hỏi + đoạn văn) cùng lúc rồi chấm điểm → chính xác hơn nhiều nhưng đắt, chỉ dùng cho vài chục ứng viên cuối |
| **RRF** | Reciprocal Rank Fusion — cách gộp nhiều bảng xếp hạng chỉ dựa trên **vị trí**, không cần so điểm |
| **ANN / IVF_PQ** | chỉ mục giúp tìm vector gần nhất mà không phải quét hết 1,11 triệu vector |

---

## 4. Các quyết định kỹ thuật quan trọng và lý do

### 4.1 Bỏ qua config `embeddings` 28,4 GB có sẵn trên HuggingFace

Ba bộ dữ liệu gốc **đã kèm sẵn embedding**, tổng **28,4 GB**, sinh bằng Nemotron-8B với **4.096 chiều**.
Nghe thì có vẻ "dùng luôn cho nhanh". Dự án **bỏ hẳn**, tự sinh vector **384 chiều**.

| | Embedding có sẵn | Tự sinh (đang dùng) |
|---|---|---|
| Mô hình | Nemotron-8B | `intfloat/multilingual-e5-small` (118M) |
| Số chiều | 4.096 | **384** (nhẹ hơn ~10 lần mỗi vector; cả chỉ mục nhẹ hơn ~17 lần) |
| Dung lượng tải về | **28,4 GB** | 0 (sinh tại chỗ) |
| Chỉ mục kết quả | chưa đo | **1,7 GB** |

Lý do quyết định (theo docstring `luatvn/download.py` và README):

1. **Phải khớp mô hình lúc truy vấn.** Muốn dùng embedding có sẵn thì lúc người dùng hỏi, câu hỏi
   *cũng* phải được nhúng bằng đúng Nemotron-8B. Tức là phải nạp một mô hình 8B vào RAM **chỉ để mã
   hoá câu hỏi**, trên máy 16 GB đã phải dành RAM cho Qwen3-8B sinh câu trả lời. Vector nhúng bằng
   mô hình A mà truy vấn bằng mô hình B là **vô nghĩa** — hai không gian vector khác nhau hoàn toàn.
2. **Ổ cứng.** Docstring `luatvn/db.py` ghi rõ ràng buộc thiết kế: *"ổ cứng còn ~51 GB"*.
   28,4 GB embedding + 3,1 GB dữ liệu thô là hết hơn nửa dung lượng còn lại.
3. **Không mất gì đáng kể.** Chất lượng thật do **reranker** quyết định chứ không phải vòng nhúng
   đầu tiên (xem 4.4 và bảng ablation mục 5). Vòng 1 chỉ cần đủ tốt để đưa đáp án đúng vào top-40.

> Ghi chú trung thực: dự án **chưa đo** chất lượng của phương án dùng embedding Nemotron-8B có sẵn,
> nên không thể khẳng định phương án 384 chiều tốt hơn về *độ chính xác* — chỉ khẳng định được là
> nó khả thi trên phần cứng hiện có, còn phương án kia thì không.

### 4.2 Chunk theo **Điều**, không cắt theo độ dài

Cách làm thông thường trong RAG là cắt văn bản thành mẩu ~500 hoặc ~1.000 ký tự. Dự án **không** làm vậy.

**Lý do (docstring `luatvn/chunk.py`):** câu trả lời pháp lý phải trích dẫn được "Điều x Khoản y".
Cắt mù theo ký tự sẽ **làm đứt đôi một Điều** — nửa sau không còn biết mình thuộc Điều nào, và trích
dẫn sẽ sai. Ở đây mỗi đoạn **luôn nằm gọn trong một Điều và mang theo số Điều của nó**.

Cách thực hiện và các số đo:

- **Bắt tiêu đề Điều bằng regex neo đầu dòng.** Dữ liệu thật (`vn_text`) là text thuần, xuống dòng
  cứng ngay giữa câu, nhưng tiêu đề `Điều N.` **luôn đứng đầu dòng**. Neo đầu dòng để tránh bắt nhầm
  các tham chiếu giữa câu như "Căn cứ Điều 5 Luật Doanh nghiệp".
- **Bắt buộc có dấu chấm/hai chấm ngay sau số.** Đây là ràng buộc nhỏ nhưng đo được hiệu quả rõ —
  thử trên 1.500 văn bản tier ≤ 1, lấy trường `num_articles` trong metadata làm chuẩn:

  | | Không có ràng buộc | **Có ràng buộc dấu chấm** |
  |---|---|---|
  | Khớp chính xác số Điều | 53,3% | **58,1%** |
  | Bắt thừa (nhận nhầm tham chiếu thành tiêu đề) | 24,0% | **18,7%** |

- **Điều quá dài thì cắt tiếp theo Khoản**, trần `MAX_CHUNK_CHARS = 2.400` ký tự; phần thứ 2 trở đi
  được **gắn lại tiêu đề Điều** (`[tiếp] Điều 5. …`) để đoạn vẫn tự đứng được về ngữ cảnh.
- **Nhận diện văn bản ban hành kèm theo.** Quy chế/Điều lệ đính kèm có hệ thống đánh số riêng
  (lại bắt đầu từ Điều 1). Mã tăng bộ đếm `book` **chỉ khi số Điều thực sự reset về 1–2**, không
  tăng với mọi lần tụt số — một lần bắt nhầm sẽ làm vỡ toàn bộ đánh số của văn bản.
- **Cắt bỏ phụ lục/biểu mẫu ở đuôi** (bảng biểu dài, giá trị tra cứu thấp) và chặn
  `MAX_DOC_CHARS = 400.000` ký tự cho văn bản dài bất thường.

**Kết quả đo:** bộ tách khớp chính xác `num_articles` ở **58%** số văn bản, **lệch tuyệt đối trung vị
bằng 0** (nghĩa là ở đa số văn bản còn lại thì cũng chỉ lệch một vài Điều, không lệch hệ thống).
Phần sai chủ yếu là văn bản có quy chế ban hành kèm theo.

### 4.3 Lai **BM25 + vector**, không dùng thuần vector

Đây là quyết định kiến trúc quan trọng nhất, và bảng ablation ở mục 5 chứng minh nó bằng số.

**Vì sao không thuần vector** (docstring `luatvn/retrieve.py` và README):

| Loại câu hỏi | Vector | BM25 |
|---|---|---|
| "Điều 5 **Luật Doanh nghiệp 2020**", "**Nghị định 100/2019**" | ✗ rất kém — embedding không phân biệt được các chuỗi định danh gần giống nhau | ✓ bắt chuẩn |
| "**bị sa thải**" ↔ "**đơn phương chấm dứt hợp đồng lao động**" | ✓ hiểu là cùng ý | ✗ mù hoàn toàn vì không trùng chữ nào |

Hai lớp **bù nhau chứ không thay nhau**. Câu hỏi pháp luật thực tế trộn lẫn cả hai kiểu.

**Vì sao gộp bằng RRF chứ không cộng điểm.** Điểm BM25 và điểm cosine của vector **không cùng thang đo**.
Chuẩn hoá lại để cộng thì luôn phải chỉnh tay theo từng truy vấn. RRF chỉ nhìn **thứ hạng**
(đứng thứ 1, thứ 2, thứ 3…) nên miễn nhiễm với vấn đề đó. Hằng số `RRF_K = 60`.

**Năm nguồn chạy song song, mỗi luồng một kết nối SQLite riêng.** Chi tiết dễ bỏ sót nhưng quan
trọng (docstring `luatvn/db.py`): một kết nối SQLite **tự tuần tự hoá** các câu lệnh, nên nếu 5 tác
vụ dùng chung một kết nối thì "song song" không nhanh hơn chút nào. `db.thread_conn()` cấp cho mỗi
luồng một kết nối chỉ-đọc riêng.

**Nguồn `docs` phủ toàn bộ corpus.** Nhánh BM25 thứ tư tra theo *tiêu đề và số hiệu* của cả
**573.851** văn bản — kể cả 456.594 văn bản chưa có toàn văn trong CSDL. Nhờ đó, hỏi về một văn bản
cấp tỉnh vẫn ra được định danh và link gốc, dù không có nội dung để trích.

Nhánh này có thêm hai lớp ưu tiên chồng lên điểm BM25 (`retrieve.bm25_docs`), vì `tier` không tách
được Luật với Nghị định (cả hai đều tier 0):

| Tín hiệu | Điều chỉnh | Lý do |
|---|---|---|
| Cấp hiệu lực `tier` | hệ số `1 + 0,18 × (3 − tier)` | văn bản cấp cao xếp trước |
| Loại văn bản khớp câu hỏi | `+0,12` | tín hiệu phụ, tiêu đề đã chứa sẵn loại văn bản |
| Năm ban hành (khi câu hỏi **không** nêu năm) | `+0,006/năm`, trần 40 năm | hỏi "Luật Doanh nghiệp" gần như chắc chắn muốn bản đang áp dụng |
| Văn bản đã hết hiệu lực | `−0,30` | đẩy xuống nhưng vẫn giữ |
| Tiêu đề bắt đầu bằng "sửa đổi/bãi bỏ/đính chính/hướng dẫn" | `−0,28` | người hỏi tên luật hầu như luôn muốn văn bản gốc |

### 4.4 Bi-encoder **nhỏ** + cross-encoder **lớn**

Đây là chỗ dễ làm ngược nhất: bản năng bảo "dùng mô hình nhúng mạnh nhất có thể". Dự án làm ngược lại.

**Đo thực tế trên M4/16 GB** (docstring `luatvn/config.py` + README):

| Mô hình nhúng (bi-encoder) | Kích thước | Tốc độ benchmark | Thời gian nhúng 1,11 triệu đoạn |
|---|---|---|---|
| `AITeamVN/Vietnamese_Embedding` (bge-m3) | 568M | 20,6/giây | ~15 giờ |
| `intfloat/multilingual-e5-base` | 278M | 67,9/giây | ~4,5 giờ |
| **`intfloat/multilingual-e5-small`** ✔ | 118M | 217,1/giây | **~2,2 giờ** |

> Lưu ý về con số: 217/giây là **tốc độ benchmark**. Trên dữ liệu thật, các chunk luật dài hơn nên
> tốc độ ổn định thực tế là **135 mục/giây** cho chunk (986.292 mục trong 121,5 phút),
> ~680/giây cho hỏi đáp, 371/giây cho thuật ngữ — đúng như log `data/embed.log` ghi lại.

**Logic của lựa chọn:**

```
   1,86 triệu đoạn
        │
        ▼  vòng 1: bi-encoder e5-small (nhanh, chạy trước cho CẢ triệu đoạn)
        │          nhiệm vụ duy nhất: đưa đáp án đúng vào top-40
        ▼
   ~40 ứng viên  ──► RRF ──► 32 ứng viên
        │
        ▼  vòng 2: cross-encoder AITeamVN/Vietnamese_Reranker
        │          đọc từng CẶP (câu hỏi, đoạn) → chấm điểm chính xác
        │          chỉ 32 cặp/truy vấn → chi phí không đáng kể
        ▼
     8 đoạn cuối cùng
```

Mô hình nhỏ chỉ cần **đủ tốt để lọt vào top-40**; **chất lượng thật do cross-encoder quyết định**.
Bảng ablation mục 5 xác nhận: bỏ reranker, MRR rơi từ **0,650 xuống 0,195**.

Ba tối ưu nhỏ đã đo được ở lớp mô hình (`luatvn/models.py`, `luatvn/retrieve.py`, `luatvn/api.py`):

| Tối ưu | Hiệu quả đo được |
|---|---|
| Chạy fp16 trên MPS (GPU tích hợp Apple) thay vì fp32 | nhanh hơn ~**20%**, giảm **một nửa** bộ nhớ |
| Cắt đoạn còn 800 ký tự khi rerank (thay vì 2.000 ≈ 512 token) | 2.000 ký tự mất **5,2 giây** cho 50 cặp; 800 ký tự vẫn đủ ngữ cảnh mà nhanh hơn nhiều lần |
| Sắp cặp theo độ dài trước khi chia lô | đệm (padding) tính theo cặp dài nhất trong lô — trộn ngắn với dài làm phần lớn phép tính chạy trên token đệm vô nghĩa |
| Nạp sẵn model lúc khởi động server | tránh cho request **đầu tiên** phải chờ thêm ~**18 giây** (embedder 12,4s + reranker 5,7s) |

Một chi tiết nhỏ nhưng dễ mất chất lượng: mô hình e5 **bắt buộc** tiền tố `"query: "` cho câu hỏi và
`"passage: "` cho đoạn văn bản. Docstring `config.py` ghi rõ: **sai tiền tố là mất ~10% chất lượng**.

Và một bài học vận hành đã được sửa: trước đây lỗi của reranker bị **nuốt im lặng**, hệ thống âm
thầm chạy không có reranker mà nhìn kết quả rất khó nhận ra. Giờ `retrieve.rerank` ghi cảnh báo rõ
ràng rồi mới rơi về thứ hạng RRF.

### 4.5 Lọc token có tần suất cao khỏi truy vấn BM25

Đây là quyết định thuần về **tốc độ**, và là chênh lệch lớn nhất trong toàn hệ thống.

**Vấn đề.** FTS5 phải chấm điểm BM25 cho **mọi dòng khớp**. Từ "điều" — sau khi qua tokenizer thành
`đieu` — có mặt ở **97,4% số đoạn**. Một truy vấn chứa nó buộc SQLite duyệt gần như toàn bộ
1,86 triệu đoạn.

**Đo trên bảng 1,86 triệu đoạn** (giới hạn 40 kết quả, có JOIN sang `docs`) — docstring `retrieve.fts_queries`:

| Kiểu truy vấn | Thời gian |
|---|---|
| OR các token đơn | **14,6 giây** → không dùng được, bỏ hẳn |
| OR các cụm 2 từ | 1,1 giây (lên tới **13,5 giây** với cụm phổ biến) |
| AND các token đơn | **0,17 giây** |

**Giải pháp — hai nhánh, đối xử khác nhau vì chi phí khác hẳn nhau:**

1. **Nhánh AND (chạy trước, ưu tiên độ chính xác):** giữ **nguyên mọi token**. Bản thân phép AND
   đã đủ chọn lọc nên nhanh bất kể token có phổ biến hay không — và giữ đủ token thì BM25 mới bắt
   đúng được định danh ("168", "2024").
2. **Nhánh OR (dự phòng, mở rộng recall):** chỉ lấy các **cụm 2 từ có ít nhất một token hiếm**,
   tối đa **4 cụm** (`MAX_OR_PHRASES`). Độ dài posting list của một cụm bị chặn bởi token hiếm hơn
   trong cụm, nên xếp các cụm theo tần suất tăng dần rồi cắt là vừa rẻ nhất vừa nhiều thông tin nhất.
   Không chặn thì một truy vấn toàn từ phổ biến ("điều kiện thành lập công ty") ngốn **13,5 giây**.

Hàm `_fts` **dừng sớm** khi đã đủ kết quả, nên nhánh OR thường không phải chạy.

**Hai chi tiết hạ tầng để cái này hoạt động:**

- **Bảng `token_df`** (dựng lúc `ingest`) lưu tần suất của các token xuất hiện ở > 2.000 đoạn.
  Ngưỡng loại: token có mặt ở hơn **35%** số đoạn (`MAX_DF_RATIO = 0.35`).
- **Phải tokenize câu hỏi bằng chính tokenizer của chỉ mục.** Không thể đoán bằng Python:
  tokenizer `unicode61 remove_diacritics 2` **giữ nguyên chữ "đ" nhưng bỏ dấu các nguyên âm**,
  nên "điều" thành `đieu` chứ **không** phải `dieu`. Đoán sai thì tra tần suất trượt và toàn bộ
  việc lọc mất tác dụng. Mã tạo một bảng FTS5 tạm dùng đúng cấu hình tokenizer để tách token truy vấn.

Ngoài ra còn một danh sách **stopword tiếng Việt đã bỏ dấu** (`la, va, cua, co, đuoc, …`) để loại
các từ nối vô nghĩa khỏi truy vấn.

### 4.6 Một số quyết định đáng nhắc khác

**`enrich` — trích tiêu đề thật.** Trường `title` trong dataset gốc là **slug ASCII không dấu** suy
ra từ URL ("Nghi dinh 73 2016 ND CP huong dan Luat Kinh doanh bao hiem"), và `doc_number` bị nối
thêm cả slug ("20/2017/TT-BYT-HUONG-DAN-LUAT-DUOC-54-2017-ND-CP…"). Dùng trực tiếp thì trích dẫn xấu
và — quan trọng hơn — **tiêu đề không dấu bị nhúng kèm mỗi chunk sẽ làm nhiễu vector**.
Bước `enrich` lấy khối tiêu đề in hoa nằm giữa dòng loại văn bản và chữ "Căn cứ".
**Kết quả: trích được tiêu đề tiếng Việt cho 86,4% văn bản có toàn văn**, trong ~10 giây.

Bước này còn ghép loại văn bản vào trước tiêu đề (`full_title`), vì nếu không thì Luật Doanh nghiệp
mang tên "Doanh nghiệp" — và khi tra "Luật Doanh nghiệp", **nghị định hướng dẫn chứa nguyên cụm đó
trong tiêu đề sẽ thắng chính bộ luật**. Cùng lý do, cột `doc_type` **bắt buộc** phải nằm trong chỉ
mục `fts_docs`. Và `infer_year` suy năm ban hành từ số hiệu khi metadata để trống — Bộ luật Lao động
45/2019/QH14 là một ví dụ thật bị thiếu năm, mà thiếu năm thì chính bộ luật lại thua nghị định
hướng dẫn nó trong xếp hạng.

**Khử trùng lặp giữ bản đáng trích dẫn nhất.** Corpus có nhiều bản sao của cùng một quy định:
văn bản hợp nhất, bản đăng lại. Nếu chỉ "giữ bản gặp trước" thì Điều 111 Luật Doanh nghiệp 2020
có thể bị thay bằng bản hợp nhất mang tiêu đề slug không dấu — cùng nội dung nhưng trích dẫn kém hẳn.
`_dedupe` giữ **vị trí** của lần gặp đầu nhưng thay **nội dung** bằng bản có thứ bậc pháp lý cao nhất
(Hiến pháp → Bộ luật/Luật → Pháp lệnh → Nghị định → Nghị quyết → Thông tư → Văn bản hợp nhất → Quyết định).

**Tokenizer bỏ dấu (`remove_diacritics 2`).** Người dùng gõ "co phan" vẫn khớp "cổ phần" — tiếng
Việt không dấu là kiểu gõ rất phổ biến. Đây là lựa chọn **thiên về recall**; độ chính xác do BM25 +
reranker xử lý ở lớp trên.

**Mọi job dài đều dừng-được-chạy-tiếp.** `ingest` ghi từng shard parquet đã nạp, `embed` ghi con trỏ
sau mỗi lô, cả hai vào bảng `build_state`. Cần thiết vì vbpl có **96 shard / 2,96 GB**, không thể nạp
hết vào RAM 16 GB một lúc; và `embed` chạy hơn 2 giờ liền — tắt máy hay Ctrl-C không mất công đã làm.

**Không lưu lặp toàn văn.** Bản đầy đủ của một văn bản được **dựng lại bằng cách nối các chunk**
theo thứ tự `seq`. FTS5 dùng chế độ external-content nên cũng không nhân đôi text.
(Đi kèm một cảnh báo vận hành ghi trong mã: **không chạy `VACUUM`** trên CSDL này, vì VACUUM có thể
đánh số lại rowid và làm lệch chỉ mục external-content.)

---

## 5. Chất lượng đo được

### 5.1 Đo bằng cách nào

`eval/gold.json` gồm **10 câu hỏi pháp luật thực tế**, mỗi câu gắn với **đúng một Điều của đúng một
văn bản**. Ví dụ:

| Câu hỏi | Đáp án đúng phải là |
|---|---|
| "Điều kiện thành lập công ty cổ phần" | Điều 111 · Luật Doanh nghiệp 59/2020/QH14 |
| "Tuổi nghỉ hưu của người lao động" | Điều 169 · Bộ luật Lao động 45/2019/QH14 |
| "Các trường hợp thu hồi đất vì mục đích quốc phòng an ninh" | Điều 78 · Luật Đất đai 31/2024/QH15 |
| "Tội cố ý gây thương tích bị phạt thế nào" | Điều 134 · Bộ luật Hình sự 100/2015/QH13 |

Hai chỉ số:

- **Recall@k (R@k)** — "đáp án đúng có nằm trong k kết quả đầu không". R@1 = 0,60 nghĩa là **6/10 câu
  cho đáp án đúng ngay ở vị trí số 1**.
- **MRR** (Mean Reciprocal Rank) — nghịch đảo thứ hạng của đáp án đúng, lấy trung bình.
  Đáp án đứng #1 được 1,0 điểm; đứng #2 được 0,5; đứng #5 được 0,2; không có được 0.
  **1,000 = luôn đứng đầu.** MRR phạt nặng việc đáp án tụt hạng, nên nhạy hơn Recall.

### 5.2 Bảng ablation

"Ablation" = tắt bớt từng thành phần để xem mỗi phần đóng góp bao nhiêu.
Chạy `uv run python eval/run_eval.py --ablate`:

| Cấu hình | R@1 | R@3 | MRR | giây/truy vấn |
|---|---|---|---|---|
| **Đầy đủ (BM25 + vector + rerank)** | **0,60** | **0,70** | **0,650** | 4,2 |
| Bỏ rerank | 0,10 | 0,20 | 0,195 | 1,2 |
| Bỏ vector (chỉ BM25 + rerank) | 0,10 | 0,20 | 0,150 | 2,2 |
| Chỉ BM25, không rerank | 0,10 | 0,10 | 0,100 | 0,6 |

### 5.3 Bảng này nói gì

**(1) Bỏ reranker: MRR rơi từ 0,650 → 0,195 (mất 70%).** Đây là con số quan trọng nhất trong cả dự án.
Nó chứng minh rằng kiến trúc "bi-encoder nhỏ + cross-encoder mạnh" **không phải một lựa chọn tiết kiệm
chấp nhận đánh đổi chất lượng** — cross-encoder chính là thứ tạo ra phần lớn chất lượng.
Đổi lại 3 giây mỗi truy vấn (1,2 → 4,2 giây) để MRR tăng gấp hơn 3 lần là một cái giá rất hời.

**(2) Bỏ vector: MRR rơi xuống 0,150.** Rơi tương đương. Chứng minh trực tiếp cho lập luận ở 4.3:
BM25 một mình không đủ. Chú ý cấu hình này **vẫn có reranker** mà vẫn tệ — vì reranker chỉ xếp lại
được những gì vòng 1 đã tìm ra. **Nếu vòng 1 không tìm thấy đáp án đúng, không lớp nào phía sau cứu được.**

**(3) Hai lớp phải đi cùng nhau.** Chỉ BM25 + không rerank cho MRR 0,100 — tức mỗi lớp riêng lẻ đóng
góp rất ít, giá trị nằm ở **sự kết hợp**. Đây là lý do kiến trúc phức tạp hơn "cắm một mô hình
embedding vào là xong".

**(4) Vì sao R@3 (0,70) phản ánh đúng hơn R@1 (0,60).** Ba câu trượt đều là **trượt sát** — ví dụ hệ
thống trả về **Điều 135** thay vì **Điều 134** Bộ luật Hình sự (hai điều liền kề về cùng nhóm tội).
Với người dùng thật, kết quả nằm ở vị trí 2–3 vẫn hoàn toàn dùng được vì giao diện hiển thị cả 8
nguồn kèm trích dẫn. Nên **R@3 = 0,70 là con số sát với trải nghiệm thực tế hơn.**

**(5) Cảnh báo về cỡ mẫu — đọc kỹ.** 10 câu hỏi là **quá ít** để so sánh các thay đổi nhỏ.
Với n = 10, mỗi câu đúng/sai làm R@1 nhảy nguyên **0,10**. Bảng này đủ tin để kết luận
*"reranker và vector là thiết yếu"* (chênh lệch 3–4 lần, không thể do ngẫu nhiên), nhưng **không đủ**
để kết luận kiểu *"đổi tham số X làm R@1 tăng từ 0,60 lên 0,70"*. Xem mục 7.

---

## 6. Hạn chế đã biết

Trình bày trung thực, lấy nguyên từ README và mã nguồn.

### 6.1 Về độ chính xác của dữ liệu

| Hạn chế | Số đo | Hệ quả thực tế |
|---|---|---|
| Bộ tách Điều không hoàn hảo | khớp chính xác `num_articles` ở **58%** văn bản (lệch tuyệt đối **trung vị = 0**) | Một số văn bản bị tách sai số Điều. Phần sai chủ yếu là văn bản có **quy chế ban hành kèm theo** với hệ thống đánh số riêng. |
| Trích tiêu đề tiếng Việt không phủ hết | **86,4%** văn bản có toàn văn | ~13% còn lại hiển thị theo **slug không dấu** của nguồn — trích dẫn xấu, khó đọc. |
| **Cột `status` rỗng ở ~92% số dòng** | 573.851 văn bản, chỉ **5.836** ghi rõ "Hết hiệu lực" | ⚠️ **Không thể coi `status` là cờ "còn hiệu lực".** Hệ thống chỉ loại được những văn bản ghi rõ đã hết hiệu lực và đẩy chúng xuống cuối; **văn bản không rõ trạng thái vẫn được giữ và vẫn có thể được trích dẫn**. |

> **Đây là hạn chế nghiêm trọng nhất.** Hệ thống **không đảm bảo** điều luật nó trích dẫn còn hiệu
> lực. Vì vậy: **luôn đối chiếu lại với văn bản gốc trước khi sử dụng.** Mỗi nguồn trong giao diện
> đều có liên kết tới trang gốc trên thuvienphapluat.vn.

### 6.2 Về phạm vi phủ

| Lớp tra cứu | Phủ được gì |
|---|---|
| **Từ khoá (BM25)** — tiêu đề & số hiệu | **Toàn bộ 573.851 văn bản** |
| **Từ khoá (BM25)** — toàn văn | **117.257** văn bản (tier 0–1) |
| **Ngữ nghĩa (vector)** | **986.292 đoạn** thuộc văn bản tier 0 còn hiệu lực (Luật, Bộ luật, Pháp lệnh, Nghị định, Thông tư), cộng toàn bộ 109.543 hỏi đáp và 17.091 thuật ngữ |
| **Chưa phủ nội dung** | Văn bản cấp tỉnh (tier 2, **~419.000** văn bản) — hiện chỉ tra được theo metadata |

Phân bố tier (đã kiểm chứng trên mẫu 72.000 dòng): tier 0 ~61.800 · tier 1 ~54.200 ·
tier 2 ~418.700 · tier 3 ~39.200 (công văn, TCVN).

### 6.3 Về chỉ mục vector hiện tại

Vector hiện tại **được nhúng TRƯỚC khi `enrich` chạy**, nên tiền tố gắn vào mỗi đoạn còn là
"Doanh nghiệp" thay vì "Luật doanh nghiệp". Đây là một khoản **nợ kỹ thuật đã biết** —
cách sửa ở mục 7.

### 6.4 Về bộ đánh giá

**10 câu là quá ít.** Đủ để bắt lỗi lớn, không đủ để so sánh các thay đổi nhỏ (xem 5.3 điểm 5).

### 6.5 Về bản chất công cụ

**Đây là công cụ tra cứu, không phải tư vấn pháp lý.** Mỗi câu trả lời đều kết thúc bằng dòng cảnh báo
này, và prompt hệ thống bắt buộc mô hình phải nói thẳng khi nguồn không đủ.

### 6.6 Ghi chú nhỏ về mã nguồn

Một vài docstring còn giữ con số của bản nháp trước (`config.py` nhắc "vòng 1 dùng e5-base",
`download.py` nhắc "tự sinh embedding 1024-d"), trong khi cấu hình **đang chạy thật** là
**e5-small / 384 chiều**. Không ảnh hưởng vận hành, nhưng nên dọn để tránh nhầm về sau.

---

## 7. Hướng phát triển tiếp

Xếp theo tỷ lệ *giá trị thu được / công bỏ ra*.

### Ưu tiên 1 — Dựng lại chỉ mục vector sau khi đã có tiêu đề chuẩn

**Công:** ~2,2 giờ máy, gần như không có công người.
**Được:** sửa nợ kỹ thuật ở 6.3 — mỗi đoạn được nhúng kèm tiêu đề đầy đủ ("Luật doanh nghiệp") thay
vì slug cụt ("Doanh nghiệp"), nên vector khớp đúng hơn với cách người dùng đặt câu hỏi.

```bash
# xoá data/index/lance và các dòng embed_cursor_* trong bảng build_state trước
uv run python -m luatvn.embed
```

### Ưu tiên 2 — Mở rộng bộ đánh giá lên 50–100 câu

**Công:** công người soạn câu hỏi + tra đáp án đúng.
**Được:** **điều kiện tiên quyết cho mọi cải tiến sau này.** Với 10 câu, không thể phân biệt "thay
đổi thật sự tốt hơn" với "may mắn". Với 50–100 câu, mỗi câu chỉ còn ảnh hưởng 1–2% chỉ số, và có thể
tin tưởng tinh chỉnh tham số (ngưỡng `MAX_DF_RATIO`, số ứng viên rerank, trọng số BM25…).

### Ưu tiên 3 — Phủ vector cho tier 1

```bash
LUATVN_MAX_TIER=1 uv run python -m luatvn.embed
```
Thêm ~54.000 văn bản cấp trung ương (quyết định, chỉ thị) vào lớp tra cứu ngữ nghĩa.
Chi phí thời gian thêm: **chưa đo**.

### Ưu tiên 4 — Nâng bi-encoder lên e5-base (nếu có bộ eval lớn xác nhận đáng)

```bash
LUATVN_EMBED_MODEL=intfloat/multilingual-e5-base LUATVN_EMBED_DIM=768 \
  uv run python -m luatvn.embed
```
Recall vòng 1 tốt hơn, đổi lại **~4,5 giờ** dựng thay vì 2,2 giờ và chỉ mục lớn hơn.
**Chỉ nên làm sau Ưu tiên 2** — hiện chưa có cách đo xem nó có thật sự tốt hơn không.

### Ưu tiên 5 — Xử lý vấn đề hiệu lực văn bản

Hạn chế nghiêm trọng nhất (6.1) không sửa được bằng mô hình, mà bằng **dữ liệu**: cần một nguồn
thông tin hiệu lực đáng tin hơn cột `status` hiện đang rỗng 92%. Chưa có phương án cụ thể.

---

### Khi nào fine-tuning mới thực sự đáng làm

Không phải "không bao giờ", nhưng **rõ ràng chưa phải bây giờ**. Fine-tuning chỉ đáng cân nhắc khi
**tất cả** các điều kiện sau đã thoả:

1. **Bộ đánh giá đã đủ lớn (≥ 100 câu) và ổn định.** Không có thước đo thì không biết fine-tune có
   cải thiện gì không — và đây là cách nhanh nhất để đốt nhiều giờ GPU mà không thu được gì.
2. **Lớp truy xuất đã chạm trần.** Hiện R@3 = 0,70, còn nhiều dư địa ở các Ưu tiên 1–3 với chi phí
   rẻ hơn nhiều. Chừng nào truy xuất còn cải thiện được bằng công việc rẻ, chưa nên đụng tới
   fine-tuning.
3. **Đã xác định được lỗi còn lại nằm ở khâu *diễn đạt*, không phải khâu *tìm kiếm*.**
   Đây là ranh giới quyết định:
   - Nếu hệ thống **tìm đúng Điều** mà mô hình **viết câu trả lời dở** (bỏ sót ý, trích dẫn lộn xộn,
     văn phong không hợp) → fine-tuning **có thể** giúp, vì đây là vấn đề về *phong cách và định dạng*
     — đúng thứ fine-tuning làm tốt.
   - Nếu hệ thống **tìm sai Điều** → fine-tuning **hoàn toàn không giúp gì**. Phải sửa ở lớp truy xuất.

4. **Đã có dữ liệu huấn luyện chất lượng.** Hàng chục nghìn cặp (câu hỏi, câu trả lời mẫu có trích
   dẫn đúng) do người am hiểu luật kiểm duyệt. Fine-tune trên dữ liệu chưa kiểm duyệt sẽ dạy mô hình
   học luôn cả lỗi trong dữ liệu.

Và ngay cả khi làm, cần nhớ: **fine-tuning bổ sung cho RAG chứ không thay thế RAG.**
Mô hình đã fine-tune vẫn phải được đưa văn bản luật thật và vẫn bị cấm dựa vào trí nhớ —
vì như đã phân tích ở 2.4, **không có lượng huấn luyện nào làm cho một bộ máy đoán chữ trở nên
đáng tin về số Điều và số hiệu văn bản.** Cách duy nhất để trích dẫn đúng là **đọc từ nguồn**.

---

## Phụ lục — Tham số vận hành

| Tham số | Giá trị | Ở đâu |
|---|---|---|
| Mô hình nhúng | `intfloat/multilingual-e5-small`, 384 chiều, tối đa 512 token | `config.EMBED_MODEL` |
| Mô hình rerank | `AITeamVN/Vietnamese_Reranker`, tối đa 512 token | `config.RERANK_MODEL` |
| Mô hình sinh câu trả lời | `qwen3:8b` qua Ollama, `num_ctx` 16.384 | `config.LLM_MODEL` |
| Tham số sinh | temperature 0,2 · top_p 0,9 · repeat_penalty 1,05 | `llm._payload` |
| Ứng viên BM25 mỗi nguồn | 40 | `config.BM25_TOPK` |
| Ứng viên vector | 40 | `config.VECTOR_TOPK` |
| Hằng số RRF | 60 | `config.RRF_K` |
| Số ứng viên đưa vào rerank | 32 | `config.RERANK_CANDIDATES` |
| Số đoạn cuối vào prompt | 8, tối đa 24.000 ký tự | `config.CONTEXT_TOPK` |
| Trần chunk | 2.400 ký tự (tối thiểu 50) | `config.MAX_CHUNK_CHARS` |
| Trần văn bản | 400.000 ký tự | `config.MAX_DOC_CHARS` |
| Tier tối đa cho vector | 0 (đổi bằng `LUATVN_MAX_TIER`) | `config.VECTOR_MAX_TIER` |
| Tier tối đa cho chunk toàn văn | 1 (đổi bằng `LUATVN_CHUNK_MAX_TIER`) | `ingest.CHUNK_MAX_TIER` |
| Ngưỡng lọc token phổ biến | 35% số đoạn | `retrieve.MAX_DF_RATIO` |
| Chỉ mục vector | LanceDB IVF_PQ, 1.054 phân vùng, metric cosine | `embed.build_ann_index` |

**Nguồn tài liệu:** `README.md`, docstring trong `luatvn/{config,chunk,embed,enrich,retrieve,models,db,ingest,rag,llm}.py`,
`eval/run_eval.py`, `eval/gold.json`, và log build thật trong `data/`.
