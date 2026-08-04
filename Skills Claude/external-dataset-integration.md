---
name: external-dataset-integration
description: Dùng khi cần đưa một dataset pháp luật có sẵn từ bên ngoài (vd. YuITC/Vietnamese-Legal-Documents trên HuggingFace, hoặc bất kỳ nguồn tương tự với schema phẳng không có law_id/hierarchy) vào dự án LegalRAG chatbot — cho 1 trong 3 mục đích: fine-tune embedding/reranker, benchmark retrieval độc lập, hoặc mở rộng corpus sản xuất. KHÔNG dùng skill này để tái cấu trúc chunker.py/parser.py/models.LawChunk theo schema của dataset ngoài — hướng đó đã bị loại bỏ (xem "Quyết định kiến trúc" bên dưới).
---

# External Dataset Integration (E1–E4)

## Quyết định kiến trúc (đọc trước khi code)

**KHÔNG tái cấu trúc `backend/ingestion/chunker.py`, `parser.py`, `models.LawChunk` để khớp schema phẳng của dataset ngoài.** Dataset dạng `(cid, text)` / `(qid, question, cid, context_list)` (như YuITC) không có `law_id`, không có Chương/Mục/Điều/Khoản/Điểm, không có `effective_date`. Đổi code theo hướng đó nghĩa là bỏ chính những thứ đã được thiết kế có chủ đích: rule-based chunking (tránh cắt ngang "trừ trường hợp"), lọc hiệu lực văn bản, hierarchy cho rerank, và tách `article_num` khỏi `aid` (đã tốn công sửa 1 lần vì bug Micro Law F1 = 0.000 do lệch namespace ID).

**Hướng đúng: viết adapter một chiều (dataset ngoài → format nội bộ), không đụng vào pipeline lõi**, và tuỳ mục đích mà đi vào 1 trong 3 nhánh dưới đây — không nhánh nào đòi hỏi sửa `chunker.py`/`parser.py` gốc.

---

## E1 — Adapter nạp dataset ngoài thành `RawArticle`

File mới: `backend/ingestion/external_datasets.py` (KHÔNG sửa `parser.py::load_law_corpus()` — giữ nguyên cho corpus gốc, tránh 2 luồng dữ liệu lẫn lộn logic).

```python
"""
Adapter nạp dataset pháp luật ngoài (schema phẳng, không hierarchy) thành
RawArticle để tái dùng chunker.chunk_articles() sẵn có — KHÔNG viết chunking
mới, KHÔNG sửa parser.py gốc.

Áp dụng cho bất kỳ nguồn nào có shape (id, text) — không riêng YuITC.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from backend.ingestion.parser import RawArticle

_RE_DIEU_NUM = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)


def load_external_corpus_as_articles(
    parquet_path: str | Path,
    source_tag: str,
    text_col: str = "text",
    id_col: str = "cid",
) -> list[RawArticle]:
    """Nạp 1 file corpus.parquet dạng (id, text) thành list[RawArticle].

    - `law_id` được gán DUY NHẤT theo `source_tag` + id gốc (vd
      "ext-yuitc-142820") — KHÔNG cố nhóm nhiều row lại thành 1 "văn bản luật"
      vì dataset nguồn không cho biết văn bản gốc là gì. Hệ quả: mỗi row là
      1 "document" độc lập gồm đúng 1 "article" — chunker.chunk_articles()
      vẫn chạy được bình thường trên input này (không cần sửa).
    - `aid` = id gốc của dataset ngoài, giữ nguyên để có thể truy vết ngược
      lại hàng gốc khi debug, KHÔNG bao giờ coi id này là "article_num".
    - title để trống — chunker._parse_real_article_num() sẽ tự tìm "Điều N."
      trong `body`; nếu không có (rất phổ biến ở dữ liệu ngoài — xem
      cảnh báo bên dưới), article_num sẽ là None và mọi nơi hiển thị đã có
      sẵn fallback về `aid` (xem prompt_builder._display_num()).
    """
    df = pd.read_parquet(parquet_path)
    articles: list[RawArticle] = []
    for row in df.itertuples(index=False):
        raw_id = getattr(row, id_col)
        text = getattr(row, text_col) or ""
        articles.append(
            RawArticle(
                law_id=f"ext-{source_tag}-{raw_id}",
                aid=int(raw_id),
                title="",
                chuong=None,
                muc=None,
                body=text.strip(),
            )
        )
    return articles
```

**Cảnh báo bắt buộc đọc trước khi chạy:** dữ liệu ngoài thường KHÔNG có header "Điều N." ở đầu mỗi đoạn (đã xác nhận trực tiếp trên nhiều dòng mẫu của YuITC — vd `"1. Sử dụng lái xe bảo đảm sức khỏe..."` không có "Điều X." phía trước). Với các row này:
- `chunker._parse_real_article_num()` trả `None` → `article_num=None` → hiển thị fallback về `aid` (số ID nội bộ vô nghĩa với người dùng, giống hệt vấn đề `aid` cũ của ALQAC).
- **Không cố sửa bằng cách suy đoán** — chấp nhận giới hạn này, và cân nhắc lọc bỏ trước khi ingest những row không parse được `article_num` NẾU mục đích là E3 (mở rộng corpus sản xuất, nơi citation hiển thị cho người dùng thật). Với E2/E3-benchmark (không hiển thị cho người dùng), giữ nguyên không sao.

```python
def filter_articles_with_parseable_number(articles: list[RawArticle]) -> list[RawArticle]:
    """Dùng khi mục đích là E3 (mở rộng corpus sản xuất) — loại bỏ trước các
    article không có "Điều N." nhận diện được, tránh citation vô nghĩa lọt
    vào câu trả lời cuối cùng cho người dùng."""
    return [a for a in articles if _RE_DIEU_NUM.search(a.body or a.title or "")]
```

---

## E2 — Index benchmark độc lập (KHÔNG merge vào production)

File mới: `scripts/build_external_eval_index.py`. Mục tiêu: đo recall@k của **chính** `hybrid_search()`/`rerank()` hiện có trên một corpus rộng hơn, dùng gold ngay trong namespace ID của chính dataset ngoài — tránh hoàn toàn vấn đề lệch `aid`/`article_num` vì không so khớp xuyên 2 nguồn.

```python
"""
Build BM25 + Pinecone index RIÊNG cho dataset ngoài, tách biệt namespace với
corpus sản xuất (config.PINECONE_NAMESPACE). Mục đích DUY NHẤT: benchmark
retrieval, không phải để chatbot trả lời thật từ đây.
"""
import argparse
import pickle

from backend import config
from backend.indexing import vector_store
from backend.indexing.bm25_index import BM25Index
from backend.ingestion.chunker import chunk_articles
from backend.ingestion.external_datasets import load_external_corpus_as_articles

EXTERNAL_BM25_PATH = config.DATA_DIR / "external_eval_bm25_index.pkl"
EXTERNAL_PINECONE_NAMESPACE = "external-eval"  # KHÁC config.PINECONE_NAMESPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-parquet", required=True)
    parser.add_argument("--source-tag", default="yuitc")
    parser.add_argument("--rebuild-pinecone", action="store_true")
    args = parser.parse_args()

    articles = load_external_corpus_as_articles(args.corpus_parquet, source_tag=args.source_tag)
    chunks = chunk_articles(articles)

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(EXTERNAL_BM25_PATH)
    print(f"BM25 (external) saved to {EXTERNAL_BM25_PATH}, {len(chunks)} chunks")

    if args.rebuild_pinecone:
        # Namespace RIÊNG — không đụng dữ liệu sản xuất trong cùng index.
        original_ns = config.PINECONE_NAMESPACE
        config.PINECONE_NAMESPACE = EXTERNAL_PINECONE_NAMESPACE
        try:
            count = vector_store.upsert_chunks(chunks)
            print(f"Upserted {count} chunks to Pinecone namespace={EXTERNAL_PINECONE_NAMESPACE}")
        finally:
            config.PINECONE_NAMESPACE = original_ns


if __name__ == "__main__":
    main()
```

Script eval đi kèm (tái dùng `hybrid_search`, chỉ trỏ vào index/namespace ngoài, gold lấy trực tiếp từ `test.parquet` của chính dataset — KHÔNG remap qua bất kỳ bảng ánh xạ nào):

```python
# test/test_external_retrieval_benchmark.py
import pandas as pd
from backend.retrieval.hybrid_search import hybrid_search  # trỏ namespace ngoài khi gọi

def evaluate_recall_at_k(test_parquet_path: str, k: int = 5) -> float:
    df = pd.read_parquet(test_parquet_path)
    recalls = []
    for row in df.itertuples(index=False):
        gold_ids = set(row.cid)  # list gold cid của chính dataset — cùng namespace
        retrieved = hybrid_search(row.question, top_k=k)  # namespace ngoài đã set qua config tạm thời
        pred_ids = {int(c.aid) for c in retrieved}
        if gold_ids:
            recalls.append(len(pred_ids & gold_ids) / len(gold_ids))
    return sum(recalls) / len(recalls) if recalls else 0.0
```

**Nguyên tắc bắt buộc:** không bao giờ so khớp `aid` của index ngoài với `aid` của corpus sản xuất — hai không gian ID hoàn toàn độc lập, dùng chung `hybrid_search()`/`rerank()` là hợp lệ (cùng logic), nhưng dữ liệu và gold luôn phải cùng một namespace.

---

## E3 — Mở rộng corpus sản xuất thật (chỉ khi cần chatbot trả lời được domain rộng hơn dân sự)

Chỉ làm bước này SAU khi đã quyết định (không phải mặc định). Quy trình:

1. `load_external_corpus_as_articles()` (E1) → `filter_articles_with_parseable_number()` để loại citation vô nghĩa → `chunker.chunk_articles()`.
2. Gắn thêm field `source_dataset` vào metadata Pinecone để phân biệt khi audit — sửa nhỏ trong `_chunk_metadata()`:
   ```python
   # backend/indexing/vector_store.py::_chunk_metadata()
   def _chunk_metadata(chunk: LawChunk, extra: Optional[dict] = None) -> dict:
       meta = {
           "law_id": chunk.law_id,
           "aid": chunk.aid,
           "article_num": chunk.article_num if chunk.article_num is not None else -1,
           "level": chunk.level,
           "parent_id": chunk.parent_id or "",
           "breadcrumb": chunk.breadcrumb,
           "text": chunk.text,
           "source_dataset": extra.pop("source_dataset", "primary") if extra else "primary",  # MỚI
       }
       if extra:
           meta.update(extra)
       return meta
   ```
3. Chạy `scripts/build_index.py` như bình thường nhưng với input đã gộp corpus gốc + corpus ngoài đã lọc (KHÔNG viết pipeline ingest song song thứ 2 — dùng đúng 1 đường `build_index.py` cho cả 2 nguồn, chỉ khác ở bước nạp `RawArticle` đầu vào).
4. **Bắt buộc kèm E4** — không được bỏ qua.

## E4 — Đồng bộ phạm vi guardrail (bắt buộc nếu làm E3)

Nếu corpus mở rộng ra ngoài phạm vi dân sự (lao động, chứng khoán, hải quan, y tế, điều lệ Đảng — đã thấy trong dữ liệu mẫu), 2 chỗ phải cập nhật cùng lúc, nếu không sẽ có tình huống corpus có câu trả lời nhưng guardrail/prompt vẫn từ chối vì nghĩ ngoài phạm vi:

```python
# backend/generation/prompt_builder.py — CHAT_SYSTEM_PROMPT dòng đầu
"Bạn là trợ lý tra cứu pháp luật Việt Nam."
# -> nếu mở rộng domain, sửa thành mô tả đúng phạm vi mới, vd:
"Bạn là trợ lý tra cứu pháp luật Việt Nam (dân sự, lao động, doanh nghiệp, hành chính...)."
```

```python
# backend/guardrail.py — _TRIVIAL_OUT_OF_SCOPE_PATTERNS chỉ cần xem lại nếu
# domain mới đủ rộng khiến một số câu hỏi trước đây coi là "chắc chắn ngoài
# phạm vi" (vd câu hỏi về thủ tục hành chính) nay lại nằm TRONG phạm vi mới.
# Không cần đổi nếu domain mới không giao với các pattern hiện có.
```

Kiểm tra bằng `test/test_chatbot.py::evaluate_abstention()` sau khi đổi — nếu tỉ lệ abstention giảm bất thường trên `NEGATIVE_SAFETY_QUERIES` cũ, nghĩa là phạm vi mới đã vô tình "nuốt" một số câu từng bị coi là ngoài phạm vi.

---

## Bảng quyết định nhanh: chọn nhánh nào

| Mục đích | Nhánh | Có sửa index/corpus sản xuất không? | Rủi ro |
|---|---|---|---|
| Cải thiện embedding/reranker | Dùng thẳng `train.parquet` làm cặp positive, không qua E1 | Không | Thấp — chỉ ảnh hưởng model weight, không ảnh hưởng dữ liệu |
| Benchmark retrieval trên domain rộng | E1 (không lọc) → E2 | Không (namespace riêng) | Thấp |
| Chatbot trả lời được domain rộng hơn thật sự | E1 (có lọc) → E3 → E4 bắt buộc | Có | Trung bình — cần theo dõi lại abstention/guardrail |

## Acceptance criteria
- E1: chạy `load_external_corpus_as_articles()` trên 1 sample nhỏ, xác nhận `chunk_articles()` không lỗi và log tỷ lệ `article_num` parse được (kỳ vọng THẤP với dữ liệu dạng YuITC — không phải bug, là đặc điểm dữ liệu nguồn).
- E2: `test_external_retrieval_benchmark.py` chạy ra một con số recall@k cụ thể, không phụ thuộc corpus sản xuất — xoá namespace `external-eval` không ảnh hưởng gì tới namespace chính.
- E3: sau khi ingest, `scripts/build_index.py`'s log "citation fast-path index: N/M articles have a parseable article_num" phải phản ánh đúng tỷ lệ thấp hơn (do phần mở rộng), không được để lẫn lộn coi đây là lỗi ingest.
- E4: `test/test_chatbot.py --skip-generation` (trục abstention) không có regression so với trước khi mở rộng corpus.
