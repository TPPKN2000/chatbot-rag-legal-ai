"""
Central configuration for the LegalRAG chatbot.
All values are overridable via environment variables (.env).

CHATBOT_MIGRATION_PLAN.md §2.2 (removed here): ALQAC Case Content API
budget settings (API_BUDGET_MULTIPLIER, API_HARD_CEILING_MULTIPLIER,
DEFAULT_MAX_API_CALLS_PER_CASE, ALQAC_*), the 4-label schema knobs
(VALID_PREDICTIONS, USE_RATIO_DERIVED_LABEL), and submission paths
(SUBMISSION_OUT_PATH) are gone — there is no "case", no per-case API
budget, and no fixed-label submission file anymore.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set externally


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None else default


# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("LEGALRAG_DATA_DIR", str(ROOT_DIR / "data")))
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
LAW_CORPUS_PATH = Path(_env("LAW_CORPUS_PATH", str(DATA_DIR / "corpus_law_pub.json")))
# Hand-curated small eval set for eval-harness-chatbot skill (retrieval
# recall@k) — NOT the old ALQAC2026_public_test.json (no gold verdict here).
CHATBOT_EVAL_SET_PATH = Path(_env("CHATBOT_EVAL_SET_PATH", str(DATA_DIR / "chatbot_eval_set.json")))

# --- Pinecone (vector store) -------------------------------------------
PINECONE_API_KEY = _env("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = _env("INDEX_NAME", "legalrag-law-corpus")
PINECONE_CLOUD = _env("PINECONE_CLOUD", "aws")
PINECONE_REGION = _env("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = _env("PINECONE_NAMESPACE", "law-corpus")

# --- Models --------------------------------------------------------------
# B0-B4 (coding_plan.md): Nhóm B đề xuất nâng cấp EMBEDDING_MODEL_NAME lên
# Qwen/Qwen3-Embedding-4B (+ RERANKER_MODEL_NAME lên Qwen/Qwen3-Reranker-4B
# cùng cặp). QUYẾT ĐỊNH CÓ CHỦ Ý ở đây: KHÔNG đổi default trực tiếp trong
# code, dù bản patch mẫu trong coding_plan.md có set default mới — vì B4
# của chính tài liệu đó ghi rõ "KHÔNG được merge vào production dù
# VRAM/latency ổn" nếu Recall@5 A/B test (cần data/chatbot_eval_set.json
# soạn trên CORPUS THẬT, xem A3) chưa pass. Đổi default trong config.py
# chính là merge vào production ngay khi deploy — mâu thuẫn với gate đó.
# Cách dùng đúng theo B3 (reindex procedure): override qua env, trỏ
# INDEX_NAME sang index MỚI (vd. legalrag-law-corpus-v2-qwen4b) để giữ
# production index cũ chạy song song cho tới khi A/B test pass:
#   EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-4B
#   EMBEDDING_DIM=1536
#   EMBEDDING_QUERY_INSTRUCTION="Given a Vietnamese legal question, retrieve the most relevant statute passage that answers it."
#   EMBEDDING_USE_MRL_TRUNCATE=true
#   RERANKER_MODEL_NAME=Qwen/Qwen3-Reranker-4B
#   INDEX_NAME=legalrag-law-corpus-v2-qwen4b
# Code (embed.py) đã sẵn sàng cho cấu hình này (instruction-prefix bất đối
# xứng query/doc, MRL truncate_dim) — chỉ còn thiếu bước chạy B3/B4 thật
# trên GPU + corpus thật + Pinecone thật (không có trong sandbox này).
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)
# Instruction-prefix cho model instruction-aware (Qwen3-Embedding). Rỗng
# theo mặc định vì AITeamVN/Vietnamese_Embedding KHÔNG instruction-aware —
# thêm prefix vào model không hỗ trợ sẽ chỉ thêm nhiễu, không lỗi cứng.
EMBEDDING_QUERY_INSTRUCTION = _env("EMBEDDING_QUERY_INSTRUCTION", "")
# MRL (Matryoshka) truncate_dim — chỉ bật khi model hỗ trợ (Qwen3-Embedding).
EMBEDDING_USE_MRL_TRUNCATE = _env("EMBEDDING_USE_MRL_TRUNCATE", "false").lower() == "true"
RERANKER_MODEL_NAME = _env("RERANKER_MODEL_NAME", "AITeamVN/Vietnamese_Reranker")
GENERATION_MODEL_NAME = _env("GENERATION_MODEL_NAME", "Qwen/Qwen3.5-0.8B")
DEVICE = _env("LEGALRAG_DEVICE", "cuda")  # falls back to cpu automatically in models.py
GENERATION_ENABLE_THINKING = _env("GENERATION_ENABLE_THINKING", "false").lower() == "true"
GENERATION_ATTN_IMPL = _env("GENERATION_ATTN_IMPL", "sdpa")
GENERATION_MAX_NEW_TOKENS_DEFAULT = _env_int("GENERATION_MAX_NEW_TOKENS_DEFAULT", 500)
# B3: cap for the final streamed chat answer specifically (kept separate
# from the default above since digest/condense calls want a much smaller
# budget — see CASE_DIGEST_MAX_NEW_TOKENS-equivalent below).
CHAT_ANSWER_MAX_NEW_TOKENS = _env_int("CHAT_ANSWER_MAX_NEW_TOKENS", 600)

# --- Retrieval parameters -------------------------------------------------
BM25_TOP_K = _env_int("BM25_TOP_K", 30)
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 30)
RRF_K = _env_int("RRF_K", 60)  # standard RRF damping constant
RERANK_TOP_K = _env_int("RERANK_TOP_K", 20)
FINAL_LAW_TOP_K = _env_int("FINAL_LAW_TOP_K", 8)

# --- Query transformation ---------------------------------------------------
NER_MODEL_NAME = _env("NER_MODEL_NAME", "NlpHUST/ner-vietnamese-electra-base")
QUERY_DECOMPOSITION_ENABLED = _env("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true"
QUERY_DECOMPOSITION_MAX_SUBQUERIES = _env_int("QUERY_DECOMPOSITION_MAX_SUBQUERIES", 4)
RRF_WEIGHT_STANDARD = _env_float("RRF_WEIGHT_STANDARD", 1.0)
RRF_WEIGHT_AGENT = _env_float("RRF_WEIGHT_AGENT", 2.0)
# D1 (CHATBOT_MIGRATION_PLAN.md): weight for the accent-folded auxiliary BM25
# channel — deliberately lower than RRF_WEIGHT_STANDARD so an accent-folded
# match can surface a result without out-ranking a precise diacritics-aware
# hit on an ordinary (accented) query.
RRF_WEIGHT_FOLDED = _env_float("RRF_WEIGHT_FOLDED", 0.5)
QUERY_REWRITE_MAX_VARIANTS = _env_int("QUERY_REWRITE_MAX_VARIANTS", 2)

# --- Retrieval evaluator loop -----------------------------------------------
# citation-fast-path skill: this threshold now also gates the HARD refusal
# branch in chat_pipeline.py (grounded-chat-generation skill) — if even
# after the evaluator's extra round the best rerank score stays below this,
# the chatbot must refuse rather than let the LLM guess from weak context.
RETRIEVAL_EVALUATOR_ENABLED = _env("RETRIEVAL_EVALUATOR_ENABLED", "true").lower() == "true"
RETRIEVAL_EVALUATOR_SCORE_THRESHOLD = _env_float("RETRIEVAL_EVALUATOR_SCORE_THRESHOLD", 0.75)

# --- Multi-hop cross-reference (conversational-retrieval skill §Bước 3) ----
CROSSREF_ENABLED = _env("CROSSREF_ENABLED", "true").lower() == "true"
CROSSREF_TOP_N_SOURCE_CHUNKS = _env_int("CROSSREF_TOP_N_SOURCE_CHUNKS", 3)  # only top-3 reranked chunks
CROSSREF_MAX_HOPS = _env_int("CROSSREF_MAX_HOPS", 1)  # single hop only, no recursion

# --- Chunking --------------------------------------------------------------
CHILD_MAX_CHARS = _env_int("CHILD_MAX_CHARS", 900)
PARENT_LOOKUP_PATH = DATA_DIR / "parent_lookup.pkl"
ARTICLE_NUM_LOOKUP_PATH = DATA_DIR / "article_num_lookup.pkl"  # A3: citation fast-path index

# --- Prompt compression -----------------------------------------------
COMPRESSION_ENABLED = _env("COMPRESSION_ENABLED", "true").lower() == "true"
COMPRESSION_TARGET_RATIO = _env_float("COMPRESSION_TARGET_RATIO", 0.5)

# --- Conversation state (conversational-retrieval skill) --------------------
SESSION_MAX_TURNS = _env_int("SESSION_MAX_TURNS", 20)
# When history exceeds this many turns, condense_question is fed a digest
# (via build_conversation_digest) instead of the raw transcript.
CONVERSATION_DIGEST_TRIGGER_TURNS = _env_int("CONVERSATION_DIGEST_TRIGGER_TURNS", 10)
CONVERSATION_DIGEST_MAX_NEW_TOKENS = _env_int("CONVERSATION_DIGEST_MAX_NEW_TOKENS", 220)
CONDENSE_QUESTION_MAX_NEW_TOKENS = _env_int("CONDENSE_QUESTION_MAX_NEW_TOKENS", 150)

# --- Scope/intent guardrail (C1) --------------------------------------------
GUARDRAIL_ENABLED = _env("GUARDRAIL_ENABLED", "true").lower() == "true"
LEGAL_DISCLAIMER = (
    "Lưu ý: đây là công cụ hỗ trợ tra cứu pháp luật tự động, không thay thế "
    "tư vấn pháp lý chính thức từ luật sư. Với quyết định quan trọng, bạn nên "
    "tham khảo ý kiến luật sư hoặc cơ quan có thẩm quyền."
)
ABSTENTION_MESSAGE = (
    "Xin lỗi, tôi không tìm thấy căn cứ pháp lý phù hợp trong corpus hiện có "
    "để trả lời chắc chắn câu hỏi này. Bạn có thể cung cấp thêm chi tiết, "
    "hoặc tham khảo ý kiến luật sư cho tình huống cụ thể của mình."
)
OUT_OF_SCOPE_MESSAGE = (
    "Tôi là trợ lý tra cứu pháp luật Việt Nam và chỉ có thể hỗ trợ các câu hỏi "
    "liên quan đến pháp luật. Câu hỏi này nằm ngoài phạm vi hỗ trợ của tôi."
)
