"""
Build the BM25 + Pinecone indexes for the law corpus, plus the auxiliary
lookups used by retrieval:
  - parent_lookup (whole-Điều text, used to give the reranker full-article
    context — unchanged from the ALQAC-era pipeline).
  - fastpath index (CHATBOT_MIGRATION_PLAN.md §A3 citation-fast-path skill):
    {article_lookup, khoan_lookup, chunk_by_id} — resolves a parsed
    "Điều X [khoản Y]" reference straight to its chunk(s), and is ALSO reused
    by pipeline.py's C2 cross-reference hop (same index, two call sites —
    see legacy-prototype-salvage skill's "don't create a 3rd parallel code
    path" principle).

Routes through the real ingestion pipeline: `parser.load_law_corpus` ->
`chunker.chunk_articles` (real Chương>Mục>Điều>Khoản>Điểm structural
splitting, with the soft-split for oversized clauses) -> the three lookup
builders above.
"""
import argparse
import logging
import pickle

from backend import config
from backend.indexing import vector_store
from backend.indexing.bm25_index import BM25Index
from backend.ingestion.chunker import build_article_num_lookup, build_khoan_lookup, build_parent_lookup, chunk_articles
from backend.ingestion.parser import load_law_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--rebuild-pinecone", action="store_true")
    args = parser.parse_args()

    logger.info(f"loading law corpus from {args.corpus}")
    law_docs = load_law_corpus(args.corpus)
    logger.info(f"loaded {len(law_docs)} law documents")

    law_chunks = []
    for doc in law_docs:
        law_chunks.extend(chunk_articles(doc.articles))

    n_parent = sum(1 for c in law_chunks if c.level == "parent")
    n_child = sum(1 for c in law_chunks if c.level == "child")
    logger.info(f"chunked into {n_parent} parent + {n_child} child chunks")

    if n_child == 0:
        logger.error("No chunks created.")
        return

    # 0a. Parent (whole-Điều) lookup, for rerank full-article context.
    parent_lookup = build_parent_lookup(law_chunks)
    config.PARENT_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PARENT_LOOKUP_PATH, "wb") as f:
        pickle.dump(parent_lookup, f)
    logger.info(f"parent lookup ({len(parent_lookup)} articles) saved to {config.PARENT_LOOKUP_PATH}")

    # 0b. Citation fast-path index (A3): article_lookup + khoan_lookup +
    # chunk_by_id (ALL chunks, parent and child — cross-ref resolution in
    # pipeline.py wants whole-article parent text; fast-path Khoản lookups
    # want child text).
    article_lookup = build_article_num_lookup(law_chunks)
    khoan_lookup = build_khoan_lookup(law_chunks)
    chunk_by_id = {c.chunk_id: c for c in law_chunks}
    n_with_article_num = sum(1 for c in law_chunks if c.level == "parent" and c.article_num is not None)
    logger.info(
        "citation fast-path index: %d/%d articles have a parseable article_num "
        "(the rest fall back to full retrieval — see chunker.py's _parse_real_article_num)",
        n_with_article_num, n_parent,
    )
    config.ARTICLE_NUM_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ARTICLE_NUM_LOOKUP_PATH, "wb") as f:
        pickle.dump(
            {"article_lookup": article_lookup, "khoan_lookup": khoan_lookup, "chunk_by_id": chunk_by_id}, f
        )
    logger.info(f"fastpath index saved to {config.ARTICLE_NUM_LOOKUP_PATH}")

    # 1. Build BM25 Index (now also builds the D1 accent-folded auxiliary index).
    logger.info("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(law_chunks)
    bm25.save()
    logger.info(f"BM25 index saved to {config.BM25_INDEX_PATH}")

    # 2. Build Pinecone Index
    if args.rebuild_pinecone:
        logger.info("Rebuilding Pinecone index...")
        count = vector_store.upsert_chunks(law_chunks)
        logger.info(f"Successfully upserted {count} chunks to Pinecone.")


if __name__ == "__main__":
    main()
