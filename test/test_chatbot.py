"""
Eval harness for the LegalRAG chatbot (eval-harness-chatbot skill).

CHATBOT_MIGRATION_PLAN.md §2.2 (removed): `test/test_all_backend.py`
(OutcomeAccuracy / Micro Law F1 against ALQAC2026_public_test.json's
`verdict_label`) doesn't apply anymore — there's no gold label for a free-
form chat answer. This harness measures 4 axes instead, 3 of which need NO
LLM-judge (pure set/pattern checks, cheap and reliable):

  1. Groundedness  — every "[Điều X, law_id]" in an answer must be in the
     retrieved set the model was shown (reuses generate.py's own
     hallucination-guard mechanism — this is what it was built to check).
  2. Abstention correctness — NEGATIVE_SAFETY_QUERIES (shared with
     backend/guardrail.py) must all be refused/deflected.
  3. Citation-fastpath correctness — positive cases return the exact
     verbatim corpus text with zero LLM calls; cases with situational
     conditions must NOT trigger the fast-path.
  4. Retrieval recall@k — needs a small hand-curated {question, gold_aids}
     set (config.CHATBOT_EVAL_SET_PATH); skipped with a clear message if
     that file doesn't exist yet, rather than failing the whole run (per
     the skill: "không cần lớn... 30-50 câu đủ để phát hiện regression",
     and it's fine for this file to not exist on a fresh checkout).

"Answer helpfulness/coherence" (the skill's 5th axis) is intentionally NOT
implemented here — it's the one axis the skill says may need an LLM-judge,
which is a separate decision the migration plan defers ("Có thể cân nhắc
lại việc dùng thư viện eval ngoài nếu team thấy lợi ích rõ ràng, nhưng cần
đánh giá riêng").

Usage:
    python -m test.test_chatbot                  # all axes
    python -m test.test_chatbot --skip-generation # groundedness/fastpath/
                                                    # recall only, no LLM calls
                                                    # (fast smoke test)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Groundedness
# ---------------------------------------------------------------------------
def evaluate_groundedness(answer_text: str, allowed_keys) -> dict:
    """Parse every "[Điều X, law_id]" in `answer_text`, check each against
    `allowed_keys` (a dict or set keyed by (law_id, display_num)). Returns
    {"n_citations": int, "n_grounded": int, "precision": float}. No recall
    is reported — there's no fixed gold citation list for a free-form
    answer, only "did you cite something you weren't shown"."""
    from backend.generation.generate import _CITATION_RE

    total, grounded = 0, 0
    for m in _CITATION_RE.finditer(answer_text):
        total += 1
        try:
            key = (m.group(2).strip(), int(m.group(1)))
        except ValueError:
            continue
        if key in allowed_keys:
            grounded += 1
    precision = grounded / total if total else 1.0  # vacuously grounded if no citations made
    return {"n_citations": total, "n_grounded": grounded, "precision": precision}


# ---------------------------------------------------------------------------
# 2. Abstention correctness
# ---------------------------------------------------------------------------
def evaluate_abstention(queries=None) -> dict:
    from backend.chat_pipeline import handle_chat_turn
    from backend.guardrail import NEGATIVE_SAFETY_QUERIES

    queries = queries or NEGATIVE_SAFETY_QUERIES
    n_correct = 0
    details = []
    for i, q in enumerate(queries):
        session_id = f"eval-abstention-{i}"
        try:
            answer = handle_chat_turn(session_id, q)
            correct = bool(answer.is_abstention)
        except Exception as e:
            log.warning("abstention eval crashed on %r: %s", q, e)
            correct = False
        n_correct += int(correct)
        details.append({"query": q, "abstained": correct})
    return {"n": len(queries), "n_correct": n_correct, "rate": n_correct / len(queries) if queries else 1.0,
            "details": details}


# ---------------------------------------------------------------------------
# 3. Citation fast-path correctness
# ---------------------------------------------------------------------------
_FASTPATH_POSITIVE_CASES = [
    # (question, expected substring — filled in per-corpus; left generic here
    # since the actual article text depends on which corpus is indexed).
]

_FASTPATH_NEGATIVE_CASES = [
    "Điều 12 áp dụng thế nào cho trường hợp ly hôn đơn phương?",
    "Nếu vi phạm Điều 5 thì bị xử lý ra sao trong trường hợp tái phạm?",
]


def evaluate_fastpath() -> dict:
    from backend.retrieval.citation_fastpath import try_lookup

    false_positives = []
    for q in _FASTPATH_NEGATIVE_CASES:
        hit = try_lookup(q)
        if hit is not None:
            false_positives.append(q)

    n_zero_llm_calls_verified = 0  # positive-case verification needs a real corpus; see note below.
    return {
        "negative_cases_tested": len(_FASTPATH_NEGATIVE_CASES),
        "false_positives": false_positives,
        "false_positive_rate": len(false_positives) / len(_FASTPATH_NEGATIVE_CASES) if _FASTPATH_NEGATIVE_CASES else 0.0,
        "note": (
            "Positive-case exact-match verification needs real corpus-specific "
            "(question, expected article text) pairs — add them to "
            "_FASTPATH_POSITIVE_CASES once data/corpus_law_pub.json is available "
            "to hand-pick real Điều numbers from."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Retrieval recall@k
# ---------------------------------------------------------------------------
def evaluate_retrieval_recall(hand_curated_qa: list[dict], k: int = 5) -> dict:
    from backend.pipeline import collect_law_evidence

    if not hand_curated_qa:
        return {"n": 0, "recall_at_k": None, "note": "no hand-curated eval set provided"}

    recalls = []
    for item in hand_curated_qa:
        question = item["question"]
        gold_aids = set(item.get("gold_aids", []))
        if not gold_aids:
            continue
        retrieved = collect_law_evidence(question)[:k]
        pred_aids = {c.aid for c in retrieved}
        recalls.append(len(pred_aids & gold_aids) / len(gold_aids))

    avg_recall = sum(recalls) / len(recalls) if recalls else None
    return {"n": len(recalls), "recall_at_k": avg_recall, "k": k}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-generation", action="store_true",
                         help="Skip axes that call the generation LLM (abstention uses generate indirectly "
                              "via the refusal gate, which does NOT call the LLM, so it still runs).")
    args = parser.parse_args()

    from backend import config

    log.info("=" * 60)
    log.info("2) Abstention correctness (NEGATIVE_SAFETY_QUERIES)")
    log.info("=" * 60)
    abstention = evaluate_abstention()
    log.info("Abstention rate: %d/%d (%.1f%%)", abstention["n_correct"], abstention["n"],
              100 * abstention["rate"])
    for d in abstention["details"]:
        if not d["abstained"]:
            log.warning("  FAILED to abstain on: %r", d["query"])

    log.info("=" * 60)
    log.info("3) Citation fast-path — false-positive check")
    log.info("=" * 60)
    fastpath = evaluate_fastpath()
    log.info("Fast-path false-positive rate on situational questions: %.1f%% (%d/%d)",
              100 * fastpath["false_positive_rate"], len(fastpath["false_positives"]),
              fastpath["negative_cases_tested"])
    if fastpath["false_positives"]:
        log.warning("  Fast-path incorrectly triggered on: %s", fastpath["false_positives"])

    log.info("=" * 60)
    log.info("4) Retrieval recall@k (hand-curated eval set)")
    log.info("=" * 60)
    hand_curated = []
    if config.CHATBOT_EVAL_SET_PATH.exists():
        hand_curated = json.loads(config.CHATBOT_EVAL_SET_PATH.read_text(encoding="utf-8"))
    else:
        log.info("No hand-curated eval set at %s — skipping recall@k. See eval-harness-chatbot skill "
                  "for the {question, gold_aids} format to create one.", config.CHATBOT_EVAL_SET_PATH)
    recall = evaluate_retrieval_recall(hand_curated, k=5)
    if recall["recall_at_k"] is not None:
        log.info("Recall@%d: %.3f (n=%d)", recall["k"], recall["recall_at_k"], recall["n"])

    if not args.skip_generation:
        log.info("=" * 60)
        log.info("1) Groundedness (sample live chat turns)")
        log.info("=" * 60)
        from backend.chat_pipeline import handle_chat_turn
        from backend.generation.prompt_builder import allowed_citation_keys
        from backend.pipeline import collect_law_evidence

        sample_questions = [item["question"] for item in hand_curated[:5]] or []
        if not sample_questions:
            log.info("No sample questions available (hand-curated eval set empty) — skipping groundedness.")
        for q in sample_questions:
            answer = handle_chat_turn(f"eval-groundedness-{hash(q)}", q)
            law_chunks = collect_law_evidence(q)
            allowed = allowed_citation_keys(law_chunks)
            g = evaluate_groundedness(answer.answer, allowed)
            log.info("  %r -> citations=%d grounded=%d precision=%.2f", q, g["n_citations"], g["n_grounded"],
                      g["precision"])
    else:
        log.info("--skip-generation set: skipping groundedness axis (calls the generation LLM).")

    log.info("=" * 60)
    log.info("Done. This harness is meant to catch REGRESSIONS, not to produce a single leaderboard "
              "number — see eval-harness-chatbot skill for why no single composite score is computed.")


if __name__ == "__main__":
    main()
