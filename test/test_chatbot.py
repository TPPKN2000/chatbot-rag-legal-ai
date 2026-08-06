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


def evaluate_groundedness(answer_text: str, allowed_keys) -> dict:
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
    precision = grounded / total if total else 1.0
    return {"n_citations": total, "n_grounded": grounded, "precision": precision}


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
    return {"n": len(queries), "n_correct": n_correct, "rate": n_correct / len(queries) if queries else 1.0, "details": details}


def _load_eval_set() -> list:
    from backend import config
    if not config.CHATBOT_EVAL_SET_PATH.exists():
        return []
    return json.loads(config.CHATBOT_EVAL_SET_PATH.read_text(encoding="utf-8"))


# Fallback tối thiểu CHỈ dùng khi data/chatbot_eval_set.json chưa có case
# type=="fastpath" nào (vd. checkout hoàn toàn mới, chưa chạy A3) — để trục
# này không skip hoàn toàn.
_FASTPATH_NEGATIVE_CASES_FALLBACK = [
    "Điều 12 áp dụng thế nào cho trường hợp ly hôn đơn phương?",
    "Nếu vi phạm Điều 5 thì bị xử lý ra sao trong trường hợp tái phạm?",
]


def evaluate_fastpath() -> dict:
    """A3 (coding_plan.md): case dương/âm nạp từ config.CHATBOT_EVAL_SET_PATH
    (type == "fastpath") thay vì để trống — trước đây _FASTPATH_POSITIVE_CASES
    luôn rỗng vì "cần biết số Điều thật trong corpus". Với corpus thật, thay
    data/chatbot_eval_set.json bằng bộ 30-50 câu soạn theo corpus đó; các case
    fastpath ở đây tự động được dùng lại, không cần sửa hàm này."""
    from backend.retrieval.citation_fastpath import try_lookup

    eval_set = _load_eval_set()
    fastpath_items = [item for item in eval_set if item.get("type") == "fastpath"]
    positive_items = [it for it in fastpath_items if it.get("expected_fastpath")]
    negative_items = [it for it in fastpath_items if not it.get("expected_fastpath")]

    false_positives = []
    for item in negative_items:
        if try_lookup(item["question"]) is not None:
            false_positives.append(item["question"])

    false_negatives = []
    wrong_content = []
    for item in positive_items:
        hit = try_lookup(item["question"])
        if hit is None:
            false_negatives.append(item["question"])
            continue
        expected_substr = item.get("expected_substring")
        if expected_substr and expected_substr not in hit.answer:
            wrong_content.append(item["question"])

    n_negative = len(negative_items) or len(_FASTPATH_NEGATIVE_CASES_FALLBACK)
    if not negative_items:
        # data/chatbot_eval_set.json chưa có case fastpath -> fallback về
        # bộ tối thiểu hard-coded để trục này vẫn chạy được (degrade, không
        # bỏ qua hoàn toàn).
        for q in _FASTPATH_NEGATIVE_CASES_FALLBACK:
            if try_lookup(q) is not None:
                false_positives.append(q)

    return {
        "negative_cases_tested": n_negative,
        "false_positives": false_positives,
        "false_positive_rate": len(false_positives) / n_negative if n_negative else 0.0,
        "positive_cases_tested": len(positive_items),
        "false_negatives": false_negatives,
        "wrong_content": wrong_content,
        "positive_accuracy": (
            (len(positive_items) - len(false_negatives) - len(wrong_content)) / len(positive_items)
            if positive_items else None
        ),
    }


def evaluate_retrieval_recall(hand_curated_qa: list, k: int = 5) -> dict:
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
                         help="Skip axes that call the generation LLM (abstention uses the refusal gate, "
                              "which does NOT call the LLM, so it still runs).")
    args = parser.parse_args()

    from backend import config

    log.info("=" * 60)
    log.info("2) Abstention correctness (NEGATIVE_SAFETY_QUERIES)")
    log.info("=" * 60)
    abstention = evaluate_abstention()
    log.info("Abstention rate: %d/%d (%.1f%%)", abstention["n_correct"], abstention["n"], 100 * abstention["rate"])
    for d in abstention["details"]:
        if not d["abstained"]:
            log.warning("  FAILED to abstain on: %r", d["query"])

    log.info("=" * 60)
    log.info("3) Citation fast-path — positive/negative case check")
    log.info("=" * 60)
    fastpath = evaluate_fastpath()
    log.info("Fast-path false-positive rate: %.1f%% (%d/%d)", 100 * fastpath["false_positive_rate"],
              len(fastpath["false_positives"]), fastpath["negative_cases_tested"])
    if fastpath["positive_cases_tested"]:
        log.info("Fast-path positive accuracy: %.1f%% (%d case, %d false-negative, %d sai nội dung)",
                  100 * (fastpath["positive_accuracy"] or 0), fastpath["positive_cases_tested"],
                  len(fastpath["false_negatives"]), len(fastpath["wrong_content"]))
    if fastpath["false_positives"]:
        log.warning("  Fast-path kích hoạt SAI trên: %s", fastpath["false_positives"])
    if fastpath["false_negatives"]:
        log.warning("  Fast-path KHÔNG kích hoạt (đáng lẽ phải) trên: %s", fastpath["false_negatives"])
    if fastpath["wrong_content"]:
        log.warning("  Fast-path trả sai nội dung trên: %s", fastpath["wrong_content"])

    log.info("=" * 60)
    log.info("4) Retrieval recall@k (hand-curated eval set)")
    log.info("=" * 60)
    hand_curated = _load_eval_set()
    if not hand_curated:
        log.info("No hand-curated eval set at %s — skipping recall@k. See A3 (coding_plan.md).",
                  config.CHATBOT_EVAL_SET_PATH)
    recall = evaluate_retrieval_recall(hand_curated, k=5)
    if recall["recall_at_k"] is not None:
        log.info("Recall@%d: %.3f (n=%d)", recall["k"], recall["recall_at_k"], recall["n"])

    # sample_questions dùng cho bước 1 (groundedness) và bước 5 (faithfulness/
    # helpfulness) — chỉ lấy type=="retrieval": câu hỏi fastpath không qua LLM
    # nên groundedness/faithfulness không áp dụng, đưa vào sẽ cho số liệu vô nghĩa.
    retrieval_questions = [item["question"] for item in hand_curated if item.get("type") == "retrieval"][:5]

    if not args.skip_generation:
        log.info("=" * 60)
        log.info("1) Groundedness (sample live chat turns)")
        log.info("=" * 60)
        from backend.chat_pipeline import handle_chat_turn
        from backend.generation.prompt_builder import allowed_citation_keys
        from backend.pipeline import collect_law_evidence

        answers_by_question = {}
        if not retrieval_questions:
            log.info("No retrieval-type sample questions available — skipping groundedness.")
        for q in retrieval_questions:
            answer = handle_chat_turn(f"eval-groundedness-{hash(q)}", q)
            answers_by_question[q] = answer
            law_chunks = collect_law_evidence(q)
            allowed = allowed_citation_keys(law_chunks)
            g = evaluate_groundedness(answer.answer, allowed)
            log.info("  %r -> citations=%d grounded=%d precision=%.2f", q, g["n_citations"], g["n_grounded"], g["precision"])

        # --- 5) Faithfulness / Hallucination + Helpfulness (D1-D4) ---
        log.info("=" * 60)
        log.info("5) Faithfulness / Hallucination (claim-level, judge-based)")
        log.info("=" * 60)
        from test.eval_faithfulness import evaluate_faithfulness, judge_answer
        from test.judge_client import is_judge_available

        if not is_judge_available():
            log.info("No external judge configured (ANTHROPIC_API_KEY not set) — skipping "
                     "faithfulness/helpfulness axes. See coding_plan.md Nhóm D3 for how to configure one.")
        else:
            faith_scores, help_scores = [], []
            for q, answer in answers_by_question.items():
                law_chunks = collect_law_evidence(q)
                try:
                    f = evaluate_faithfulness(answer.answer, law_chunks)
                    if f["faithfulness_score"] is not None:
                        faith_scores.append(f["faithfulness_score"])
                    j = judge_answer(q, answer.answer)
                    if j.get("helpfulness") is not None:
                        help_scores.append(j["helpfulness"])
                    log.info("  %r -> faithfulness=%s hallucination=%s helpfulness=%s",
                              q, f["faithfulness_score"], f["hallucination_rate"], j.get("helpfulness"))
                except Exception as e:
                    log.warning("  faithfulness/judge eval crashed on %r: %s (degrade, không chặn eval run)", q, e)

            if faith_scores or help_scores:
                avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else float("nan")
                avg_hall = 1 - avg_faith if faith_scores else float("nan")
                avg_help = sum(help_scores) / len(help_scores) if help_scores else float("nan")
                log.info("Faithfulness: %.2f | Hallucination rate: %.2f | Helpfulness (avg 1-5): %.1f",
                          avg_faith, avg_hall, avg_help)
    else:
        log.info("--skip-generation set: skipping groundedness + faithfulness/helpfulness axes "
                 "(cả hai đều gọi LLM sinh câu trả lời trước khi đo).")

    log.info("=" * 60)
    log.info("Done. Harness này để bắt REGRESSION, không phải để ra 1 con số leaderboard duy nhất — "
             "xem eval-harness-chatbot skill cho lý do không tính composite score.")


if __name__ == "__main__":
    main()
