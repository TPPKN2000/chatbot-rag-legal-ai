"""
Named-entity recognition helper.

CHATBOT_MIGRATION_PLAN.md §2.1: kept unchanged — retrieval-purpose-agnostic.
Used to strip plaintiff/defendant/person/org names out of the query used for
LAW retrieval (a name is pure noise when searching for a statute).
"""
from __future__ import annotations

from functools import lru_cache

from backend import config

_MASKED_ENTITY_GROUPS = ("PERSON", "ORGANIZATION", "ORG")


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=config.NER_MODEL_NAME,
        aggregation_strategy="simple",
    )


def extract_entities(text: str) -> list[dict]:
    """Fail-safe: a model-loading or inference error must never block
    retrieval — an empty list means "no entities found", not an error."""
    if not text.strip():
        return []
    try:
        return list(_get_ner_pipeline()(text))
    except Exception:
        return []


def mask_person_org_entities(text: str, entities: list[dict] | None = None) -> str:
    """Replace PERSON/ORGANIZATION spans with a neutral placeholder."""
    entities = entities if entities is not None else extract_entities(text)
    spans = sorted(
        [e for e in entities if e.get("entity_group") in _MASKED_ENTITY_GROUPS and "start" in e and "end" in e],
        key=lambda e: e["start"],
        reverse=True,
    )
    for e in spans:
        text = text[: e["start"]] + "[BÊN LIÊN QUAN]" + text[e["end"] :]
    return text
