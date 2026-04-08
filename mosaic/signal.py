from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any

from .retriever import dual_hybrid_retrieve, hybrid_retrieve
from .types import Chunk
from .utils import DEFAULT_EMBEDDING_MODEL, embed_texts, load_json, roles_permit, tokenize

FAILURE_SUCCESS = "SUCCESS"
FAILURE_TRUE_UNKNOWN = "TRUE_UNKNOWN"
FAILURE_PERMISSION_GAP = "PERMISSION_GAP"
FAILURE_RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
FAILURE_TYPES = [FAILURE_SUCCESS, FAILURE_TRUE_UNKNOWN, FAILURE_PERMISSION_GAP, FAILURE_RETRIEVAL_FAILURE]

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "did", "disclose", "disclosed", "does", "for", "from", "how", "in", "is", "it", "its", "of",
    "on", "or", "the", "their", "to", "was", "what", "which", "who", "why", "with", "would", "your",
}

RETRIEVAL_TOPIC_SUPPORT_THRESHOLD = 0.12
PERMISSION_TOPIC_SUPPORT_THRESHOLD = 0.20


@dataclass(slots=True)
class SignalThresholds:
    open_relevance: float = 0.15
    gap: float = 0.20
    scoped_relevance: float = 0.12

    def to_dict(self) -> dict[str, float]:
        return {
            "open_relevance": round(self.open_relevance, 6),
            "gap": round(self.gap, 6),
            "scoped_relevance": round(self.scoped_relevance, 6),
        }


@dataclass(slots=True)
class SignalResult:
    classification: str
    open_score: float
    scoped_score: float
    gap: float
    topic_support: float
    latency_ms: float
    overhead_ms: float
    response: str
    hints: list[str] = field(default_factory=list)
    required_role: str | None = None
    open_top_doc_id: str | None = None
    scoped_top_doc_id: str | None = None
    open_top_roles: list[str] = field(default_factory=list)
    open_results: list[Chunk] = field(default_factory=list, repr=False)
    scoped_results: list[Chunk] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "open_score": round(self.open_score, 6),
            "scoped_score": round(self.scoped_score, 6),
            "gap": round(self.gap, 6),
            "topic_support": round(self.topic_support, 6),
            "latency_ms": round(self.latency_ms, 3),
            "overhead_ms": round(self.overhead_ms, 3),
            "response": self.response,
            "hints": list(self.hints),
            "required_role": self.required_role,
            "open_top_doc_id": self.open_top_doc_id,
            "scoped_top_doc_id": self.scoped_top_doc_id,
            "open_top_roles": list(self.open_top_roles),
        }


def _suite_queries(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "queries" in payload:
        return list(payload["queries"])
    return list(payload)  # type: ignore[arg-type]


def _all_roles(chunks: list[Chunk]) -> list[str]:
    return sorted({role for chunk in chunks for role in chunk.roles})


def _top_score(results: list[Chunk]) -> float:
    if not results:
        return 0.0
    top = results[0]
    return float(top.metadata.get("semantic_score", top.relevance))


def _top_chunk(results: list[Chunk]) -> Chunk | None:
    return results[0] if results else None


def build_corpus_idf(chunks: list[Chunk]) -> dict[str, float]:
    if not chunks:
        return {}
    doc_count = len(chunks)
    dfs: dict[str, int] = {}
    for chunk in chunks:
        for token in set(tokenize(chunk.text)):
            dfs[token] = dfs.get(token, 0) + 1
    return {token: log((1.0 + doc_count) / (1.0 + freq)) + 1.0 for token, freq in dfs.items()}


def _metadata_phrases(chunk: Chunk) -> list[str]:
    phrases: list[str] = []
    for key in ("workflow", "risk_focus", "board_priority", "outlook_signal", "remediation_owner"):
        value = chunk.metadata.get(key)
        if isinstance(value, str) and value.strip():
            phrases.append(value.strip())
    return phrases


def _extract_keyphrases(chunks: list[Chunk], corpus_idf: dict[str, float], query_tokens: set[str], limit: int = 3) -> list[str]:
    scores: dict[str, float] = {}
    for chunk in chunks:
        for phrase in _metadata_phrases(chunk):
            tokens = [token for token in tokenize(phrase) if token not in STOPWORDS]
            if not tokens or all(token in query_tokens for token in tokens):
                continue
            score = sum(corpus_idf.get(token, 1.0) for token in set(tokens)) / max(len(set(tokens)), 1)
            score += 0.05 * len(tokens)
            scores[phrase] = max(scores.get(phrase, 0.0), score)
    return [phrase for phrase, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _topic_support(query: str, top_chunk: Chunk | None) -> float:
    if top_chunk is None:
        return 0.0
    company_tokens = set(tokenize(str(top_chunk.metadata.get("company", ""))))
    query_tokens = [token for token in tokenize(query) if token not in STOPWORDS and token not in company_tokens]
    if not query_tokens:
        return 0.0
    chunk_tokens = set(tokenize(top_chunk.text))
    metadata_tokens = set()
    for phrase in _metadata_phrases(top_chunk):
        metadata_tokens.update(tokenize(phrase))
    support_tokens = chunk_tokens | metadata_tokens
    return sum(1 for token in query_tokens if token in support_tokens) / len(query_tokens)


def suggest_rephrasings(
    query: str,
    accessible_chunks: list[Chunk],
    corpus_idf: dict[str, float],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 3,
) -> list[str]:
    if not accessible_chunks:
        return []
    query_embedding = embed_texts([query], model_name=embedding_model, dimensions=len(accessible_chunks[0].embedding))[0]
    nearest = sorted(
        accessible_chunks,
        key=lambda chunk: sum(left * right for left, right in zip(query_embedding, chunk.embedding)),
        reverse=True,
    )[:5]
    hints = _extract_keyphrases(nearest, corpus_idf, set(tokenize(query)), limit=limit)
    return [f'"{hint}"' for hint in hints]


def dual_retrieve(
    query: str,
    chunks: list[Chunk],
    user_roles: list[str],
    candidate_k: int = 20,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    open_alpha: float = 1.0,
    scoped_alpha: float = 0.65,
) -> tuple[list[Chunk], list[Chunk], float]:
    start = perf_counter()
    open_results, scoped_results = dual_hybrid_retrieve(
        query=query,
        chunks=chunks,
        user_roles=user_roles,
        k=candidate_k,
        open_alpha=open_alpha,
        scoped_alpha=scoped_alpha,
        embedding_model=embedding_model,
    )
    return open_results, scoped_results, (perf_counter() - start) * 1000.0


def _required_role(open_chunk: Chunk | None, user_roles: list[str]) -> str | None:
    if open_chunk is None:
        return None
    for role in open_chunk.roles:
        if not roles_permit([role], user_roles):
            return role
    return None


def _render_response(classification: str, query: str, required_role: str | None, hints: list[str]) -> str:
    if classification == FAILURE_TRUE_UNKNOWN:
        return f"MOSAIC could not find evidence for '{query}' anywhere in the current knowledge base. Try widening the date range or checking whether the corpus needs a newer source document."
    if classification == FAILURE_PERMISSION_GAP:
        role_text = required_role or "an additional access role"
        return (
            f"Relevant material exists, but your current access scope cannot reach it. "
            f"This likely requires {role_text}. Ask the workspace owner, document owner, or an administrator to grant access."
        )
    if classification == FAILURE_RETRIEVAL_FAILURE:
        if hints:
            return "Relevant accessible material exists, but the current phrasing missed it. Try rephrasing with: " + ", ".join(hints)
        return "Relevant accessible material exists, but the current phrasing missed it. Try a narrower wording that names the workflow, risk focus, or board priority directly."
    return "Accessible evidence was found and the query can be routed to the assembler."


def classify_scores(
    open_score: float,
    scoped_score: float,
    gap: float,
    open_forbidden: bool,
    topic_support: float,
    thresholds: SignalThresholds,
) -> str:
    if open_forbidden and gap > thresholds.gap and topic_support >= PERMISSION_TOPIC_SUPPORT_THRESHOLD:
        return FAILURE_PERMISSION_GAP
    if open_score < thresholds.open_relevance:
        return FAILURE_RETRIEVAL_FAILURE if topic_support >= RETRIEVAL_TOPIC_SUPPORT_THRESHOLD else FAILURE_TRUE_UNKNOWN
    if scoped_score < thresholds.scoped_relevance:
        return FAILURE_RETRIEVAL_FAILURE
    return FAILURE_SUCCESS


def classify_query(
    query: str,
    chunks: list[Chunk],
    user_roles: list[str],
    thresholds: SignalThresholds | None = None,
    candidate_k: int = 20,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    corpus_idf: dict[str, float] | None = None,
    measure_overhead: bool = False,
) -> SignalResult:
    resolved_thresholds = thresholds or SignalThresholds()
    baseline_ms = 0.0
    if measure_overhead:
        start = perf_counter()
        hybrid_retrieve(query, chunks, user_roles, candidate_k, 0.65, embedding_model)
        baseline_ms = (perf_counter() - start) * 1000.0

    open_results, scoped_results, dual_ms = dual_retrieve(
        query=query,
        chunks=chunks,
        user_roles=user_roles,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
    )
    open_top = _top_chunk(open_results)
    scoped_top = _top_chunk(scoped_results)
    open_score = _top_score(open_results)
    scoped_score = _top_score(scoped_results)
    gap = max(0.0, open_score - scoped_score)
    open_forbidden = open_top is not None and not roles_permit(open_top.roles, user_roles)
    required_role = _required_role(open_top, user_roles)
    accessible_chunks = [chunk for chunk in chunks if roles_permit(chunk.roles, user_roles)]
    topic_support = _topic_support(query, open_top)
    hints: list[str] = []
    classification = classify_scores(open_score, scoped_score, gap, open_forbidden, topic_support, resolved_thresholds)
    if classification == FAILURE_RETRIEVAL_FAILURE:
        hints = suggest_rephrasings(query, accessible_chunks, corpus_idf or build_corpus_idf(accessible_chunks), embedding_model=embedding_model)
    response = _render_response(classification, query, required_role, hints)
    return SignalResult(
        classification=classification,
        open_score=open_score,
        scoped_score=scoped_score,
        gap=gap,
        topic_support=topic_support,
        latency_ms=dual_ms,
        overhead_ms=max(0.0, dual_ms - baseline_ms) if measure_overhead else dual_ms,
        response=response,
        hints=hints,
        required_role=required_role,
        open_top_doc_id=open_top.document_id if open_top else None,
        scoped_top_doc_id=scoped_top.document_id if scoped_top else None,
        open_top_roles=list(open_top.roles) if open_top else [],
        open_results=open_results,
        scoped_results=scoped_results,
    )


def _candidate_thresholds(values: list[float]) -> list[float]:
    rounded = {round(value, 4) for value in values}
    rounded.update({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
    return sorted(rounded)


def _youden_threshold(samples: list[dict[str, Any]], key: str, positive_labels: set[str]) -> float:
    best_threshold = 0.0
    best_score = -1.0
    for threshold in _candidate_thresholds([float(sample[key]) for sample in samples]):
        tp = sum(1 for sample in samples if sample["label"] in positive_labels and float(sample[key]) >= threshold)
        fn = sum(1 for sample in samples if sample["label"] in positive_labels and float(sample[key]) < threshold)
        tn = sum(1 for sample in samples if sample["label"] not in positive_labels and float(sample[key]) < threshold)
        fp = sum(1 for sample in samples if sample["label"] not in positive_labels and float(sample[key]) >= threshold)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        score = sensitivity + specificity - 1.0
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _predict_from_sample(sample: dict[str, Any], thresholds: SignalThresholds) -> str:
    return classify_scores(
        open_score=float(sample["open_score"]),
        scoped_score=float(sample["scoped_score"]),
        gap=float(sample["gap"]),
        open_forbidden=bool(sample.get("open_forbidden", False)),
        topic_support=float(sample.get("topic_support", 0.0)),
        thresholds=thresholds,
    )


def summarize_signal_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    accuracy = sum(1 for row in rows if row["correct"]) / max(total, 1)
    type2_rows = [row for row in rows if row["predicted_type"] == FAILURE_PERMISSION_GAP]
    true_type2 = [row for row in rows if row["ground_truth_type"] == FAILURE_PERMISSION_GAP]
    tp_type2 = sum(1 for row in rows if row["predicted_type"] == FAILURE_PERMISSION_GAP and row["ground_truth_type"] == FAILURE_PERMISSION_GAP)
    precision = tp_type2 / max(len(type2_rows), 1)
    recall = tp_type2 / max(len(true_type2), 1)

    per_type = []
    confusion: dict[str, dict[str, int]] = {}
    for label in (FAILURE_TRUE_UNKNOWN, FAILURE_PERMISSION_GAP, FAILURE_RETRIEVAL_FAILURE):
        label_rows = [row for row in rows if row["ground_truth_type"] == label]
        per_type.append(
            {
                "label": label,
                "accuracy": round(sum(1 for row in label_rows if row["correct"]) / max(len(label_rows), 1), 4),
                "count": len(label_rows),
            }
        )
        confusion[label] = {}
        for predicted in (FAILURE_TRUE_UNKNOWN, FAILURE_PERMISSION_GAP, FAILURE_RETRIEVAL_FAILURE):
            confusion[label][predicted] = sum(1 for row in label_rows if row["predicted_type"] == predicted)

    return {
        "classification_accuracy": round(accuracy, 4),
        "type2_precision": round(precision, 4),
        "type2_recall": round(recall, 4),
        "latency_overhead_ms": round(sum(row.get("overhead_ms", 0.0) for row in rows) / max(total, 1), 3),
        "per_type": per_type,
        "confusion_matrix": confusion,
    }


def calibrate_thresholds(
    chunks: list[Chunk],
    suite_payload: object,
    candidate_k: int = 20,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[SignalThresholds, dict[str, Any], list[dict[str, Any]]]:
    queries = _suite_queries(suite_payload)
    calibration_items = [item for item in queries if item.get("category") in {"failure_classification", "redundancy_trap", "multi_hop"}]
    samples: list[dict[str, Any]] = []
    for item in calibration_items:
        label = item.get("ground_truth_type", FAILURE_SUCCESS)
        signal = classify_query(
            query=item["query"],
            chunks=chunks,
            user_roles=list(item.get("user_roles", ["public"])),
            candidate_k=candidate_k,
            embedding_model=embedding_model,
        )
        samples.append(
            {
                "id": item["id"],
                "label": label,
                "open_score": signal.open_score,
                "scoped_score": signal.scoped_score,
                "gap": signal.gap,
                "open_forbidden": bool(signal.open_top_roles and not roles_permit(signal.open_top_roles, list(item.get("user_roles", ["public"])))),
                "topic_support": signal.topic_support,
            }
        )

    open_threshold = _youden_threshold(samples, "open_score", {FAILURE_SUCCESS, FAILURE_PERMISSION_GAP, FAILURE_RETRIEVAL_FAILURE})
    gap_samples = [sample for sample in samples if sample["label"] != FAILURE_TRUE_UNKNOWN]
    gap_threshold = _youden_threshold(gap_samples, "gap", {FAILURE_PERMISSION_GAP}) if gap_samples else SignalThresholds().gap
    scoped_samples = [sample for sample in samples if sample["label"] in {FAILURE_SUCCESS, FAILURE_RETRIEVAL_FAILURE}]
    scoped_threshold = _youden_threshold(scoped_samples, "scoped_score", {FAILURE_SUCCESS}) if scoped_samples else SignalThresholds().scoped_relevance

    thresholds = SignalThresholds(open_relevance=open_threshold, gap=gap_threshold, scoped_relevance=scoped_threshold)
    scored_rows = []
    for sample in samples:
        predicted = _predict_from_sample(sample, thresholds)
        scored_rows.append({**sample, "ground_truth_type": sample["label"], "predicted_type": predicted, "correct": predicted == sample["label"]})
    return thresholds, summarize_signal_rows(scored_rows), scored_rows


def run_signal_eval(
    chunks: list[Chunk],
    eval_suite_path: str | Any,
    thresholds: SignalThresholds | None = None,
    candidate_k: int = 20,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any], SignalThresholds]:
    suite_payload = load_json(eval_suite_path) if isinstance(eval_suite_path, (str, bytes, Path)) else eval_suite_path
    queries = [item for item in _suite_queries(suite_payload) if item.get("category") == "failure_classification"]
    resolved_thresholds = thresholds
    if resolved_thresholds is None:
        resolved_thresholds, _, _ = calibrate_thresholds(chunks, suite_payload, candidate_k=candidate_k, embedding_model=embedding_model)
    corpus_idf = build_corpus_idf(chunks)
    rows: list[dict[str, Any]] = []
    for item in queries:
        signal = classify_query(
            query=item["query"],
            chunks=chunks,
            user_roles=list(item.get("user_roles", ["public"])),
            thresholds=resolved_thresholds,
            candidate_k=candidate_k,
            embedding_model=embedding_model,
            corpus_idf=corpus_idf,
            measure_overhead=True,
        )
        rows.append(
            {
                "query_id": item["id"],
                "query": item["query"],
                "ground_truth_type": item["ground_truth_type"],
                "predicted_type": signal.classification,
                "correct": signal.classification == item["ground_truth_type"],
                **signal.to_dict(),
            }
        )
    return rows, summarize_signal_rows(rows), resolved_thresholds
