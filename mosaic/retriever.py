from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .ingestor import load_index
from .types import Chunk, RetrievalIndex
from .utils import DEFAULT_EMBEDDING_MODEL, cosine_similarity, embed_texts, normalize_scores, roles_permit, tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None


def lexical_overlap_score(query_tokens: list[str], chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(tokenize(chunk_text))
    hits = sum(1 for token in query_tokens if token in chunk_tokens)
    return hits / len(query_tokens)


def _bm25_scores(query_tokens: list[str], corpus_tokens: list[list[str]]) -> list[float]:
    if not corpus_tokens:
        return []
    if BM25Okapi is None:
        return [lexical_overlap_score(query_tokens, " ".join(tokens)) for tokens in corpus_tokens]
    model = BM25Okapi(corpus_tokens)
    return [float(value) for value in model.get_scores(query_tokens)]


def _score_components(query: str, chunks: list[Chunk], embedding_model: str) -> tuple[list[float], list[float]]:
    if not chunks:
        return [], []
    query_tokens = tokenize(query)
    query_embedding = embed_texts([query], model_name=embedding_model, dimensions=len(chunks[0].embedding))[0]
    bm25_scores = _bm25_scores(query_tokens, [tokenize(chunk.text) for chunk in chunks])
    semantic_scores = [cosine_similarity(query_embedding, chunk.embedding) for chunk in chunks]
    return bm25_scores, semantic_scores


def _scored_chunks(chunks: list[Chunk], bm25_scores: list[float], semantic_scores: list[float], alpha: float) -> list[Chunk]:
    if not chunks:
        return []
    bm25_norm = normalize_scores(bm25_scores)
    semantic_norm = normalize_scores(semantic_scores)
    results: list[Chunk] = []
    for chunk, bm25_score, semantic_score, bm25_value, semantic_value in zip(chunks, bm25_norm, semantic_norm, bm25_scores, semantic_scores):
        hybrid_score = (alpha * semantic_score) + ((1.0 - alpha) * bm25_score)
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "bm25_score": round(float(bm25_value), 6),
                "semantic_score": round(float(semantic_value), 6),
                "hybrid_score": round(float(hybrid_score), 6),
            }
        )
        results.append(replace(chunk, relevance=float(hybrid_score), metadata=metadata))
    results.sort(key=lambda item: item.relevance, reverse=True)
    return results


def hybrid_retrieve(
    query: str,
    chunks: list[Chunk],
    user_roles: list[str],
    k: int = 20,
    alpha: float = 0.65,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[Chunk]:
    permitted_chunks = [chunk for chunk in chunks if roles_permit(chunk.roles, user_roles)]
    if not permitted_chunks:
        return []
    bm25_scores, semantic_scores = _score_components(query, permitted_chunks, embedding_model)
    return _scored_chunks(permitted_chunks, bm25_scores, semantic_scores, alpha)[:k]


def dual_hybrid_retrieve(
    query: str,
    chunks: list[Chunk],
    user_roles: list[str],
    k: int = 20,
    open_alpha: float = 1.0,
    scoped_alpha: float = 0.65,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[Chunk], list[Chunk]]:
    if not chunks:
        return [], []
    bm25_scores, semantic_scores = _score_components(query, chunks, embedding_model)
    open_results = _scored_chunks(chunks, bm25_scores, semantic_scores, open_alpha)[:k]
    scoped_ranked = _scored_chunks(chunks, bm25_scores, semantic_scores, scoped_alpha)
    scoped_results = [chunk for chunk in scoped_ranked if roles_permit(chunk.roles, user_roles)][:k]
    return open_results, scoped_results


def retrieve_from_index(
    index_path: str | Path,
    query: str,
    user_roles: list[str],
    k: int = 20,
    alpha: float = 0.65,
) -> list[Chunk]:
    index: RetrievalIndex = load_index(index_path)
    embedding_model = str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    return hybrid_retrieve(query=query, chunks=index.chunks, user_roles=user_roles, k=k, alpha=alpha, embedding_model=embedding_model)