from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ledger import ContextLedger
from .types import Chunk
from .utils import cosine_similarity, roles_permit

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


@dataclass(slots=True)
class OptimizationResult:
    selected: list[Chunk]
    weights: list[float]
    tokens_used: int
    objective_history: list[float]
    backend: str
    remaining_budget: int = 0


def build_similarity_matrix(chunks: list[Chunk]) -> list[list[float]]:
    if not chunks:
        return []
    embeddings = np.array([chunk.embedding for chunk in chunks], dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    return (normalized @ normalized.T).tolist()


def objective(
    weights: list[float],
    relevance: list[float],
    sim_matrix: list[list[float]],
    lam: float,
    cross_turn_penalties: list[float] | None = None,
    cross_turn_lambda: float = 0.0,
) -> float:
    weights_array = np.asarray(weights, dtype=float)
    relevance_array = np.asarray(relevance, dtype=float)
    similarity = np.asarray(sim_matrix, dtype=float)
    coverage = float(weights_array @ relevance_array)
    redundancy = float(np.triu(np.outer(weights_array, weights_array) * similarity, k=1).sum())
    cross_turn = 0.0
    if cross_turn_penalties is not None:
        cross_turn = float(weights_array @ np.asarray(cross_turn_penalties, dtype=float))
    return coverage - (lam * redundancy) - (cross_turn_lambda * cross_turn)


def gradient(
    weights: list[float],
    relevance: list[float],
    sim_matrix: list[list[float]],
    lam: float,
    cross_turn_penalties: list[float] | None = None,
    cross_turn_lambda: float = 0.0,
) -> list[float]:
    weights_array = np.asarray(weights, dtype=float)
    relevance_array = np.asarray(relevance, dtype=float)
    similarity = np.asarray(sim_matrix, dtype=float)
    overlap = similarity @ weights_array
    cross_turn = np.asarray(cross_turn_penalties or [0.0] * len(weights), dtype=float)
    grads = relevance_array - (lam * (overlap - np.diag(similarity) * weights_array)) - (cross_turn_lambda * cross_turn)
    return grads.tolist()


def project_budget(
    weights: list[float],
    token_counts: list[int],
    budget: int,
    permitted: list[bool] | None = None,
) -> list[float]:
    projected = []
    for index, weight in enumerate(weights):
        allowed = True if permitted is None else permitted[index]
        projected.append(min(max(weight, 0.0), 1.0) if allowed else 0.0)

    used = sum(weight * tokens for weight, tokens in zip(projected, token_counts))
    if used > budget and used > 0:
        scale = budget / used
        projected = [weight * scale for weight in projected]
    return projected


def greedy_round(
    weights: list[float],
    chunks: list[Chunk],
    token_budget: int,
    permitted: list[bool] | None = None,
) -> list[Chunk]:
    allowed = permitted if permitted is not None else [True] * len(chunks)
    order = sorted(
        range(len(chunks)),
        key=lambda index: (weights[index], chunks[index].relevance, -chunks[index].token_count),
        reverse=True,
    )

    selected: list[Chunk] = []
    tokens_used = 0
    for index in order:
        if not allowed[index]:
            continue
        chunk = chunks[index]
        if tokens_used + chunk.token_count > token_budget:
            continue
        selected.append(chunk)
        tokens_used += chunk.token_count
    return selected


def select_topk(chunks: list[Chunk], token_budget: int, user_roles: list[str]) -> list[Chunk]:
    ordered = [chunk for chunk in sorted(chunks, key=lambda item: item.relevance, reverse=True) if roles_permit(chunk.roles, user_roles)]
    selected: list[Chunk] = []
    used = 0
    for chunk in ordered:
        if used + chunk.token_count > token_budget:
            continue
        selected.append(chunk)
        used += chunk.token_count
    return selected


def select_mmr(
    chunks: list[Chunk],
    token_budget: int,
    user_roles: list[str],
    diversity_lambda: float = 0.7,
) -> list[Chunk]:
    remaining = [chunk for chunk in chunks if roles_permit(chunk.roles, user_roles)]
    selected: list[Chunk] = []
    used = 0

    while remaining:
        best_index = None
        best_score = None
        for index, chunk in enumerate(remaining):
            if used + chunk.token_count > token_budget:
                continue
            if not selected:
                score = chunk.relevance
            else:
                max_similarity = max(cosine_similarity(chunk.embedding, item.embedding) for item in selected)
                score = (diversity_lambda * chunk.relevance) - ((1.0 - diversity_lambda) * max_similarity)
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        used += chosen.token_count

    return selected


def compute_redundancy_score(chunks: list[Chunk]) -> float:
    if len(chunks) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for left in range(len(chunks)):
        for right in range(left + 1, len(chunks)):
            total += cosine_similarity(chunks[left].embedding, chunks[right].embedding)
            pairs += 1
    return total / pairs if pairs else 0.0


def compute_coverage_score(chunks: list[Chunk]) -> float:
    return sum(chunk.relevance for chunk in chunks)


def _cross_turn_penalties(chunks: list[Chunk], ledger: ContextLedger | None) -> list[float]:
    if ledger is None:
        return [0.0 for _ in chunks]
    penalties = ledger.cross_turn_similarity([chunk.embedding for chunk in chunks])
    for index, chunk in enumerate(chunks):
        if chunk.id in ledger.selected_ids:
            penalties[index] = max(penalties[index], 1.0)
    return penalties


if jax is not None:
    @jax.jit
    def _jax_objective(
        weights: jnp.ndarray,
        relevance: jnp.ndarray,
        sim_matrix: jnp.ndarray,
        lam: float,
        cross_turn_penalties: jnp.ndarray,
        cross_turn_lambda: float,
    ) -> jnp.ndarray:
        coverage = jnp.dot(weights, relevance)
        redundancy = jnp.sum(jnp.triu(jnp.outer(weights, weights) * sim_matrix, k=1))
        cross_turn = jnp.dot(weights, cross_turn_penalties)
        return coverage - (lam * redundancy) - (cross_turn_lambda * cross_turn)


    _jax_grad = jax.jit(jax.grad(_jax_objective, argnums=0))


    @jax.jit
    def _jax_project_budget(weights: jnp.ndarray, token_counts: jnp.ndarray, budget: float, permitted: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(weights, 0.0, 1.0) * permitted
        used = jnp.dot(clipped, token_counts)
        scale = jnp.where(used > budget, budget / jnp.maximum(used, 1e-8), 1.0)
        return jnp.clip(clipped * scale, 0.0, 1.0) * permitted


def _effective_budget(token_budget: int, ledger: ContextLedger | None) -> int:
    if ledger is None:
        return max(0, token_budget)
    return max(0, min(token_budget, ledger.remaining_budget))


def _optimize_with_jax(
    chunks: list[Chunk],
    user_roles: list[str],
    token_budget: int,
    lam: float,
    lr: float,
    steps: int,
    ledger: ContextLedger | None,
    cross_turn_lambda: float,
) -> OptimizationResult:
    effective_budget = _effective_budget(token_budget, ledger)
    if effective_budget <= 0:
        return OptimizationResult(selected=[], weights=[], tokens_used=0, objective_history=[], backend="jax", remaining_budget=0)

    permitted = jnp.array([1.0 if roles_permit(chunk.roles, user_roles) else 0.0 for chunk in chunks], dtype=jnp.float32)
    relevance = jnp.array([chunk.relevance for chunk in chunks], dtype=jnp.float32) * permitted
    token_counts = jnp.array([chunk.token_count for chunk in chunks], dtype=jnp.float32)
    sim_matrix = jnp.array(build_similarity_matrix(chunks), dtype=jnp.float32)
    cross_turn = jnp.array(_cross_turn_penalties(chunks, ledger), dtype=jnp.float32) * permitted

    adjusted = jnp.clip(relevance - (cross_turn_lambda * cross_turn), 0.0)
    total_relevance = jnp.sum(adjusted)
    initial = jnp.where(total_relevance > 0, adjusted / jnp.maximum(total_relevance, 1e-8), permitted)
    weights = _jax_project_budget(initial, token_counts, float(effective_budget), permitted)
    history = [float(_jax_objective(weights, relevance, sim_matrix, lam, cross_turn, cross_turn_lambda))]

    for _ in range(steps):
        grads = _jax_grad(weights, relevance, sim_matrix, lam, cross_turn, cross_turn_lambda)
        weights = _jax_project_budget(weights + (lr * grads), token_counts, float(effective_budget), permitted)
        history.append(float(_jax_objective(weights, relevance, sim_matrix, lam, cross_turn, cross_turn_lambda)))

    weight_list = np.asarray(weights, dtype=float).tolist()
    permitted_list = [bool(value) for value in np.asarray(permitted, dtype=float).tolist()]
    selected = greedy_round(weight_list, chunks, token_budget=effective_budget, permitted=permitted_list)
    tokens_used = sum(chunk.token_count for chunk in selected)
    remaining_budget = max(0, effective_budget - tokens_used)
    return OptimizationResult(selected=selected, weights=weight_list, tokens_used=tokens_used, objective_history=history, backend="jax", remaining_budget=remaining_budget)


def _optimize_with_numpy(
    chunks: list[Chunk],
    user_roles: list[str],
    token_budget: int,
    lam: float,
    lr: float,
    steps: int,
    ledger: ContextLedger | None,
    cross_turn_lambda: float,
) -> OptimizationResult:
    effective_budget = _effective_budget(token_budget, ledger)
    if effective_budget <= 0:
        return OptimizationResult(selected=[], weights=[], tokens_used=0, objective_history=[], backend="numpy", remaining_budget=0)

    permitted = [roles_permit(chunk.roles, user_roles) for chunk in chunks]
    relevance = [chunk.relevance if allowed else 0.0 for chunk, allowed in zip(chunks, permitted)]
    token_counts = [chunk.token_count for chunk in chunks]
    cross_turn = [value if allowed else 0.0 for value, allowed in zip(_cross_turn_penalties(chunks, ledger), permitted)]
    adjusted = [max(0.0, value - (cross_turn_lambda * penalty)) for value, penalty in zip(relevance, cross_turn)]
    total_relevance = sum(adjusted)
    if total_relevance > 0:
        weights = [value / total_relevance for value in adjusted]
    else:
        weights = [1.0 if allowed else 0.0 for allowed in permitted]
    weights = project_budget(weights, token_counts, effective_budget, permitted=permitted)

    sim_matrix = build_similarity_matrix(chunks)
    history = [objective(weights, relevance, sim_matrix, lam, cross_turn, cross_turn_lambda)]
    for _ in range(steps):
        grads = gradient(weights, relevance, sim_matrix, lam, cross_turn, cross_turn_lambda)
        updated = [weight + (lr * grad_value) for weight, grad_value in zip(weights, grads)]
        weights = project_budget(updated, token_counts, effective_budget, permitted=permitted)
        history.append(objective(weights, relevance, sim_matrix, lam, cross_turn, cross_turn_lambda))

    selected = greedy_round(weights, chunks, token_budget=effective_budget, permitted=permitted)
    tokens_used = sum(chunk.token_count for chunk in selected)
    return OptimizationResult(selected=selected, weights=weights, tokens_used=tokens_used, objective_history=history, backend="numpy", remaining_budget=max(0, effective_budget - tokens_used))


def optimize(
    chunks: list[Chunk],
    user_roles: list[str],
    token_budget: int,
    lam: float = 0.5,
    lr: float = 0.05,
    steps: int = 100,
    use_jax: bool | None = None,
    ledger: ContextLedger | None = None,
    cross_turn_lambda: float = 0.0,
) -> OptimizationResult:
    if not chunks:
        return OptimizationResult(selected=[], weights=[], tokens_used=0, objective_history=[], backend="none", remaining_budget=max(0, token_budget))

    effective_budget = _effective_budget(token_budget, ledger)
    if effective_budget <= 0:
        return OptimizationResult(selected=[], weights=[], tokens_used=0, objective_history=[], backend="no-budget", remaining_budget=0)

    if lam <= 0 and cross_turn_lambda <= 0:
        selected = select_topk(chunks, token_budget=effective_budget, user_roles=user_roles)
        weights = [1.0 if chunk in selected else 0.0 for chunk in chunks]
        return OptimizationResult(
            selected=selected,
            weights=weights,
            tokens_used=sum(chunk.token_count for chunk in selected),
            objective_history=[],
            backend="topk-shortcut",
            remaining_budget=max(0, effective_budget - sum(chunk.token_count for chunk in selected)),
        )

    should_use_jax = (jax is not None) if use_jax is None else bool(use_jax and jax is not None)
    if should_use_jax:
        return _optimize_with_jax(chunks, user_roles, effective_budget, lam, lr, steps, ledger, cross_turn_lambda)
    return _optimize_with_numpy(chunks, user_roles, effective_budget, lam, lr, steps, ledger, cross_turn_lambda)


def pareto_sweep(
    chunks: list[Chunk],
    user_roles: list[str],
    token_budget: int,
    lam_values: list[float],
    use_jax: bool | None = None,
) -> list[dict]:
    results = []
    for lam in lam_values:
        result = optimize(chunks, user_roles=user_roles, token_budget=token_budget, lam=lam, use_jax=use_jax)
        results.append(
            {
                "lam": lam,
                "n_chunks": len(result.selected),
                "tokens_used": result.tokens_used,
                "coverage_score": compute_coverage_score(result.selected),
                "redundancy_score": compute_redundancy_score(result.selected),
                "backend": result.backend,
            }
        )
    return results
