from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .types import Chunk


class LedgerStore(Protocol):
    def load(self, conversation_id: str, total_budget: int, factory: callable) -> Any:
        ...

    def save(self, conversation_id: str, ledger: Any, *, audit_id: str | None = None, principal_id: str | None = None, turn: int | None = None) -> None:
        ...


@dataclass(slots=True)
class ContextLedger:
    total_budget: int
    tokens_used: int = 0
    selected_embeddings: list[list[float]] = field(default_factory=list)
    selected_ids: set[str] = field(default_factory=set)
    selected_chunk_ids: list[str] = field(default_factory=list)
    turn_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        return max(0, self.total_budget - self.tokens_used)

    def cross_turn_similarity(self, candidate_embeddings: list[list[float]]) -> list[float]:
        if not candidate_embeddings:
            return []
        if not self.selected_embeddings:
            return [0.0 for _ in candidate_embeddings]
        held = np.array(self.selected_embeddings, dtype=float)
        cands = np.array(candidate_embeddings, dtype=float)
        held = held / np.maximum(np.linalg.norm(held, axis=1, keepdims=True), 1e-8)
        cands = cands / np.maximum(np.linalg.norm(cands, axis=1, keepdims=True), 1e-8)
        return (cands @ held.T).max(axis=1).astype(float).tolist()

    def add_chunks(self, chunks: list[Chunk], turn: int, query_id: str | None = None, faithfulness: float | None = None) -> int:
        added_tokens = 0
        added_ids: list[str] = []
        for chunk in chunks:
            self.selected_embeddings.append(list(chunk.embedding))
            self.selected_ids.add(chunk.id)
            self.selected_chunk_ids.append(chunk.id)
            self.tokens_used += chunk.token_count
            added_tokens += chunk.token_count
            added_ids.append(chunk.id)
        self.turn_history.append(
            {
                "turn": turn,
                "query_id": query_id,
                "n_chunks": len(chunks),
                "tokens": added_tokens,
                "remaining_budget": self.remaining_budget,
                "chunk_ids": added_ids,
                "faithfulness": faithfulness,
            }
        )
        return added_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "tokens_used": self.tokens_used,
            "selected_embeddings": self.selected_embeddings,
            "selected_chunk_ids": self.selected_chunk_ids,
            "turn_history": self.turn_history,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextLedger":
        chunk_ids = list(payload.get("selected_chunk_ids", []))
        return cls(
            total_budget=int(payload.get("total_budget", 0)),
            tokens_used=int(payload.get("tokens_used", 0)),
            selected_embeddings=[list(row) for row in payload.get("selected_embeddings", [])],
            selected_ids=set(chunk_ids),
            selected_chunk_ids=chunk_ids,
            turn_history=list(payload.get("turn_history", [])),
        )
