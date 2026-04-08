from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    embedding: list[float]
    token_count: int
    roles: list[str]
    relevance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            id=payload["id"],
            document_id=payload["document_id"],
            text=payload["text"],
            embedding=list(payload["embedding"]),
            token_count=int(payload["token_count"]),
            roles=list(payload.get("roles", [])),
            relevance=float(payload.get("relevance", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class RetrievalIndex:
    version: int
    chunks: list[Chunk]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "chunk_count": len(self.chunks),
            "config": self.config,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalIndex":
        return cls(
            version=int(payload.get("version", 1)),
            chunks=[Chunk.from_dict(item) for item in payload.get("chunks", [])],
            config=dict(payload.get("config", {})),
        )
