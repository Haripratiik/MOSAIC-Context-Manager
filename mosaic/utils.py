from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

APPROX_TOKENS_PER_WORD = 2.0
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HASH_EMBEDDING_MODEL = "hash"

DEFAULT_PUBLIC_ROLE = "public"
WILDCARD_ROLE = "*"
DEMO_ROLE_INHERITANCE = {
    "public": {"public"},
    "analyst": {"public", "analyst"},
    "senior_analyst": {"public", "analyst", "senior_analyst"},
    "executive": {"public", "analyst", "executive"},
    "compliance": {"public", "analyst", "senior_analyst", "executive", "compliance"},
}

_SENTENCE_MODEL_CACHE: dict[str, Any] = {}
_SENTENCE_MODEL_FAILURES: set[str] = set()


@contextmanager
def _quiet_external_output() -> Any:
    if os.environ.get("MOSAIC_VERBOSE_EMBEDDINGS") == "1":
        yield
        return
    sink = io.StringIO()
    logger_names = [
        "sentence_transformers",
        "transformers",
        "transformers.modeling_utils",
        "transformers.utils.loading_report",
    ]
    previous_levels = {name: logging.getLogger(name).level for name in logger_names}
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return text.strip("-")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def sentencize(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def parse_roles(raw: str | None) -> list[str]:
    if not raw:
        return [DEFAULT_PUBLIC_ROLE]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_roles(values: Iterable[str]) -> list[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _normalize_role_inheritance(role_inheritance: dict[str, Iterable[str]] | None) -> dict[str, set[str]]:
    if not role_inheritance:
        return {}
    normalized: dict[str, set[str]] = {}
    for role, implied in role_inheritance.items():
        key = str(role).strip().lower()
        if not key:
            continue
        scope = {key, DEFAULT_PUBLIC_ROLE}
        scope.update(_normalize_roles(implied))
        normalized[key] = scope
    return normalized


def resolve_user_roles(user_roles: Iterable[str], role_inheritance: dict[str, Iterable[str]] | None = None) -> set[str]:
    normalized_roles = _normalize_roles(user_roles)
    if not normalized_roles:
        return {DEFAULT_PUBLIC_ROLE}
    if WILDCARD_ROLE in normalized_roles:
        return {WILDCARD_ROLE}

    resolved: set[str] = {DEFAULT_PUBLIC_ROLE}
    normalized_inheritance = _normalize_role_inheritance(role_inheritance)
    use_demo_inheritance = not normalized_inheritance and all(role in DEMO_ROLE_INHERITANCE for role in normalized_roles)

    for role in normalized_roles:
        if normalized_inheritance:
            resolved.update(normalized_inheritance.get(role, {DEFAULT_PUBLIC_ROLE, role}))
        elif use_demo_inheritance:
            resolved.update(DEMO_ROLE_INHERITANCE.get(role, {DEFAULT_PUBLIC_ROLE, role}))
        else:
            resolved.add(role)
    return resolved


def roles_permit(
    chunk_roles: list[str],
    user_roles: list[str],
    role_inheritance: dict[str, Iterable[str]] | None = None,
) -> bool:
    if not chunk_roles:
        return True
    required = {role.strip().lower() for role in chunk_roles if role.strip()}
    resolved_roles = resolve_user_roles(user_roles, role_inheritance=role_inheritance)
    if WILDCARD_ROLE in resolved_roles:
        return True
    return bool(required & resolved_roles)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        if maximum <= 0:
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _get_encoding(encoding_name: str = "cl100k_base"):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = _get_encoding(encoding_name)
    if encoding is None:
        return max(1, int(math.ceil(len(text.split()) * APPROX_TOKENS_PER_WORD)))
    return max(1, len(encoding.encode(text)))


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64, encoding_name: str = "cl100k_base") -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    encoding = _get_encoding(encoding_name)
    if encoding is None:
        words = text.split()
        approx_chunk_size = max(32, int(chunk_size / APPROX_TOKENS_PER_WORD))
        approx_overlap = max(8, int(overlap / APPROX_TOKENS_PER_WORD))
        step = max(1, approx_chunk_size - approx_overlap)
        return [
            " ".join(words[start : start + approx_chunk_size])
            for start in range(0, len(words), step)
            if words[start : start + approx_chunk_size]
        ]

    token_ids = encoding.encode(text)
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_size]
        if not window:
            break
        decoded = encoding.decode(window).strip()
        if decoded:
            chunks.append(decoded)
    return chunks


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def hashed_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        magnitude = 1.0 + (digest[3] / 255.0)
        vector[index] += sign * magnitude
    return normalize_vector(vector)


def resolve_embedding_model(model_name: str | None) -> str:
    raw = (model_name or DEFAULT_EMBEDDING_MODEL).strip()
    return raw or DEFAULT_EMBEDDING_MODEL


def embed_texts(texts: list[str], model_name: str | None = None, dimensions: int = 384) -> list[list[float]]:
    resolved_model = resolve_embedding_model(model_name)
    normalized = resolved_model.lower()
    if normalized == HASH_EMBEDDING_MODEL:
        return [hashed_embedding(text, dimensions=dimensions) for text in texts]

    if SentenceTransformer is not None and resolved_model not in _SENTENCE_MODEL_FAILURES:
        try:
            if resolved_model not in _SENTENCE_MODEL_CACHE:
                with _quiet_external_output():
                    _SENTENCE_MODEL_CACHE[resolved_model] = SentenceTransformer(resolved_model, local_files_only=True)
            with _quiet_external_output():
                encoded = _SENTENCE_MODEL_CACHE[resolved_model].encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return [list(map(float, row)) for row in encoded]
        except TypeError:
            with _quiet_external_output():
                encoded = _SENTENCE_MODEL_CACHE[resolved_model].encode(texts, normalize_embeddings=True)
            return [list(map(float, row)) for row in encoded]
        except Exception:
            _SENTENCE_MODEL_FAILURES.add(resolved_model)

    return [hashed_embedding(text, dimensions=dimensions) for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_context(chunks: list[Any]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        title = str(chunk.metadata.get("title", chunk.document_id)) if getattr(chunk, "metadata", None) else chunk.document_id
        parts.append(f"[{title}]\n{chunk.text}")
    return "\n\n".join(parts)


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

