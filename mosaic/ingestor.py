from __future__ import annotations

"""Index construction for synthetic and manifest-backed corpora."""

from pathlib import Path
from typing import Any

from .types import Chunk, RetrievalIndex
from .utils import DEFAULT_EMBEDDING_MODEL, chunk_text, count_tokens, dump_json, embed_texts, load_json, parse_roles, slugify

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None


SUPPORTED_SOURCE_EXTENSIONS = (".md", ".markdown", ".txt")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    metadata: dict[str, str] = {}
    closing_index = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            closing_index = index
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()

    if closing_index is None:
        return {}, text
    return metadata, "\n".join(lines[closing_index + 1 :]).strip()


def _coerce_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"roles", "title", "doc_type"}:
            continue
        if not isinstance(value, str):
            coerced[key] = value
            continue
        if value.lstrip("-").isdigit():
            coerced[key] = int(value)
            continue
        try:
            coerced[key] = float(value)
            continue
        except ValueError:
            coerced[key] = value
    return coerced


def _normalize_roles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return parse_roles(value)
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Unsupported roles value: {value!r}")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sidecar_candidates(path: Path) -> list[Path]:
    return [
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(".meta.json"),
        path.with_suffix(".metadata.json"),
    ]


def _load_sidecar_metadata(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for candidate in _sidecar_candidates(path):
        if candidate.exists():
            loaded = load_json(candidate)
            if not isinstance(loaded, dict):
                raise ValueError(f"Sidecar metadata must be a JSON object: {candidate}")
            payload.update(loaded)
    return payload


def _reserved_metadata_keys() -> set[str]:
    return {"roles", "title", "doc_type", "document_id", "metadata", "path"}


def _merge_metadata_sources(*sources: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        if not source:
            continue
        nested = source.get("metadata")
        if isinstance(nested, dict):
            merged.update(_coerce_metadata(nested))
        merged.update(_coerce_metadata({key: value for key, value in source.items() if key not in _reserved_metadata_keys()}))
    return merged


def _resolve_document_root(source_dir: str | Path | None, manifest_root: Path | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    if manifest_root is not None:
        return manifest_root
    raise ValueError("Provide either a source directory or a manifest path.")


def _load_manifest(manifest_path: str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if manifest_path is None:
        return None, None
    manifest_file = Path(manifest_path)
    payload = load_json(manifest_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_file}")
    source_root_value = payload.get("source_root")
    if source_root_value:
        manifest_root = Path(str(source_root_value))
        if not manifest_root.is_absolute():
            manifest_root = (manifest_file.parent / manifest_root).resolve()
    else:
        manifest_root = manifest_file.parent.resolve()
    return payload, manifest_root


def _discover_directory_files(source: Path, source_extensions: tuple[str, ...]) -> list[Path]:
    allowed = {extension.lower() for extension in source_extensions}
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        files.append(path)
    return files


def _manifest_document_entries(source_root: Path, manifest_payload: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    documents = manifest_payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Manifest must contain a non-empty 'documents' list.")

    default_entry: dict[str, Any] = {}
    for key in ("roles", "title", "doc_type", "metadata"):
        if key in manifest_payload:
            default_entry[key] = manifest_payload[key]

    entries: list[tuple[Path, dict[str, Any]]] = []
    for item in documents:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError("Each manifest document entry must be an object with a 'path'.")
        raw_path = Path(str(item["path"]))
        resolved = raw_path if raw_path.is_absolute() else source_root / raw_path
        entries.append((resolved, {**default_entry, **item}))
    return entries


def _document_payload(
    path: Path,
    source_root: Path,
    *,
    manifest_entry: dict[str, Any] | None = None,
) -> tuple[str, list[str], dict[str, Any], str]:
    frontmatter, body = _parse_frontmatter(path.read_text(encoding="utf-8-sig"))
    sidecar = _load_sidecar_metadata(path)
    entry = manifest_entry or {}
    merged = {**frontmatter, **sidecar, **entry}

    document_id = slugify(str(merged.get("document_id") or path.stem)) or path.stem
    roles = _normalize_roles(merged.get("roles")) or ["public"]
    title = str(merged.get("title") or path.stem.replace("_", " ").title())
    doc_type = str(merged.get("doc_type") or "unknown")
    metadata = _merge_metadata_sources(frontmatter, sidecar, entry)
    metadata.update(
        {
            "title": title,
            "doc_type": doc_type,
            "path": _relative_path(path, source_root),
            "source_kind": "manifest" if manifest_entry is not None else "directory",
        }
    )
    return document_id, roles, metadata, body


def _persist_chroma(chunks: list[Chunk], chroma_dir: str | Path, collection_name: str) -> Path:
    if chromadb is None:
        raise RuntimeError("ChromaDB is not installed. Install chromadb or choose the json vector store.")

    target_dir = Path(chroma_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target_dir))
    collection = client.get_or_create_collection(collection_name)
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=[chunk.embedding for chunk in chunks],
        metadatas=[
            {
                "document_id": chunk.document_id,
                "roles": ",".join(chunk.roles),
                "token_count": chunk.token_count,
                **{key: str(value) for key, value in chunk.metadata.items()},
            }
            for chunk in chunks
        ],
    )
    return target_dir


def ingest_directory(
    source_dir: str | Path | None,
    output_path: str | Path | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = 384,
    vector_store: str = "chroma",
    chroma_dir: str | Path | None = None,
    collection_name: str = "mosaic_chunks",
    manifest_path: str | Path | None = None,
    source_extensions: tuple[str, ...] = SUPPORTED_SOURCE_EXTENSIONS,
) -> RetrievalIndex:
    manifest_payload, manifest_root = _load_manifest(manifest_path)
    source = _resolve_document_root(source_dir, manifest_root)
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    chunk_specs: list[tuple[str, str, list[str], dict[str, Any], str]] = []
    if manifest_payload is not None:
        files_with_entries = _manifest_document_entries(source, manifest_payload)
    else:
        files_with_entries = [(path, None) for path in _discover_directory_files(source, source_extensions)]

    if not files_with_entries:
        raise ValueError(f"No supported source documents found under {source}")

    for path, manifest_entry in files_with_entries:
        if not path.exists():
            raise FileNotFoundError(f"Manifest document does not exist: {path}")
        document_id, roles, metadata, body = _document_payload(path, source, manifest_entry=manifest_entry)
        for index, chunk_body in enumerate(chunk_text(body, chunk_size=chunk_size, overlap=overlap)):
            chunk_specs.append((f"{document_id}::chunk:{index}", document_id, roles, metadata, chunk_body))

    embeddings = embed_texts([spec[4] for spec in chunk_specs], model_name=embedding_model, dimensions=embedding_dimensions)
    chunks = [
        Chunk(
            id=chunk_id,
            document_id=document_id,
            text=chunk_body,
            embedding=embedding,
            token_count=count_tokens(chunk_body),
            roles=list(roles),
            metadata=dict(metadata),
        )
        for (chunk_id, document_id, roles, metadata, chunk_body), embedding in zip(chunk_specs, embeddings)
    ]

    resolved_chroma_dir = None
    if vector_store == "chroma":
        if chroma_dir is not None:
            resolved_chroma_dir = Path(chroma_dir)
        elif output_path is not None:
            resolved_chroma_dir = Path(output_path).parent / "chroma"
        else:
            resolved_chroma_dir = source / ".chroma"
        _persist_chroma(chunks, resolved_chroma_dir, collection_name)

    index = RetrievalIndex(
        version=2,
        chunks=chunks,
        config={
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "vector_store": vector_store,
            "collection_name": collection_name,
            "chroma_dir": str(resolved_chroma_dir) if resolved_chroma_dir is not None else None,
            "source_dir": str(source),
            "manifest_path": str(Path(manifest_path)) if manifest_path is not None else None,
            "corpus_source": "manifest" if manifest_payload is not None else "directory",
            "document_count": len({chunk.document_id for chunk in chunks}),
            "source_extensions": list(source_extensions),
        },
    )

    if output_path is not None:
        dump_json(output_path, index.to_dict())
    return index


def load_index(index_path: str | Path) -> RetrievalIndex:
    payload = Path(index_path).read_text(encoding="utf-8-sig")
    return RetrievalIndex.from_dict(__import__("json").loads(payload))
