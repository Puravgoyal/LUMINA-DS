"""
ingest.py
─────────────────────────────────────────────────────────────────────────────
LUMINA-DS ETL & Indexing Pipeline.

Pipeline stages
───────────────
  1. load_real_data()     — Parse post_info.txt + json_files-008.zip
  2. chunk_posts()        — Split captions with RecursiveCharacterTextSplitter
  3. EmbeddingEngine      — Wrap BGE-M3 for dense & sparse vectors
  4. MilvusIndexer        — Create schema, build indexes, insert in batches

Incremental ingestion
─────────────────────
  Run with --offset N --limit M to ingest the next M posts starting at row N
  in post_info.txt.  Chunk IDs are deterministic (<post_id>_chunk<i>), so
  re-inserting the same post is safe (Milvus upsert semantics on primary key).

CLI
───
  python ingest.py [--limit 5000] [--offset 0] [--log-level INFO]

  --limit     Number of posts to ingest (default: INGEST_LIMIT from config)
  --offset    Row offset in post_info.txt (default: INGEST_OFFSET from config)
  --log-level Python logging level (default: INFO)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Generator, TypedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
)
from tqdm import tqdm

from config import (
    BGE_MODEL_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DENSE_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    TEXT_CHUNK_MAX_LEN,
    VARCHAR_MAX_LEN,
    configure_logging,
    settings,
)

logger = logging.getLogger(__name__)

# ─── File names inside the raw data directory ─────────────────────────────────
POST_INFO_FILENAME: str = "post_info.txt"
JSON_ZIP_FILENAME: str = "json_files-008.zip"
JSON_ZIP_PREFIX: str = "json/"  # directory prefix inside the zip


# ═════════════════════════════════════════════════════════════════════════════
# TypedDicts — shared data contracts
# ═════════════════════════════════════════════════════════════════════════════


class PostRecord(TypedDict):
    """Represents a single Instagram post extracted from raw data."""

    post_id: str
    caption: str
    timestamp: int        # Unix epoch seconds
    likes: int
    username: str
    hashtags: str         # Space-separated hashtags extracted from caption


class ChunkRecord(TypedDict):
    """A text chunk derived from a PostRecord, ready for embedding."""

    chunk_id: str         # Deterministic: "<post_id>_chunk<index>"
    post_id: str
    text_chunk: str
    likes: int
    timestamp: int
    username: str


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Data Loading
# ═════════════════════════════════════════════════════════════════════════════

_HASHTAG_RE = re.compile(r"#\w+")


def _extract_hashtags(caption: str) -> str:
    """
    Extract all hashtags from *caption* and return them as a space-separated
    string (e.g. ``"#food #travel #love"``).

    Parameters
    ----------
    caption:
        Raw Instagram caption text.

    Returns
    -------
    str
        Space-joined hashtags, or an empty string if none found.
    """
    return " ".join(_HASHTAG_RE.findall(caption))


def _parse_post_json(raw: bytes, username: str) -> PostRecord | None:
    """
    Parse a single Instagram post JSON blob into a :class:`PostRecord`.

    Parameters
    ----------
    raw:
        Raw bytes of the JSON file.
    username:
        Username associated with the post (from post_info.txt).

    Returns
    -------
    PostRecord or None
        *None* is returned when the post has no usable caption text.
    """
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("JSON decode error: %s", exc)
        return None

    # Caption —————————————————————————————————————————————————————————————————
    try:
        caption: str = (
            data["edge_media_to_caption"]["edges"][0]["node"]["text"]
        )
    except (KeyError, IndexError):
        caption = ""

    caption = caption.strip()
    if not caption:
        return None  # Skip posts with no caption

    # Likes ———————————————————————————————————————————————————————————————————
    likes: int = 0
    try:
        likes = int(
            data.get("edge_media_preview_like", {}).get("count", 0)
            or data.get("edge_liked_by", {}).get("count", 0)
            or 0
        )
    except (TypeError, ValueError):
        pass

    # Timestamp ———————————————————————————————————————————————————————————————
    timestamp: int = int(data.get("taken_at_timestamp", 0) or 0)

    # Post ID —————————————————————————————————————————————————————————————————
    post_id: str = str(data.get("id", "")).strip()
    if not post_id:
        return None

    # Owner username (fall back to post_info username) ————————————————————————
    owner_username: str = (
        data.get("owner", {}).get("username", username) or username
    )

    return PostRecord(
        post_id=post_id,
        caption=caption,
        timestamp=timestamp,
        likes=likes,
        username=owner_username,
        hashtags=_extract_hashtags(caption),
    )


def load_real_data(
    limit: int,
    offset: int = 0,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load real Instagram posts from ``post_info.txt`` + ``json_files-008.zip``.

    Parameters
    ----------
    limit:
        Maximum number of posts to return. Posts with empty captions are
        skipped and do **not** count toward this limit.
    offset:
        Number of rows to skip from the start of ``post_info.txt`` before
        beginning to read.  Use for incremental ingestion runs.
    raw_dir:
        Path to the raw data directory.  Defaults to ``settings.data_raw_dir``.

    Returns
    -------
    pd.DataFrame
        Columns: ``post_id``, ``caption``, ``timestamp``, ``likes``,
        ``username``, ``hashtags``.

    Raises
    ------
    FileNotFoundError
        If ``post_info.txt`` or ``json_files-008.zip`` is missing.
    RuntimeError
        If no valid posts are found after scanning.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    post_info_path = raw_dir / POST_INFO_FILENAME
    json_zip_path = raw_dir / JSON_ZIP_FILENAME

    if not post_info_path.exists():
        raise FileNotFoundError(f"post_info.txt not found at {post_info_path}")
    if not json_zip_path.exists():
        raise FileNotFoundError(f"{JSON_ZIP_FILENAME} not found at {json_zip_path}")

    logger.info(
        "Loading posts | offset=%d | limit=%d | zip=%s",
        offset,
        limit,
        json_zip_path.name,
    )

    # Read the post_info index — TSV: index, username, is_video, json_file, imgs
    post_info = pd.read_csv(
        post_info_path,
        sep="\t",
        header=None,
        names=["idx", "username", "is_video", "json_file", "images"],
        dtype={"idx": int, "username": str, "is_video": int, "json_file": str},
        on_bad_lines="skip",
    )

    # Apply offset + limit window
    post_info = post_info.iloc[offset : offset + limit * 3]  # over-fetch to account for empty captions

    records: list[PostRecord] = []
    scanned = 0

    with zipfile.ZipFile(json_zip_path, "r") as zf:
        # Build a lookup set for fast membership test
        zip_names: set[str] = set(zf.namelist())

        for _, row in post_info.iterrows():
            if len(records) >= limit:
                break

            scanned += 1
            json_filename: str = str(row["json_file"]).strip()
            username: str = str(row["username"]).strip()

            # The zip stores files under "json/" prefix
            zip_entry = JSON_ZIP_PREFIX + json_filename
            if zip_entry not in zip_names:
                # Try without prefix
                zip_entry = json_filename
                if zip_entry not in zip_names:
                    logger.debug("Missing zip entry: %s", json_filename)
                    continue

            try:
                with zf.open(zip_entry) as f:
                    raw = f.read()
            except KeyError:
                logger.debug("Cannot open zip entry: %s", zip_entry)
                continue

            record = _parse_post_json(raw, username)
            if record is not None:
                records.append(record)

    if not records:
        raise RuntimeError(
            f"No valid posts found after scanning {scanned} rows "
            f"(offset={offset}, limit={limit}). "
            "Check that json_files-008.zip contains 'json/' prefixed entries."
        )

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d posts (scanned %d rows, %.1f%% yield)",
        len(df),
        scanned,
        100 * len(df) / max(scanned, 1),
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Text Chunking
# ═════════════════════════════════════════════════════════════════════════════


def chunk_posts(df: pd.DataFrame) -> list[ChunkRecord]:
    """
    Split each Instagram post caption into overlapping text chunks.

    Uses LangChain's :class:`RecursiveCharacterTextSplitter` with
    ``CHUNK_SIZE`` and ``CHUNK_OVERLAP`` from config.  Each chunk
    carries the original post's scalar metadata (likes, timestamp, username).

    Chunk IDs are **deterministic** — ``"<post_id>_chunk<index>"`` — so that
    re-running ingest on the same posts does not create duplicate entries in
    Milvus (upsert by primary key).

    Parameters
    ----------
    df:
        DataFrame returned by :func:`load_real_data`.

    Returns
    -------
    list[ChunkRecord]
        Flat list of chunk records ready for embedding + indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[ChunkRecord] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking posts", unit="post"):
        post_id: str = str(row["post_id"])
        caption: str = str(row["caption"])

        # Prepend hashtags to the chunk if they are not already in the caption
        hashtags: str = str(row.get("hashtags", "")).strip()
        full_text = caption  # hashtags already embedded in caption

        splits = splitter.split_text(full_text)

        for chunk_idx, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Truncate to Milvus VARCHAR limit (safety guard)
            if len(chunk_text) > TEXT_CHUNK_MAX_LEN:
                chunk_text = chunk_text[:TEXT_CHUNK_MAX_LEN]

            all_chunks.append(
                ChunkRecord(
                    chunk_id=f"{post_id}_chunk{chunk_idx}",
                    post_id=post_id,
                    text_chunk=chunk_text,
                    likes=int(row.get("likes", 0)),
                    timestamp=int(row.get("timestamp", 0)),
                    username=str(row.get("username", ""))[:VARCHAR_MAX_LEN],
                )
            )

    logger.info(
        "Chunking complete | posts=%d | chunks=%d | avg_chunks_per_post=%.2f",
        len(df),
        len(all_chunks),
        len(all_chunks) / max(len(df), 1),
    )
    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Embedding Engine
# ═════════════════════════════════════════════════════════════════════════════


class EmbeddingEngine:
    """
    Wrapper around SentenceTransformer providing dense and sparse embeddings.

    Uses `SentenceTransformer` for dense embeddings and `HashingVectorizer`
    for instantaneous, fully local, training-free sparse representations.
    """

    def __init__(
        self,
        model_name: str = BGE_MODEL_NAME,
        use_fp16: bool = True,
    ) -> None:
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self._dense_model = SentenceTransformer(model_name)
        # norm='l2' ensures sparse vectors are normalized.
        # n_features=2**16 limits vocabulary space safely for memory.
        self._sparse_model = HashingVectorizer(n_features=2**16, norm="l2", alternate_sign=False)
        logger.info("Models loaded successfully.")

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """
        Generate dense and sparse embeddings for a list of texts.
        """
        dense_vectors = self._dense_model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()

        # Generate sparse lexical weights
        sparse_matrix = self._sparse_model.transform(texts)
        sparse_vectors: list[dict[int, float]] = []
        for row in sparse_matrix:
            # Convert sparse row into a {int: float} dict
            sparse_vectors.append({
                int(idx): float(val) 
                for idx, val in zip(row.indices, row.data)
            })

        return dense_vectors, sparse_vectors

        return dense_vectors, sparse_vectors

    def embed_query(
        self, query: str
    ) -> tuple[list[float], dict[int, float]]:
        """
        Convenience wrapper for embedding a single query string.

        Parameters
        ----------
        query:
            The search query text.

        Returns
        -------
        tuple[list[float], dict[int, float]]
            ``(dense_vector, sparse_vector)`` for the query.
        """
        dense_list, sparse_list = self.embed([query])
        return dense_list[0], sparse_list[0]


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 — Milvus Schema & Indexer
# ═════════════════════════════════════════════════════════════════════════════


def _build_schema() -> CollectionSchema:
    """
    Build the Milvus collection schema for ``lumina_instagram_posts``.

    Fields
    ------
    chunk_id        VARCHAR(512)  — Primary key, deterministic chunk identifier
    post_id         VARCHAR(512)  — Source post identifier
    text_chunk      VARCHAR(2000) — The embedded text content
    likes           INT64         — Like count of the parent post
    timestamp       INT64         — Unix epoch of the parent post
    username        VARCHAR(512)  — Instagram username of the author
    dense_vector    FLOAT_VECTOR  — BGE-M3 dense embedding (dim=1024)
    sparse_vector   SPARSE_FLOAT_VECTOR — BGE-M3 lexical sparse weights

    Returns
    -------
    CollectionSchema
    """
    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=VARCHAR_MAX_LEN,
            is_primary=True,
            auto_id=False,
            description="Deterministic chunk identifier: <post_id>_chunk<n>",
        ),
        FieldSchema(
            name="post_id",
            dtype=DataType.VARCHAR,
            max_length=VARCHAR_MAX_LEN,
            description="Source Instagram post ID",
        ),
        FieldSchema(
            name="text_chunk",
            dtype=DataType.VARCHAR,
            max_length=TEXT_CHUNK_MAX_LEN,
            description="Text chunk derived from the post caption",
        ),
        FieldSchema(
            name="likes",
            dtype=DataType.INT64,
            description="Like count of the source post",
        ),
        FieldSchema(
            name="timestamp",
            dtype=DataType.INT64,
            description="Unix epoch timestamp of the source post",
        ),
        FieldSchema(
            name="username",
            dtype=DataType.VARCHAR,
            max_length=VARCHAR_MAX_LEN,
            description="Instagram username of the post author",
        ),
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=DENSE_DIM,
            description="BGE-M3 dense embedding (1024-dim)",
        ),
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            description="BGE-M3 lexical sparse weights (BM25-style)",
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description="LUMINA-DS Instagram post chunks with hybrid embeddings",
        enable_dynamic_field=False,
    )


class MilvusIndexer:
    """
    Manages Milvus collection lifecycle and batch insertion.

    Parameters
    ----------
    uri:
        Milvus connection URI (Lite file path or server address).
    collection_name:
        Name of the Milvus collection to create / connect to.

    Examples
    --------
    >>> indexer = MilvusIndexer()
    >>> indexer.ensure_collection()
    >>> indexer.insert_batch(chunks, dense_vecs, sparse_vecs)
    >>> indexer.flush()
    """

    def __init__(
        self,
        uri: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._uri = uri or settings.milvus_uri
        self._collection_name = collection_name
        self._client = MilvusClient(uri=self._uri)
        logger.info("MilvusClient connected | uri=%s", self._uri)

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """
        Create the collection (with indexes) if it does not already exist.

        If the collection already exists, this method is a no-op — existing
        data is preserved, enabling incremental ingestion across multiple runs.
        """
        if self._client.has_collection(self._collection_name):
            logger.info(
                "Collection '%s' already exists — skipping creation.",
                self._collection_name,
            )
            return

        logger.info("Creating collection '%s' …", self._collection_name)
        schema = _build_schema()
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
        )

        # ── Dense index (HNSW — fast approximate nearest neighbour) ───────────
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="IP",
            params={"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        self._client.create_index(
            collection_name=self._collection_name,
            index_params=index_params,
        )

        logger.info(
            "Collection '%s' created with HNSW + SPARSE_INVERTED_INDEX indexes.",
            self._collection_name,
        )

    # ── Batch insertion ───────────────────────────────────────────────────────

    def insert_batch(
        self,
        chunks: list[ChunkRecord],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> int:
        """
        Insert a batch of chunk records into Milvus.

        Parameters
        ----------
        chunks:
            List of :class:`ChunkRecord` dicts (metadata).
        dense_vectors:
            Corresponding dense embedding vectors.
        sparse_vectors:
            Corresponding sparse embedding dicts.

        Returns
        -------
        int
            Number of records successfully inserted.
        """
        data = [
            {
                "chunk_id": c["chunk_id"],
                "post_id": c["post_id"],
                "text_chunk": c["text_chunk"],
                "likes": c["likes"],
                "timestamp": c["timestamp"],
                "username": c["username"],
                "dense_vector": dv,
                "sparse_vector": sv,
            }
            for c, dv, sv in zip(chunks, dense_vectors, sparse_vectors)
        ]

        result = self._client.upsert(
            collection_name=self._collection_name,
            data=data,
        )
        inserted = result.get("upsert_count", len(data))
        return inserted

    def flush(self) -> None:
        """Flush all buffered data to persistent storage."""
        self._client.flush(self._collection_name)
        logger.info("Flush complete for '%s'.", self._collection_name)

    def close(self) -> None:
        """Release the Milvus client connection."""
        self._client.close()
        logger.info("MilvusClient connection closed.")

    @property
    def collection_stats(self) -> dict[str, Any]:
        """Return basic collection statistics (entity count etc.)."""
        return self._client.get_collection_stats(self._collection_name)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator — ties all stages together
# ═════════════════════════════════════════════════════════════════════════════


def run_pipeline(limit: int, offset: int, batch_size: int) -> None:
    """
    Execute the full LUMINA-DS ETL & indexing pipeline.

    Parameters
    ----------
    limit:
        Maximum number of posts to ingest.
    offset:
        Row offset in post_info.txt (for incremental runs).
    batch_size:
        Number of chunks to embed + insert per batch iteration.
    """
    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    logger.info("═══ Stage 1/4: Loading data (offset=%d, limit=%d) ═══", offset, limit)
    df = load_real_data(limit=limit, offset=offset)

    # ── Stage 2: Chunk ────────────────────────────────────────────────────────
    logger.info("═══ Stage 2/4: Chunking posts ═══")
    chunks: list[ChunkRecord] = chunk_posts(df)

    # ── Stage 3 + 4: Embed & Index ────────────────────────────────────────────
    logger.info("═══ Stage 3/4: Initialising embedding engine ═══")
    engine = EmbeddingEngine()

    logger.info("═══ Stage 4/4: Indexing into Milvus ═══")
    indexer = MilvusIndexer()
    indexer.ensure_collection()

    total_inserted = 0
    total_chunks = len(chunks)

    progress = tqdm(
        total=total_chunks,
        desc="Embedding + Indexing",
        unit="chunk",
    )

    for start in range(0, total_chunks, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        texts = [c["text_chunk"] for c in batch_chunks]

        dense_vecs, sparse_vecs = engine.embed(texts)
        inserted = indexer.insert_batch(batch_chunks, dense_vecs, sparse_vecs)
        total_inserted += inserted
        progress.update(len(batch_chunks))

    progress.close()
    indexer.flush()
    indexer.close()

    stats = {"total_posts": len(df), "total_chunks": total_chunks, "inserted": total_inserted}
    logger.info(
        "Pipeline complete | posts=%d | chunks=%d | inserted=%d",
        stats["total_posts"],
        stats["total_chunks"],
        stats["inserted"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "LUMINA-DS ingestion pipeline. "
            "Loads real Instagram data, generates BGE-M3 embeddings, "
            "and indexes into Milvus."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.ingest_limit,
        help="Maximum number of posts to ingest in this run.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=settings.ingest_offset,
        help=(
            "Row offset in post_info.txt to start from. "
            "Use for incremental runs: e.g., --offset 5000 --limit 5000 "
            "to ingest the second batch."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Number of chunks to embed and insert per batch.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging verbosity level.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    configure_logging(args.log_level)

    logger.info(
        "LUMINA-DS Ingestion | limit=%d | offset=%d | batch_size=%d",
        args.limit,
        args.offset,
        args.batch_size,
    )

    run_pipeline(
        limit=args.limit,
        offset=args.offset,
        batch_size=args.batch_size,
    )
