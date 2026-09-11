"""
config.py
─────────────────────────────────────────────────────────────────────────────
Centralised configuration for LUMINA-DS.

All runtime knobs are loaded from environment variables (or a `.env` file).
Pure constants (model architecture, schema names) are defined as module-level
literals so they can be imported without instantiating the settings object.

Usage
-----
    from config import settings, DENSE_DIM, COLLECTION_NAME

    print(settings.milvus_uri)          # "http://localhost:19530"
    print(settings.ollama_model)         # "llama3.2"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Repository root (the directory that contains this file) ──────────────────
REPO_ROOT: Path = Path(__file__).parent.resolve()

# ═════════════════════════════════════════════════════════════════════════════
# Pure constants — model / schema invariants
# ═════════════════════════════════════════════════════════════════════════════

# Dense embedding model (sentence-transformers — ~90 MB, reliable download)
BGE_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Dense vector dimension for bge-small-en-v1.5
DENSE_DIM: int = 384

# Text splitting parameters
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Milvus collection name
COLLECTION_NAME: str = "lumina_instagram_posts"

# Maximum VARCHAR length for text_chunk field in Milvus
TEXT_CHUNK_MAX_LEN: int = 2000

# Maximum VARCHAR length for string scalar fields
VARCHAR_MAX_LEN: int = 512

# Hybrid-search candidate pool before reranking
HYBRID_SEARCH_LIMIT: int = 60

# RRF smoothing constant (higher → less aggressive rank fusion)
RRF_K: int = 60

# Final top-k documents returned to the caller after reranking
TOP_K: int = 5

# Cross-Encoder reranker model identifier (~80 MB)
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Default Ollama model (runs locally via Ollama server)
DEFAULT_OLLAMA_MODEL: str = "llama3.2:1b"

# Default Ollama server base URL
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"

# HNSW index build parameters for dense vector
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 256

# Search-time ef parameter for HNSW
HNSW_EF_SEARCH: int = 200

# ═════════════════════════════════════════════════════════════════════════════
# Pydantic-settings — loaded from environment / .env file
# ═════════════════════════════════════════════════════════════════════════════


class LuminaSettings(BaseSettings):
    """
    Runtime configuration for LUMINA-DS.

    All fields can be overridden via environment variables or a `.env` file
    placed in the repository root.  Field names are case-insensitive.

    Attributes
    ----------
    milvus_uri:
        Connection URI for Milvus.
        Use ``"path/to/milvus_demo.db"`` for Milvus Lite (zero-config local)
        or ``"http://localhost:19530"`` for a standalone Milvus server.
    ollama_model:
        Ollama model for local GPU-accelerated LLM synthesis (e.g. ``"llama3.2"``, ``"mistral"``).
    ollama_base_url:
        Ollama server URL. Defaults to ``"http://localhost:11434"``.
    batch_size:
        Number of records inserted per Milvus batch during ingestion.
    ingest_limit:
        Maximum number of Instagram posts to ingest per run.  Set via the
        ``INGEST_LIMIT`` env var or the ``--limit`` CLI flag in ``ingest.py``.
    ingest_offset:
        Row offset into ``post_info.txt`` to start reading from.  Used for
        incremental ingestion runs (i.e. ``--offset`` CLI flag).
    data_raw_dir:
        Absolute path to the directory containing raw data files.
    log_level:
        Python logging level string, e.g. ``"INFO"``, ``"DEBUG"``.
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_uri: str = Field(
        default=str(REPO_ROOT / "milvus_lumina.db"),
        description=(
            "Milvus Lite local DB path (default) "
            "or 'http://host:port' for a standalone server."
        ),
        alias="LUMINA_MILVUS_URI",
    )

    # ── Ollama (local LLM — free, GPU-accelerated) ─────────────────────────────────
    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        description="Ollama model to use for LLM synthesis (e.g. llama3.2, mistral).",
        alias="OLLAMA_MODEL",
    )
    ollama_base_url: str = Field(
        default=DEFAULT_OLLAMA_BASE_URL,
        description="Ollama server base URL. Change if running Ollama on a remote host.",
        alias="OLLAMA_BASE_URL",
    )
    ollama_num_gpu: int = Field(
        default=-1,
        description=(
            "Number of GPU layers to offload to VRAM. "
            "-1 = all layers on GPU (full GPU mode). "
            "0 = CPU only. Intermediate values for partial GPU offload."
        ),
        alias="OLLAMA_NUM_GPU",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    batch_size: int = Field(
        default=100,
        description="Number of records per Milvus insert batch.",
        alias="BATCH_SIZE",
    )
    ingest_limit: int = Field(
        default=5_000,
        description="Max Instagram posts to ingest per run.",
        alias="INGEST_LIMIT",
    )
    ingest_offset: int = Field(
        default=0,
        description=(
            "Row offset in post_info.txt to begin reading from. "
            "Use with --offset for incremental ingestion."
        ),
        alias="INGEST_OFFSET",
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    data_raw_dir: Path = Field(
        default=REPO_ROOT / "data" / "raw",
        description="Directory containing raw data files.",
        alias="DATA_RAW_DIR",
    )

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR).",
        alias="LOG_LEVEL",
    )


# ─── Singleton settings instance ──────────────────────────────────────────────
settings = LuminaSettings()


# ─── Logging bootstrap ────────────────────────────────────────────────────────
def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure the root logger with a structured format.

    Parameters
    ----------
    level:
        Logging level string.  If *None*, falls back to ``settings.log_level``.
    """
    _level = (level or settings.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, _level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger.debug("Logging configured at level=%s", _level)
