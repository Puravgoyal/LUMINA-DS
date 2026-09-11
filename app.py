"""
app.py
─────────────────────────────────────────────────────────────────────────────
LUMINA-DS FastAPI Application.

Serves a single POST /search endpoint that:
  1. Accepts a search query + optional min_likes filter
  2. Retrieves top-5 Instagram post chunks via LuminaRetriever
  3. Synthesises a grounded LLM response via Ollama (local GPU-accelerated LLM)
  4. Returns the answer + raw retrieved posts for full observability

Startup
───────
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints
─────────
  GET  /health   — liveness probe
  POST /search   — main RAG endpoint
  GET  /docs     — Swagger UI (automatic)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import zipfile
import glob
import io
import time

from config import TOP_K, configure_logging, settings
from retrieval import LuminaRetriever, RetrievedDoc

# ─── Bootstrap logging ────────────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are LUMINA, an AI assistant for an Instagram discovery engine.

Rules:
- Answer the user's query using ONLY the provided context chunks below.
- Each chunk is labelled with its Post ID, author username, like count, and timestamp.
- Cite specific posts in your answer using the format [Post <post_id> by @<username>].
- If none of the provided chunks contain relevant information, say so honestly.
- Be concise, insightful, and engaging — as befits an Instagram discovery tool.
"""

_HUMAN_PROMPT = """\
User Query: {query}

Retrieved Instagram Post Chunks:
──────────────────────────────────
{context}
──────────────────────────────────

Please provide a helpful, grounded answer citing the relevant posts above.
"""


# ═════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═════════════════════════════════════════════════════════════════════════════


class SearchQuery(BaseModel):
    """
    Request body for the POST /search endpoint.

    Attributes
    ----------
    query:
        Natural language search query (required, non-empty).
    min_likes:
        Minimum like count filter applied at the vector database layer.
        Posts with fewer likes are excluded from retrieval.
        Pass ``0`` (default) to disable filtering.
    top_k:
        Number of final documents to return and inject into the LLM context.
        Capped at 10 for latency reasons.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query.",
        examples=["healthy food recipes with high engagement"],
    )
    min_likes: int = Field(
        default=0,
        ge=0,
        description="Minimum like count filter (0 = no filter).",
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=10,
        description="Number of results to retrieve and inject into context.",
    )


class RetrievedPostResponse(BaseModel):
    """Serialisable representation of a single retrieved chunk."""

    chunk_id: str
    post_id: str
    text_chunk: str
    likes: int
    timestamp: int
    username: str
    rrf_score: float
    rerank_score: float


class SearchResponse(BaseModel):
    """
    Response payload for the POST /search endpoint.

    Attributes
    ----------
    query:
        Echo of the original search query.
    answer:
        LLM-synthesised answer grounded in the retrieved chunks.
    retrieved_posts:
        Raw top-k retrieved posts for observability / debugging.
    latency_ms:
        End-to-end request latency in milliseconds.
    """

    query: str
    answer: str
    retrieved_posts: list[RetrievedPostResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str
    timestamp: str
    collection: str


# ═════════════════════════════════════════════════════════════════════════════
# Application State — initialised in lifespan
# ═════════════════════════════════════════════════════════════════════════════


class _AppState:
    retriever: LuminaRetriever
    llm_chain: Any  # LangChain Runnable: prompt | llm | parser


_state = _AppState()


# ═════════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown hooks
# ═════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Initialises heavy resources (BGE-M3, CrossEncoder, Milvus client,
    LangChain chain) on startup and gracefully releases them on shutdown.
    """
    logger.info("LUMINA-DS startup — initialising retriever and LLM chain …")

    # ── LuminaRetriever ───────────────────────────────────────────────────────
    try:
        _state.retriever = LuminaRetriever()
    except Exception as exc:
        logger.critical("Failed to initialise LuminaRetriever: %s", exc, exc_info=True)
        raise RuntimeError(
            "Cannot connect to Milvus or load embedding models. "
            "Ensure Milvus is running and models are downloaded."
        ) from exc

    # ── LangChain + Ollama chain ──────────────────────────────────────────────────
    logger.info(
        "Initialising Ollama LLM | model=%s | base_url=%s | num_gpu=%s",
        settings.ollama_model,
        settings.ollama_base_url,
        settings.ollama_num_gpu,
    )
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
        num_predict=1024,        # max tokens to generate
        num_gpu=settings.ollama_num_gpu,  # -1 = all layers on GPU
        num_ctx=4096,            # context window size
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ]
    )
    _state.llm_chain = prompt | llm | StrOutputParser()
    logger.info("Ollama LLM chain ready | model=%s", settings.ollama_model)

    logger.info("LUMINA-DS startup complete. Ready to serve requests.")

    yield  # ── Application runs ──────────────────────────────────────────────

    logger.info("LUMINA-DS shutdown — releasing resources …")
    try:
        _state.retriever.close()
    except Exception as exc:
        logger.warning("Error during retriever teardown: %s", exc)
    logger.info("Shutdown complete.")


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="LUMINA-DS",
    summary="Latent Understanding Model for INstagram Assets — Hybrid RAG API",
    description=(
        "End-to-end Hybrid RAG pipeline ingesting Instagram posts, "
        "embedding with BGE-M3, indexing in Milvus, and synthesising "
        "LLM responses grounded in retrieved content."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ────────────────────────────────────────────────

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Return a clean 503 when Milvus or model resources are unavailable."""
    logger.error("RuntimeError: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service temporarily unavailable.",
            "detail": str(exc),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that prevents stack traces leaking to clients."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error.",
            "detail": "An unexpected error occurred. Check server logs.",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Observability"],
)
async def health() -> HealthResponse:
    """
    Liveness probe — confirms the service is running.

    Returns a 200 OK with service metadata.  Does **not** check whether
    Milvus has data; use ``/search`` to validate end-to-end retrieval.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        collection=settings.milvus_uri,
    )


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid RAG search",
    tags=["Search"],
)
async def search(request: SearchQuery) -> SearchResponse:
    """
    End-to-end Hybrid RAG search endpoint.

    **Pipeline:**
    1. Embeds the query with BGE-M3 (dense + sparse).
    2. Executes Milvus hybrid search fusing dense + sparse ANN results via RRF.
    3. Reranks top-60 candidates with CrossEncoder (BAAI/bge-reranker-base).
    4. Injects top-k chunks into a structured LangChain prompt.
    5. Returns the LLM answer alongside raw retrieved posts for observability.

    **Filters:**
    - ``min_likes``: exclude posts with fewer likes at the DB layer.

    **Error handling:**
    - Returns 503 if Milvus is unreachable.
    - Returns 424 if retrieval returns no results.
    """
    import time

    start_time = time.perf_counter()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    try:
        docs: list[RetrievedDoc] = _state.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_likes=request.min_likes,
        )
    except Exception as exc:
        logger.exception("Retrieval failed for query=%r: %s", request.query, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector database retrieval failed. "
                "Ensure Milvus is running and the collection is loaded."
            ),
        ) from exc

    if not docs:
        raise HTTPException(
            status_code=424,
            detail=(
                "No relevant posts found for this query. "
                "Try relaxing the min_likes filter or broadening the query."
            ),
        )

    # ── Build LLM context block ───────────────────────────────────────────────
    context_parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        # Convert Unix timestamp to readable date
        try:
            post_date = datetime.fromtimestamp(doc["timestamp"], tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        except (OSError, ValueError, OverflowError):
            post_date = "unknown date"

        context_parts.append(
            f"[{i}] Post ID: {doc['post_id']}\n"
            f"     Author: @{doc['username']}\n"
            f"     Likes: {doc['likes']:,}\n"
            f"     Date: {post_date}\n"
            f"     Text: {doc['text_chunk']}"
        )

    context_block = "\n\n".join(context_parts)

    # ── LLM synthesis ─────────────────────────────────────────────────────────
    try:
        answer: str = await _state.llm_chain.ainvoke(
            {"query": request.query, "context": context_block}
        )
    except Exception as exc:
        logger.exception("LLM synthesis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM synthesis failed. "
                "Ensure Ollama is running ('ollama serve') and the model is pulled "
                "('ollama pull llama3.2')."
            ),
        ) from exc

    # ── Build response ────────────────────────────────────────────────────────
    latency_ms = (time.perf_counter() - start_time) * 1000

    retrieved_posts = [
        RetrievedPostResponse(
            chunk_id=doc["chunk_id"],
            post_id=doc["post_id"],
            text_chunk=doc["text_chunk"],
            likes=doc["likes"],
            timestamp=doc["timestamp"],
            username=doc["username"],
            rrf_score=doc["rrf_score"],
            rerank_score=doc["rerank_score"],
        )
        for doc in docs
    ]

    logger.info(
        "Search complete | query=%r | docs=%d | latency=%.1fms",
        request.query,
        len(docs),
        latency_ms,
    )

    return SearchResponse(
        query=request.query,
        answer=answer,
        retrieved_posts=retrieved_posts,
        latency_ms=round(latency_ms, 2),
    )


# ── Image Serving ────────────────────────────────────────────────────────────

_IMAGE_ZIP_CACHE: dict[str, tuple[str, str]] = {}

def _build_image_cache():
    if _IMAGE_ZIP_CACHE:
        return
    logger.info("Building image zip cache (this takes ~12s on first request)...")
    start = time.time()
    for z in glob.glob(str(settings.data_raw_dir / "img_*.zip")):
        with zipfile.ZipFile(z, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".jpg"):
                    pid = name.split("/")[-1].split(".")[0]
                    _IMAGE_ZIP_CACHE[pid] = (z, name)
    logger.info(f"Indexed {len(_IMAGE_ZIP_CACHE)} images in {time.time()-start:.2f}s")

@app.get("/image/{post_id}", tags=["Retrieval"])
async def get_image(post_id: str):
    """Serve the raw .jpg image for a given post ID directly from the zip files."""
    _build_image_cache()
    if post_id not in _IMAGE_ZIP_CACHE:
        raise HTTPException(status_code=404, detail="Image not found in local dataset")
    
    zip_path, inner_path = _IMAGE_ZIP_CACHE[post_id]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(inner_path) as f:
                img_data = f.read()
                return StreamingResponse(io.BytesIO(img_data), media_type="image/jpeg")
    except Exception as e:
        logger.error("Failed to read image %s from %s: %s", inner_path, zip_path, e)
        raise HTTPException(status_code=500, detail="Failed to read image")
