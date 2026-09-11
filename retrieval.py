"""
retrieval.py
─────────────────────────────────────────────────────────────────────────────
LUMINA-DS Hybrid Search & Reranking Engine.

Architecture
────────────
  1. Embed query      — BGE-M3 dense + sparse vectors in one forward pass
  2. Hybrid search    — Milvus AnnSearchRequest × 2 (dense + sparse)
                        fused with RRFRanker(k=60) → top-60 candidates
  3. Rerank           — CrossEncoder('BAAI/bge-reranker-base') scores
                        each (query, chunk) pair → sorted top-5 documents

Usage
─────
    from retrieval import LuminaRetriever

    retriever = LuminaRetriever()
    docs = retriever.retrieve("healthy food recipes", min_likes=200)
    for doc in docs:
        print(doc["score"], doc["text_chunk"])
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from sentence_transformers import CrossEncoder

from config import (
    BGE_MODEL_NAME,
    COLLECTION_NAME,
    DENSE_DIM,
    HNSW_EF_SEARCH,
    HYBRID_SEARCH_LIMIT,
    RERANKER_MODEL_NAME,
    RRF_K,
    TOP_K,
    settings,
)

logger = logging.getLogger(__name__)

# Fields retrieved from Milvus for each candidate hit
_OUTPUT_FIELDS: list[str] = [
    "chunk_id",
    "post_id",
    "text_chunk",
    "likes",
    "timestamp",
    "username",
]


# ═════════════════════════════════════════════════════════════════════════════
# TypedDict — public contract for retrieved documents
# ═════════════════════════════════════════════════════════════════════════════


class RetrievedDoc(TypedDict):
    """
    A single reranked document returned by :class:`LuminaRetriever`.

    Attributes
    ----------
    chunk_id:
        Deterministic chunk identifier (``"<post_id>_chunk<n>"``).
    post_id:
        Source Instagram post ID.
    text_chunk:
        The actual text content of this chunk.
    likes:
        Like count of the source post.
    timestamp:
        Unix epoch timestamp of the source post.
    username:
        Instagram username of the post author.
    rrf_score:
        Raw RRF fusion score from Milvus hybrid search.
    rerank_score:
        Cross-Encoder relevance score (higher = more relevant).
    """

    chunk_id: str
    post_id: str
    text_chunk: str
    likes: int
    timestamp: int
    username: str
    rrf_score: float
    rerank_score: float


# ═════════════════════════════════════════════════════════════════════════════
# LuminaRetriever
# ═════════════════════════════════════════════════════════════════════════════


class LuminaRetriever:
    """
    End-to-end hybrid retriever for the LUMINA-DS Instagram RAG pipeline.

    The retriever combines:
    - **Dense search** (BGE-M3 inner-product HNSW)
    - **Sparse search** (BGE-M3 lexical weights + SPARSE_INVERTED_INDEX)
    - **RRF fusion** (Reciprocal Rank Fusion, k=60)
    - **Cross-Encoder reranking** (``BAAI/bge-reranker-base``)

    Parameters
    ----------
    milvus_uri:
        Milvus connection URI.  Defaults to ``settings.milvus_uri``.
    collection_name:
        Milvus collection to query.  Defaults to ``COLLECTION_NAME``.
    bgem3_model_name:
        HuggingFace model ID for BGE-M3.  Defaults to ``BGE_MODEL_NAME``.
    reranker_model_name:
        HuggingFace model ID for the CrossEncoder.
        Defaults to ``RERANKER_MODEL_NAME``.
    use_fp16:
        Whether to load BGE-M3 in FP16.

    Examples
    --------
    >>> retriever = LuminaRetriever()
    >>> docs = retriever.retrieve("summer fashion looks", top_k=5, min_likes=100)
    >>> docs[0]["text_chunk"]
    '...'
    """

    def __init__(
        self,
        milvus_uri: str | None = None,
        collection_name: str = COLLECTION_NAME,
        bgem3_model_name: str = BGE_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        use_fp16: bool = True,
    ) -> None:
        self._collection_name = collection_name

        # ── Milvus client ─────────────────────────────────────────────────────
        _uri = milvus_uri or settings.milvus_uri
        logger.info("Connecting to Milvus | uri=%s", _uri)
        self._client = MilvusClient(uri=_uri)

        # ── Dense & Sparse embedding models ───────────────────────────────────
        logger.info("Loading SentenceTransformer model: %s", bgem3_model_name)
        self._dense_model = SentenceTransformer(bgem3_model_name)
        self._sparse_model = HashingVectorizer(n_features=2**16, norm="l2", alternate_sign=False)
        logger.info("Embedding models loaded.")

        # ── CrossEncoder reranker ─────────────────────────────────────────────
        logger.info("Loading CrossEncoder reranker: %s", reranker_model_name)
        self._reranker = CrossEncoder(reranker_model_name)
        logger.info("CrossEncoder loaded.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_query(
        self, query: str
    ) -> tuple[list[float], dict[int, float]]:
        """
        Generate BGE-M3 dense + sparse embeddings for a query string.

        Parameters
        ----------
        query:
            User search query.

        Returns
        -------
        tuple[list[float], dict[int, float]]
            ``(dense_vector, sparse_vector)``
        """
        dense_matrix = self._dense_model.encode([query], show_progress_bar=False)
        dense: list[float] = dense_matrix[0].tolist()
        
        sparse_matrix = self._sparse_model.transform([query])
        row = sparse_matrix[0]
        sparse: dict[int, float] = {
            int(idx): float(val) for idx, val in zip(row.indices, row.data)
        }
        return dense, sparse

    def _hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int,
        filter_expr: str | None,
    ) -> list[dict[str, Any]]:
        """
        Execute Milvus hybrid search fusing dense + sparse results via RRF.

        Parameters
        ----------
        dense_vector:
            Query dense embedding.
        sparse_vector:
            Query sparse (lexical) weights.
        limit:
            Number of candidate results to retrieve per sub-request.
        filter_expr:
            Optional scalar filter expression (e.g. ``"likes >= 100"``).

        Returns
        -------
        list[dict[str, Any]]
            Raw Milvus hit dicts containing output fields + distance score.
        """
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": HNSW_EF_SEARCH}},
            limit=limit,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=limit,
            expr=filter_expr,
        )

        results = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=RRF_K),
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )

        # results is a list of lists (one per query); we sent exactly one query
        return results[0] if results else []

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[RetrievedDoc]:
        """
        Rerank candidate hits using a CrossEncoder and return top-k documents.

        Parameters
        ----------
        query:
            Original user query string.
        candidates:
            Raw Milvus hit dicts from :meth:`_hybrid_search`.
        top_k:
            Number of documents to return after reranking.

        Returns
        -------
        list[RetrievedDoc]
            Sorted by rerank score descending, length == min(top_k, len(candidates)).
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the CrossEncoder
        pairs = [(query, hit["entity"]["text_chunk"]) for hit in candidates]
        scores: list[float] = self._reranker.predict(pairs).tolist()

        # Zip scores with candidates and sort descending
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[RetrievedDoc] = []
        for rerank_score, hit in scored[:top_k]:
            entity = hit["entity"]
            results.append(
                RetrievedDoc(
                    chunk_id=str(entity.get("chunk_id", "")),
                    post_id=str(entity.get("post_id", "")),
                    text_chunk=str(entity.get("text_chunk", "")),
                    likes=int(entity.get("likes", 0)),
                    timestamp=int(entity.get("timestamp", 0)),
                    username=str(entity.get("username", "")),
                    rrf_score=float(hit.get("distance", 0.0)),
                    rerank_score=float(rerank_score),
                )
            )

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        min_likes: int = 0,
        candidate_limit: int = HYBRID_SEARCH_LIMIT,
    ) -> list[RetrievedDoc]:
        """
        Full retrieval pipeline: embed → hybrid search → rerank.

        Parameters
        ----------
        query:
            Natural language search query.
        top_k:
            Number of final documents to return after reranking.
        min_likes:
            Minimum like count filter applied at the Milvus layer.
            Pass ``0`` to disable filtering.
        candidate_limit:
            Number of candidates retrieved per sub-request (dense + sparse)
            before RRF fusion.  Higher values improve recall at the cost of
            reranking latency.

        Returns
        -------
        list[RetrievedDoc]
            Up to *top_k* documents, sorted by cross-encoder score descending.

        Raises
        ------
        RuntimeError
            If the Milvus collection does not exist or is not loaded.
        """
        logger.info(
            "Retrieving | query=%r | top_k=%d | min_likes=%d",
            query,
            top_k,
            min_likes,
        )

        # Build optional filter expression
        filter_expr: str | None = None
        if min_likes > 0:
            filter_expr = f"likes >= {min_likes}"
            logger.debug("Applying filter: %s", filter_expr)

        # Stage 1: embed
        dense_vec, sparse_vec = self._embed_query(query)

        # Stage 2: hybrid search
        candidates = self._hybrid_search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            limit=candidate_limit,
            filter_expr=filter_expr,
        )
        logger.info("Hybrid search returned %d candidates.", len(candidates))

        # Stage 3: rerank
        docs = self._rerank(query=query, candidates=candidates, top_k=top_k)
        logger.info("Reranking complete | returning %d docs.", len(docs))

        return docs

    def close(self) -> None:
        """Release the Milvus client connection."""
        self._client.close()
        logger.info("LuminaRetriever: Milvus connection closed.")
