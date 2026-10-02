"""
EmbeddingService — local in-process embedding via FastEmbed (ONNX Runtime)

Replaces Zep Cloud's built-in embedding with a local multilingual model,
running inside the backend process (no Ollama or other embedding server needed).
Produces 768-dimensional vectors, matching the Neo4j vector index.
"""

import logging
import threading
from typing import List, Optional

from ..config import Config

logger = logging.getLogger('mirofish.embedding')

EMBEDDING_DIM = 768

# nomic-embed-text models are trained with task prefixes; other models use none
NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_DOCUMENT_PREFIX = "search_document: "

# One model instance per process (loading takes a few seconds and ~500MB RAM)
_model = None
_model_lock = threading.Lock()


def _get_model(model_name: str, cache_dir: Optional[str]):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding
                logger.info(f"Loading embedding model {model_name} (first use may download it)...")
                _model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
                logger.info("Embedding model loaded")
    return _model


class EmbeddingService:
    """Generate embeddings locally with FastEmbed."""

    def __init__(
        self,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model = model or Config.EMBEDDING_MODEL
        self.cache_dir = cache_dir or Config.EMBEDDING_CACHE_DIR

        is_nomic = 'nomic' in self.model.lower()
        self._query_prefix = NOMIC_QUERY_PREFIX if is_nomic else ""
        self._document_prefix = NOMIC_DOCUMENT_PREFIX if is_nomic else ""

        # Simple in-memory cache (text -> embedding vector)
        # Using dict instead of lru_cache because lists aren't hashable
        self._cache: dict[str, List[float]] = {}
        self._cache_max_size = 2000

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single search query.

        Args:
            text: Input text to embed

        Returns:
            768-dimensional float vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        text = self._query_prefix + text.strip()

        # Check cache
        if text in self._cache:
            return self._cache[text]

        vector = self._compute_embeddings([text])[0]

        # Cache result
        self._cache_put(text, vector)

        return vector

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (graph nodes, edges, episodes).

        Args:
            texts: List of input texts
            batch_size: Number of texts per model batch

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            if not text:
                # Empty text — zero vector
                results[i] = [0.0] * EMBEDDING_DIM
                continue
            text = self._document_prefix + text
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch-embed uncached texts
        if uncached_texts:
            all_vectors = self._compute_embeddings(uncached_texts, batch_size)

            # Place results and cache
            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache_put(text, vec)

        return results  # type: ignore

    def _compute_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Run the local model over the given (already prefixed) texts."""
        try:
            model = _get_model(self.model, self.cache_dir)
            vectors = [v.tolist() for v in model.embed(texts, batch_size=batch_size)]
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}") from e

        if len(vectors) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        if vectors and len(vectors[0]) != EMBEDDING_DIM:
            raise EmbeddingError(
                f"Model {self.model} returned {len(vectors[0])}-dim vectors, "
                f"but the Neo4j index expects {EMBEDDING_DIM}"
            )
        return vectors

    def _cache_put(self, text: str, vector: List[float]) -> None:
        """Add to cache, evicting oldest entries if full."""
        if len(self._cache) >= self._cache_max_size:
            # Remove ~10% of oldest entries
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        self._cache[text] = vector

    def health_check(self) -> bool:
        """Check that the local embedding model loads and works."""
        try:
            vec = self.embed("health check")
            return len(vec) > 0
        except Exception:
            return False


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass
