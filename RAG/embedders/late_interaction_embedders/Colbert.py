from pathlib import Path

from fastembed import LateInteractionTextEmbedding
from tqdm import tqdm

from .base import LateInteractionEmbedder
from ...CacheManager import CACHE_DIR, CacheManager

_MODEL_ID: str = "colbert-ir/colbertv2.0"
"""The model used to generate late interaction embeddings for RAG"""

_VECTOR_SIZE: int = 128
"""Embedding dimension for mean-pooled ColBERT dense vectors."""


class Colbert(LateInteractionEmbedder):
    def __init__(self):
        self.model = LateInteractionTextEmbedding(_MODEL_ID, cache_dir=CACHE_DIR)

    @property
    def model_name(self) -> str:
        """Name of the sparse model."""
        model_name = _MODEL_ID.split("/")[-1]
        return model_name

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    def embed_query(self, query: str) -> list[float]:
        """
        Generates embeddings for the provided query using.

        :param query: The input string for which the embeddings will be
            generated.
        :type query: str

        :return: A list of floating-point numbers representing the embedding
            vector for the input query.
        :rtype: list[float]
        """
        embeddings = next(self.model.query_embed(query=query))
        return embeddings.tolist()

    def embed_documents(self, documents: list[str], source_file: Path = None) -> list:
        """
        Embeds a list of textual documents using the ColBERT model for late interaction and
        saves the resulting embeddings in a cache. If a source file is provided, it is used
        to cache the embeddings under a specific key.

        :param documents: A list of strings, each representing a textual document to be
            embedded by the ColBERT model.
        :param source_file: An optional Path object referencing the file from which the
            documents were obtained. Used for caching purposes.
        :return: A list of embeddings, where each embedding is a list of floats
            corresponding to the encoded representation of a document.
        """
        cache_manager = CacheManager()
        late_embeddings = [
            doc.tolist()
            for doc in tqdm(
                self.model.embed(documents),
                total=len(documents),
                desc="Embedding ColBERT (late interaction)",
                unit="doc",
            )
        ]

        if source_file:
            cache_manager.save_embeddings(
                embeddings=late_embeddings,
                source_file=source_file,
                model=self.model_name,
                vector_size=self.vector_size,
            )

        return late_embeddings
