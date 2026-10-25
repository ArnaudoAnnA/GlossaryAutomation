from fastembed import SparseTextEmbedding

from back_end.constants import CACHE_DIR
from .base import SparseEmbedder

_MODEL_ID: str = "prithivida/Splade_PP_en_v1"
"""The model ID used for serving Splade."""


class Splade(SparseEmbedder):
    def __init__(self):
        self._model = SparseTextEmbedding(_MODEL_ID, cache_dir=CACHE_DIR)

    @property
    def model_name(self) -> str:
        model_name = _MODEL_ID.split("/")[-1]
        return model_name

    def embed_query(self, query: str) -> dict[str, list[float]]:
        """
        Generates embeddings for the provided query using.

        :param query: The input string for which the embeddings will be
            generated.
        :type query: str

        :return: A list of floating-point numbers representing the embedding
            vector for the input query.
        :rtype: list[float]
        """
        embeddings = next(self._model.query_embed(query=query))
        return {
            "indices": embeddings.indices.tolist(),
            "values": embeddings.values.tolist(),
        }

    def embed_documents(self, documents: list[str]) -> list[dict[str, list]]:
        embeddings = self._model.query_embed(documents)
        return [
            {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
            for emb in embeddings
        ]
