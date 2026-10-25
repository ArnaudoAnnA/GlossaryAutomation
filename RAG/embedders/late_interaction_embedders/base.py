from abc import ABC, abstractmethod


class LateInteractionEmbedder(ABC):
    """
    Abstract base class for sparse embedding strategies.
    All sparse embedding implementations must inherit from this class and implement the required methods.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the sparse embedder model."""
        pass

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of the late interaction vectors produced by this embedder."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a given query into a vector representation.

        :param query: A string representing the query to be embedded.
        :return: A list of floating-point numbers representing the embedded vector
            of the query.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents into their vector representations.

        :param documents: A list of strings, where each string is a document to be embedded.
        :return: A list of lists, where each inner list contains floating-point numbers representing the embedded vector of a document.
        """
        pass
