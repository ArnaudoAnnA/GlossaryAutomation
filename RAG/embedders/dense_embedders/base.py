from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class DenseEmbedder(ABC):
    """
    Abstract base class for dense embedding strategies.
    All dense embedding implementations must inherit from this class and implement the required methods.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the dense vector field used in the database."""
        pass

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of the dense vectors produced by this embedder."""
        pass

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
        source_file: Optional[Path] = None,
    ) -> list[list[float]]:
        """Embeds a list of documents and returns their dense vector representations."""
        pass

    @abstractmethod
    def embed_query(self, term_definition: str) -> list[float]:
        """Embeds a single query and returns its dense vector representation."""
        pass
