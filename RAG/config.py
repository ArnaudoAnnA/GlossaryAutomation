from dataclasses import dataclass

COLLECTION_NAME: str = "glossary_demo"
"""The name of the collection used for RAG."""


@dataclass(frozen=True)
class QdrantConfig:
    host: str = "qdrant-db"
    """The URL of the Qdrant server"""

    port: int = 6333
    """The port of the Qdrant server"""

    hybrid_search: bool = False
    """Whether to use hybrid search (dense + SPLADE + ColBERT rerank) in Qdrant adapter.
    When False, uses dense vector search only.
    When True, uses RRF fusion of dense + SPLADE with ColBERT reranking."""


@dataclass(frozen=True)
class PostgresConfig:
    """
    To use the PostgreSQL adapter, you need to set the following environment variables:
    - POSTGRES_USER
    - POSTGRES_PASSWORD
    - POSTGRES_DB
    The values of these variables should not change from the execution of the first 'docker compose up'.
    """

    user: str = "POSTGRES_USER"
    """The key to retrieve the username from the environment variables"""

    password: str = "POSTGRES_PASSWORD"
    """The key to retrieve the password from the environment variables"""

    host: str = "postgres-db"
    """The host of the PostgreSQL database"""

    port: int = 5432

    db: str = "POSTGRES_DB"
    """The key to retrieve the name of the PostgreSQL database"""

    hybrid_search: bool = False
    """Whether to use hybrid search (dense + bm25 + ColBERT rerank) in PostgreSQL adapter.
    When False, uses dense vector search only.
    When True, uses RRF fusion of dense + bm25 with ColBERT reranking."""

    m: int = 16
    """Number of neighbors per node in the HNSW index for pgvector. Lower M reduces recall but improves speed and memory usage."""

    ef_construction: int = 128
    """ Size of the dynamic list of candidates during index construction for pgvector. Higher EF_CONSTRUCTION 
    improves recall but increases indexing time and memory usage."""


@dataclass(frozen=True)
class ChromaConfig:
    host: str = "localhost"
    """The host of the Chroma server"""

    port: int = 8000
    """The port of the Chroma server"""


@dataclass
class RAGConfig:
    top_k: int = 5
    """The default number of results to return for each query"""

    ranking_size: int = 200
    """The number of embeddings to take in consideration during re-ranking"""
