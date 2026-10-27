
from .config import QdrantConfig
from .RagManager import RagManager
from .adapters.base import DatabaseAdapter
from .adapters.QdrantAdapter import QdrantAdapter
from .embedders.GeneralEmbedder import GeneralEmbedder
from .embedders.dense_embedders.GeminiEmbedder import GeminiEmbedder

# Global singletons
global_embedder = None
global_db_adapter = None
global_rag_manager = None


def _initialize_embedder() -> GeneralEmbedder:
    embedder = GeneralEmbedder.get_instance(
        dense_embedder=GeminiEmbedder.get_instance()
    )
    return embedder


def _initialize_adapter(_embedder: GeneralEmbedder) -> DatabaseAdapter:
    adapter = QdrantAdapter.from_config(config=QdrantConfig(), embedder=_embedder)
    return adapter


def _initialize_rag_manager(
    _embedder: GeneralEmbedder, _adapter: DatabaseAdapter
) -> RagManager:
    retriever = RagManager(adapter=_adapter, embedder=_embedder)
    return retriever


def get_embedder() -> GeneralEmbedder:
    global global_embedder
    if global_embedder is None:
        global_embedder = _initialize_embedder()
    return global_embedder


def get_db_adapter() -> DatabaseAdapter:
    global global_db_adapter
    if global_db_adapter is None:
        global_db_adapter = _initialize_adapter(get_embedder())
    return global_db_adapter


def get_rag_manager() -> RagManager:
    global global_rag_manager
    if global_rag_manager is None:
        global_rag_manager = _initialize_rag_manager(get_embedder(), get_db_adapter())
    return global_rag_manager
