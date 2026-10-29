#  Copyright (c) 2026. Politecnico di Torino
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import threading
from enum import Enum
from pathlib import Path
from typing import Optional

from google.genai import Client, types
from tqdm.asyncio import tqdm

from utilities_google_ai.APIKeysManager import NUM_API_KEYS, APIKeysManager
from utilities_google_ai.init_genai_client import init_genai_client_APIs
from constants import MAX_ATTEMPTS
from logging_debugging import log_thread_safe
from logging_debugging.logging_debugging import GLOBAL_LOG, LogType
from .base import DenseEmbedder
from ...CacheManager import CacheManager


class TaskType(Enum):
    """
    Enumeration of task types for embedding content with the Gemini API.
    Different task types can be used to optimize the embeddings for different use cases, such as embedding documents
    for retrieval or embedding queries for better search performance.
    """

    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


_GEMINI_EMBEDDER_MODEL_ID: str = "gemini-embedding-2-preview"
"""The model used to generate dense embeddings for RAG"""

_GEMINI_VECTOR_SIZE: int = 1_536
"""Embedding dimension"""

_EMBEDDING_BATCH_SIZE: int = 100
"""The number of documents to embed in a single batch"""


class GeminiEmbedder(DenseEmbedder):
    """
    Singleton class that implements the DenseEmbedder interface using the Gemini embedding API.
    """

    _instance = None
    """The singleton instance of the class."""
    _lock = threading.Lock()
    """A lock to ensure thread-safe access to the singleton instance."""
    _client_lock = threading.Lock()
    """A lock to ensure thread-safe access to the client."""

    def __init__(
        self, client: Client | None = None, model: str = _GEMINI_EMBEDDER_MODEL_ID
    ):
        self._client = client or init_genai_client_APIs()
        self._model = model
        self._storage_config = types.EmbedContentConfig(
            task_type=TaskType.RETRIEVAL_DOCUMENT,
            output_dimensionality=_GEMINI_VECTOR_SIZE,
        )
        self._query_config = types.EmbedContentConfig(
            task_type=TaskType.RETRIEVAL_QUERY,
            output_dimensionality=_GEMINI_VECTOR_SIZE,
        )

    @property
    def model_name(self) -> str:
        return _GEMINI_EMBEDDER_MODEL_ID

    @property
    def vector_size(self) -> int:
        return _GEMINI_VECTOR_SIZE

    @classmethod
    def get_instance(
        cls,
    ) -> "GeminiEmbedder":
        """
        This method implements the singleton pattern, ensuring that only one instance of
        GeminiDenseEmbedder exists throughout the application.
        It uses a double-checked locking mechanism to ensure thread safety when creating the instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _refresh_client(self) -> None:
        self._client = init_genai_client_APIs()

    def _load_from_cache(
        self, source_file: Path, expected_count: int
    ) -> Optional[list[list[float]]]:
        cache_manager = CacheManager()

        cached_embeddings = cache_manager.get_cached_embeddings(
            path=source_file,
            model_name=self._model,
            model_size=self.vector_size,
            expected_count=expected_count,
        )

        if cached_embeddings:
            return cached_embeddings
        return None

    def embed_documents(
        self,
        texts: list[str] | str,
        source_file: Optional[Path] = None,
    ) -> list[list[float]]:
        """
        Embed a list of documents (or a single document) using the Gemini embedding API, with built-in handling for caching.

        :param texts: A list of strings to embed.
        :param source_file: An optional identifier for the source of the texts, used for cache validation. Required if cache_key is provided.
        :return: A list of embeddings, where each embedding is a list of floats.
        """
        cache_manager = CacheManager()
        config = self._storage_config

        if isinstance(texts, str):
            texts = [texts]
            config = self._query_config

        contents = [
            types.Content(parts=[types.Part.from_text(text=text)]) for text in texts
        ]

        if source_file:
            cached_embeddings = self._load_from_cache(
                source_file=source_file, expected_count=len(texts)
            )
            if cached_embeddings:
                return cached_embeddings

        dense_embeddings = []

        with tqdm(total=len(contents), desc="Embedding Gemini", unit="doc") as pbar:
            for i in range(0, len(contents), _EMBEDDING_BATCH_SIZE):
                batch = contents[i : i + _EMBEDDING_BATCH_SIZE]
                for attempts in range(MAX_ATTEMPTS * NUM_API_KEYS):
                    try:
                        response = self._client.models.embed_content(
                            model=self._model,
                            contents=batch,
                            config=config,
                        )
                        batch_embeddings = [
                            embedding.values for embedding in response.embeddings
                        ]
                        dense_embeddings.extend(batch_embeddings)
                        pbar.update(len(batch))
                        break
                    except Exception as e:
                        log_thread_safe(GLOBAL_LOG, __file__, LogType.ERROR,
                            f"[ERROR] Error during embedding: {e}. Attempt {attempts + 1} of {MAX_ATTEMPTS}.\n---\n\n"
                        )
                        attempts += 1
                        if attempts % MAX_ATTEMPTS == 0:
                            with self._client_lock:
                                APIKeysManager.change_api_key()
                                self._refresh_client()
                        if attempts >= MAX_ATTEMPTS * NUM_API_KEYS:
                            raise RuntimeError(
                                f"Failed to embed batch after {MAX_ATTEMPTS * NUM_API_KEYS} attempts. Last error: {e}"
                            )

        if source_file:
            cache_manager.save_embeddings(
                embeddings=dense_embeddings,
                source_file=source_file,
                model=self._model,
                vector_size=self.vector_size,
            )
        return dense_embeddings

    def embed_query(self, term_definition: str) -> list[float]:
        result = self.embed_documents(texts=term_definition)
        return result[0]
