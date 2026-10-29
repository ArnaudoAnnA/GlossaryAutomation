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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from RAG.embedders.GeneralEmbedder import GeneralEmbedder
from RAG.models import RequirementEntry


class DatabaseAdapter(ABC):
    """
    Abstract base class for database adapters.
    All database implementations (Qdrant, PostgreSQL, etc.) must implement this interface.
    """

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: dict[str, Any], embedder: GeneralEmbedder
    ) -> "DatabaseAdapter":
        """
        Create a DatabaseAdapter instance from configuration.

        :param embedder: GeneralEmbedder instance.
        :param config: Configuration dictionary specific to the database type.
        :return: DatabaseAdapter instance.
        """
        pass

    @abstractmethod
    def create_collection(self, name: str) -> None:
        """
        Create a collection/table for storing glossary entries.

        :param name: Name of the collection to create.
        """
        pass

    @abstractmethod
    def write_data(self, collection_name: str, entries: list[str], path: Optional[Path]) -> int:
        """
        Writes text entries into the collection.

        :param collection_name: Name of the collection.
        :param entries: List of text entries in format "term: definition".
        :param path: Path to the original source file used for embedding, needed for caching.
        :return: Number of entries loaded.
        """
        pass

    @abstractmethod
    def write_dict(self, collection_name: str, entries: list[dict[str, str]]) -> int:
        """
        Writes entries into the collection without associating them with an embedding vector.

        :param collection_name: Name of the collection.
        :param entries: List of entries in format {"message_index": str, "message": str, "score": str, "details": str}.
        :return: Number of entries loaded.
        """
        pass

    @abstractmethod
    def search_similar_requirements(
        self,
        collection_name: str,
        query_embeddings: dict[str, Any],
        top_k: int,
        k: int,
        field_name: str = "requirement_text",
        query_text: str = "",
    ) -> list[RequirementEntry]:
        """
        Search for similar entries using query embeddings.

        :param collection_name: Name of the collection to search.
        :param query_embeddings: Dictionary containing the dense, sparse, and late-interaction embeddings.
        It stores them in the following format: {'dense_model_id': ..., 'sparse_model_id': ..., 'late_model_id': ...}.
        :param top_k: Number of results to return.
        :param k: Number of candidates to use during search.
        :param field_name: Name of the field of the collection to be compared with the query.
        :param query_text: Optional query text for full-text search (ts_rank).
        :return: List of RequirementEntry objects.
        """
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection/table.

        :param name: Name of the collection to delete.
        """
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """
        Check if a collection/table exists.

        :param name: Name of the collection to check.
        :return: True if collection exists, False otherwise.
        """
        pass

    @abstractmethod
    def get_all_dense_embeddings(self, collection_name: str) -> list[list[float]]:
        """
        Retrieve all dense embeddings from the specified collection.

        :param collection_name: Name of the collection to retrieve embeddings from.
        :return: List of dense embeddings, where each embedding is a list of floats.
        """
        pass

    @abstractmethod
    def get_all_entries(self, collection_name: str) -> list[Any]:
        """
        Retrieve all entries from the collection.

        :param collection_name: Name of the collection to retrieve entries from.
        :return: List of Any objects.
        """
        pass

    @abstractmethod
    def has_entries(self, collection_name: str) -> int:
        """
        Check if collection has entries.

        :param collection_name: Name of the collection.
        :return: 0 if empty, the count if it has data.
        """
        pass

    @abstractmethod
    def get_stored_embeddings(
        self, collection_name: str
    ) -> Optional[list[list[float]]]:
        """
        Retrieve stored embeddings for a collection, if available.

        :param collection_name: Name of the collection.
        :return: List of embeddings or None if not supported.
        """
        pass

    @abstractmethod
    def load_data_from_cache(
        self,
        collection_name: str,
        entries: list[RequirementEntry],
        dense_embeddings: list[list[float]],
        late_embeddings: list[list[float]] = None,
    ) -> int:
        """
        Load text entries using pre-computed embeddings (from cache).

        :param collection_name: Name of the collection.
        :param entries: List of text entries in format "term: definition".
        :param dense_embeddings: Pre-computed dense embeddings.
        :param late_embeddings: Pre-computed late-interaction embeddings.
        :return: Number of entries loaded.
        """
        pass
