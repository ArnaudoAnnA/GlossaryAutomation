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
from pathlib import Path
from typing import Optional

from dataset.index import load_requirements, DatasetProject, get_project_path
from .CacheManager import CacheManager
from .adapters.base import DatabaseAdapter
from .config import RAGConfig
from .embedders.GeneralEmbedder import GeneralEmbedder
from .models import RequirementEntry

BASE_DIR = Path(__file__).resolve().parent.parent

PURE_GLOSSARY_PATH = BASE_DIR / "data" / "nasa_glossary.csv"
"""Path of the CSV file containing glossary entries."""

NIST_GLOSSARY_PATH = BASE_DIR / "data" / "nist_glossary.json"
"""Path of the JSON file containing glossary entries."""


class RagManager:
    """
    Manages a glossary collection using a database adapter.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        adapter: DatabaseAdapter,
        embedder: GeneralEmbedder,
        top_k: Optional[int] = None,
    ):
        """
        :param adapter: Database adapter for storage operations.
        :param top_k: Maximum number of glossary entries to return per query.
        :param embedder: class containing models for embedding generation.
        """
        default_config = RAGConfig()

        self._adapter = adapter
        self._top_k = top_k if top_k is not None else default_config.top_k
        self._ranking_size = default_config.ranking_size
        self._embedder = embedder

    def clear_collection(self, collection_name: str) -> None:
        self._adapter.delete_collection(name=collection_name)

    def load_data(
        self,
        collection_name: str,
        projects: list[DatasetProject],
    ) -> None:
        """
        Loads data from the indicated paths.

        :param projects: List of dataset projects to load.
        :param collection_name: Name of the glossary collection to check and populate.
        """

        with self._lock:
            if not self._adapter.collection_exists(name=collection_name):
                self._adapter.create_collection(name=collection_name)

            cache_manager = CacheManager()

            for project in projects:

                entries = load_requirements(project)

                cached_dense_embeddings = cache_manager.get_cached_embeddings(
                    path=get_project_path(project),
                    model_name=self._embedder.dense_vector_name,
                    model_size=self._embedder.dense_vector_size,
                    expected_count=len(entries),
                )

                if not cached_dense_embeddings:
                    cached_dense_embeddings = self._embedder.embed_dense(
                        contents=[e.requirement_text for e in entries], source_file=get_project_path(project)
                    )

                _ = self._adapter.load_data_from_cache(
                    collection_name=collection_name,
                    entries=entries,
                    dense_embeddings=cached_dense_embeddings,
                )

    def retrieve_similar_texts(
            self,
            collection_name: str,
            text: str,
            top_k: int
    ) -> list[RequirementEntry]:
        """
        Retrieves a list of matches for a given text using a dense search strategy.

        The comparison is done between the embeddings of the provided text and the text contained in the dataset.

        :param collection_name: The name of the collection to search.
        :type collection_name: str
        :param text: The input requirement text to search for.
        :type text: str
        :param top_k: The maximum number of matches to return.
        :type top_k: int
        :return: A list of matching RequirementEntry objects sorted by relevance score.
        :rtype: list[RequirementEntry]
        :raises RuntimeError: If embeddings cannot be computed.
        """
        dense_embeddings = self._embedder.embed_dense(contents=[text])

        if not dense_embeddings or not dense_embeddings[0]:
            raise RuntimeError(f"Impossible to compute embeddings for text \"{text}\"")

        query_dict = {
            self._embedder.dense_vector_name: dense_embeddings[0]
        }

        hits = self._adapter.search_similar_requirements(
            collection_name=collection_name,
            query_embeddings=query_dict,
            top_k=top_k,
            k=getattr(self, "_ranking_size", top_k * 2),
            query_text=text,
        )

        hits = sorted(hits, key=lambda x: getattr(x, "score", 0.0), reverse=True)
        return hits[:top_k]
