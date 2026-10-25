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

from back_end.BackEndBuffer import BackEndBuffer
from back_end.utilities.CacheManager import CacheManager
from .InputParser import InputParser
from .adapters.base import DatabaseAdapter
from .config import RAGConfig
from .embedders import GeneralEmbedder
from .models import GlossaryMatch

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

    def ensure_data(
        self,
        collection_name: str,
        paths: Optional[list[Path]] = None,
    ) -> None:
        """
        Ensures that the glossary collection is populated with data. If the collection is empty,
        it loads data from predefined sources.

        :param paths: List of paths to CSV or JSON files containing glossary entries.
        :param collection_name: Name of the glossary collection to check and populate.
        """

        if not paths:
            paths = [
                PURE_GLOSSARY_PATH
            ]  # We could also use NIST_GLOSSARY_PATH if we want to try some requirements regarding NIST

        with self._lock:
            if not self._adapter.collection_exists(name=collection_name):
                self._adapter.create_collection(name=collection_name)

            if not self._adapter.has_entries(collection_name):
                cache_manager = CacheManager()

                for path in paths:
                    file_format = Path(path).suffix.lower()
                    if file_format == ".csv":
                        loader_method = InputParser.parse_csv
                    elif file_format == ".json":
                        loader_method = InputParser.parse_json
                    else:
                        raise ValueError(f"Unsupported file format: {file_format}")

                    entries = loader_method(path)

                    cached_dense_embeddings = cache_manager.get_cached_embeddings(
                        path=path,
                        model_name=self._embedder.dense_vector_name,
                        model_size=self._embedder.dense_vector_size,
                        expected_count=len(entries),
                    )

                    cached_late_embeddings = cache_manager.get_cached_embeddings(
                        path=path,
                        model_name=self._embedder.late_vector_name,
                        model_size=self._embedder.late_vector_size,
                        expected_count=len(entries),
                    )

                    if not cached_dense_embeddings:
                        cached_dense_embeddings = self._embedder.embed_dense(
                            contents=entries, source_file=path
                        )

                    if not cached_late_embeddings:
                        cached_late_embeddings = self._embedder.embed_late_interaction(
                            texts=entries
                        )

                    _ = self._adapter.load_data_from_cache(
                        collection_name=collection_name,
                        entries=entries,
                        dense_embeddings=cached_dense_embeddings,
                        late_embeddings=cached_late_embeddings,
                    )

    def retrieve(
        self,
        collection_name: str,
        term_definition: str,
            user_interface_buffer: BackEndBuffer,
    ) -> list[GlossaryMatch]:
        """
        Retrieves a list of glossary matches for a given word using a hybrid search strategy.

        This method combines dense and sparse vector retrieval to prefetch candidate
        glossary entries, ensuring both semantic similarity and keyword relevance are
        considered. It then refines these candidates using a late-interaction model
        to calculate precise relevance scores.

        The process involves:
        1. Generating dense, sparse, and late-interaction embeddings for the input word.
        2. Prefetching candidates using the dense embedding model.
        3. Prefetching candidates using the sparse embedding model.
        4. Querying and rescoring the combined candidates using the late-interaction model.

        The comparison is done between 'term_definition' and the embeddings of 'term:definition' pairs.

        :param collection_name:
        :param term_definition: The input term and definition to search for in the glossary. Formatted as "term: definition".
        :type term_definition: str
        :param user_interface_buffer: User interface buffer to check for restart signals.
        :return: A list of `GlossaryMatch` objects, each containing a term, its associated
            definition, and a relevance score.
        :rtype: list[GlossaryMatch]
        """
        word_embeddings = self._embedder.embed_word(
            term_definition=term_definition,
            user_interface_buffer=user_interface_buffer,
        )

        if not word_embeddings:
            return []

        hits = self._adapter.search(
            collection_name=collection_name,
            query_embeddings=word_embeddings,
            top_k=self._top_k,
            k=self._ranking_size,
            query_text=term_definition,
        )

        hits = sorted(hits, key=lambda x: x.score, reverse=True)
        return hits[: self._ranking_size]
