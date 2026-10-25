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

import uuid
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from tqdm import tqdm

from back_end.rag.config import QdrantConfig
from back_end.rag.embedders import GeneralEmbedder
from back_end.rag.models import GlossaryMatch, GlossaryEntry
from .base import DatabaseAdapter

# TODO: load e load_from_cache si possono unire

MAX_RETRIES = 3

BATCH_SIZE: int = 100
"""The size of a batch for inserting new records into the database"""


class QdrantAdapter(DatabaseAdapter):
    """
    DatabaseAdapter implementation for Qdrant.
    Supports hybrid search with dense, sparse, and late-interaction embeddings.
    """

    def __init__(
        self, host: str, port: int, hybrid_search: bool, embedder: GeneralEmbedder
    ) -> None:
        """
        Initialize the Qdrant client.

        :param host: Host of the Qdrant server.
        :param port: Port of the Qdrant server.
        :param hybrid_search: Whether to use hybrid search.
        :param embedder: The embedder to use for generating vector embeddings associated to the entries.
        """
        self._client = QdrantClient(host=host, port=port)
        self._embedder = embedder
        self._hybrid_search = hybrid_search

    @classmethod
    def from_config(
        cls, config: QdrantConfig, embedder: GeneralEmbedder
    ) -> "QdrantAdapter":
        """
        Create a QdrantAdapter instance from configuration.

        :param embedder:
        :param config: Configuration dictionary containing the host and port.
        :return: QdrantAdapter instance.
        """
        return cls(
            host=config.host,
            port=config.port,
            hybrid_search=config.hybrid_search,
            embedder=embedder,
        )

    def create_collection(self, name: str) -> None:
        """
        If it does not exist already, create a Qdrant collection with dense, sparse, and late-interaction vectors.

        :param name: Name of the collection to create.
        """
        if self._client.collection_exists(collection_name=name):
            return

        colbert_config = models.VectorParams(
            size=self._embedder.late_vector_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        )
        sparse_config = models.SparseVectorParams(modifier=models.Modifier.IDF)

        vectors_config = {
            self._embedder.dense_vector_name: models.VectorParams(
                size=self._embedder.dense_vector_size,
                distance=models.Distance.COSINE,
            ),
            self._embedder.late_vector_name: colbert_config,
        }

        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config={self._embedder.sparse_vector_name: sparse_config},
        )

    def _generate_points(
        self,
        entries: list[str],
        dense_vecs: list[list[float]],
        sparse_vecs: list[list[float]],
        late_vecs: list[list[float]],
    ) -> list[PointStruct]:
        return [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    self._embedder.sparse_vector_name: sparse_vecs[i],
                    self._embedder.late_vector_name: late_vecs[i],
                    self._embedder.dense_vector_name: dense_vecs[i],
                },
                payload={
                    "term": (text.split(":", 1)[0].strip() if text else ""),
                    "definition": (
                        text.split(":", 1)[1].strip() if ":" in text else ""
                    ),
                },
            )
            for i, text in enumerate(entries)
        ]

    def write_data(self, collection_name: str, entries: list[str], path: Optional[Path]) -> int:
        """
        Load text entries into the Qdrant collection.

        :param collection_name: Name of the collection.
        :param entries: List of text entries in format "term: definition".
        :param path: Path to the original source file used for embedding, needed for caching.
        :return: Number of entries loaded.
        """

        with tqdm(
            total=len(entries), desc=f"Embedding {collection_name}", unit="entry"
        ) as pbar:
            for i in range(0, len(entries), BATCH_SIZE):
                batch = entries[i : i + BATCH_SIZE]
                batch_dense_vecs = self._embedder.embed_dense(
                    contents=batch, source_file=path
                )
                batch_sparse_vecs = self._embedder.embed_sparse(batch)
                batch_late_vecs = self._embedder.embed_late_interaction(batch)

                points = self._generate_points(
                    entries=batch,
                    dense_vecs=batch_dense_vecs,
                    sparse_vecs=batch_sparse_vecs,
                    late_vecs=batch_late_vecs,
                )
                self._client.upload_points(
                    collection_name=collection_name,
                    points=points,
                    batch_size=BATCH_SIZE,
                    max_retries=MAX_RETRIES,
                )
                pbar.update(len(batch))

        return len(entries)

    def write_dict(self, collection_name: str, entries: list[dict[str, str]]) -> int:
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(collection_name=collection_name, vectors_config={})
        with tqdm(
                total=len(entries), desc=f"Embedding {collection_name}", unit="entry"
        ) as pbar:
            for entry in entries:
                # Insert the document (pass an empty dict for the vector)
                self._client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector={},  # No vector data provided
                            payload=entry
                        )
                    ]
                )
                pbar.update(1)

        return len(entries)

    def _to_python(self, value: Any) -> Any:
        """
        Recursively convert numpy-like containers to plain Python types.
        Needed for pydantic validation in qdrant-client models.

        :param value: The value to convert.
        :return: The converted value.
        """
        if hasattr(value, "tolist"):  # numpy arrays/scalars
            return self._to_python(value.tolist())
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_python(v) for v in value]
        return value

    def search(
        self,
        collection_name: str,
        query_embeddings: dict[str, Any],
        top_k: int,
        k: int,
        query_text: str = "",
    ) -> list[GlossaryMatch]:
        if self._hybrid_search:
            return self._hybrid_search_query(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                top_k=top_k,
                k=k,
            )
        else:
            return self._dense_search(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                top_k=top_k,
                k=k,
            )

    def _dense_search(
        self,
        collection_name: str,
        query_embeddings: dict[str, Any],
        top_k: int,
        k: int,
    ) -> list[GlossaryMatch]:
        """
        Search the collection using only dense vector embeddings.

        :param collection_name: Name of the collection to search.
        :param query_embeddings: Dictionary containing the dense query vector.
        :param top_k: Maximum number of final matches to return.
        :param k: Limit of points to retrieve from the database.
        :return: List of the top matching GlossaryMatch objects.
        """
        dense_emb = query_embeddings.get(self._embedder.dense_vector_name)

        results = self._client.query_points(
            collection_name=collection_name,
            query=dense_emb,
            using=self._embedder.dense_vector_name,
            with_payload=True,
            limit=k,
        )

        return [
            GlossaryMatch(
                term=hit.payload["term"],
                definition=hit.payload["definition"],
                score=hit.score,
            )
            for hit in results.points
            if hit.payload is not None
        ][:top_k]

    def _hybrid_search_query(
        self,
        collection_name: str,
        query_embeddings: dict[str, Any],
        top_k: int,
        k: int,
    ) -> list[GlossaryMatch]:
        """
        Perform a hybrid search using multi-vector prefetching and late-interaction scoring.

        :param collection_name: Name of the collection to search.
        :param query_embeddings: Dictionary containing dense, sparse, and late-interaction vectors.
        :param top_k: Maximum number of final matches to return.
        :param k: Limit of points to evaluate during the late-interaction phase.
        :return: List of the top matching GlossaryMatch objects.
        """
        pre_fetch_limit = max(100, top_k * 10)
        prefetch = [
            models.Prefetch(
                query=emb,
                using=name,
                limit=pre_fetch_limit,
            )
            for name, emb in query_embeddings.items()
        ]

        results = self._client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=query_embeddings[self._embedder.late_vector_name],
            using=self._embedder.late_vector_name,
            with_payload=True,
            limit=k,
        )

        return [
            GlossaryMatch(
                term=hit.payload["term"],
                definition=hit.payload["definition"],
                score=hit.score,
            )
            for hit in results.points
            if hit.payload is not None
        ][:top_k]

    def delete_collection(self, name: str) -> None:
        """
        Delete a Qdrant collection.

        :param name: Name of the collection to delete.
        """
        if self._client.collection_exists(collection_name=name):
            self._client.delete_collection(collection_name=name)

    def collection_exists(self, name: str) -> bool:
        """
        Check if a Qdrant collection exists.

        :param name: Name of the collection to check.
        :return: True if collection exists, False otherwise.
        """
        return self._client.collection_exists(collection_name=name)

    def get_all_dense_embeddings(self, collection_name: str) -> list[list[float]]:
        """
        Retrieve all dense embeddings from the specified collection.

        :param collection_name: Name of the collection to retrieve embeddings from.
        :return: List of dense embeddings, where each embedding is a list of floats.
        """
        if not self._client.collection_exists(collection_name=collection_name):
            return []

        results, _ = self._client.scroll(
            collection_name=collection_name,
            with_vectors=True,
            limit=10000,
        )

        dense_embeddings = []
        for hit in results:
            vector = hit.vector
            if not isinstance(vector, dict):
                continue

            dense_vector = vector.get(self._embedder.dense_vector_name)
            if dense_vector is not None:
                dense_embeddings.append(dense_vector)

        return dense_embeddings

    from typing import List
    from back_end.rag.models import GlossaryEntry

    def get_all_entries(self, collection_name: str) -> List[GlossaryEntry]:
        """
        Retrieve all entries from the Qdrant collection.

        :param collection_name: Name of the collection to retrieve entries from.
        :return: List of GlossaryEntry objects.
        """
        results, _ = self._client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=10000,
        )

        entries = []
        for hit in results:
            try:
                if hit.payload is None:
                    continue

                term = hit.payload.get("term", "")
                definition = hit.payload.get("definition", "")
                entries.append(GlossaryEntry(term=term, definition=definition))
            except Exception as e:
                print(f"Error processing hit: {e}")
                continue

        return entries

    def has_entries(self, collection_name: str) -> int:
        """
        Check if collection has entries.

        :param collection_name: Name of the collection.
        :return: 0 if empty, the count if has data.
        """
        if not self._client.collection_exists(collection_name=collection_name):
            return 0

        result = self._client.count(
            collection_name=collection_name,
            exact=True,
        )

        return result.count

    def get_stored_embeddings(self, collection_name: str):
        results, _ = self._client.scroll(
            collection_name=collection_name,
            with_vectors=True,
            limit=10000,
        )
        return [hit.vector.get(self._embedder.dense_vector_name) for hit in results]

    def load_data_from_cache(
        self,
        collection_name: str,
        entries: list[str],
        dense_embeddings: list[list[float]],
        late_embeddings: Optional[list[list[float]]] = None,
    ) -> int:

        if not late_embeddings:
            raise ValueError(
                "Late embeddings are required for loading from cache in Qdrant."
            )

        with tqdm(
            total=len(entries),
            desc=f"Loading data from cache for {collection_name}",
            unit="entry",
        ) as pbar:
            for i in range(0, len(entries), BATCH_SIZE):
                batch = entries[i : i + BATCH_SIZE]
                batch_dense_vecs = dense_embeddings[i : i + BATCH_SIZE]
                batch_sparse_vecs = self._embedder.embed_sparse(batch)
                batch_late_vecs = late_embeddings[i : i + BATCH_SIZE]

                points = self._generate_points(
                    entries=batch,
                    dense_vecs=batch_dense_vecs,
                    sparse_vecs=batch_sparse_vecs,
                    late_vecs=batch_late_vecs,
                )

                self._client.upload_points(
                    collection_name=collection_name,
                    points=points,
                    batch_size=BATCH_SIZE,
                    max_retries=MAX_RETRIES,
                    wait=True,
                )
                pbar.update(len(batch))
        return len(entries)
