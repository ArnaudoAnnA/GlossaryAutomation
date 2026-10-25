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
from typing import Optional, Union

from back_end.BackEndBuffer import BackEndBuffer
from .dense_embedders.base import DenseEmbedder
from .late_interaction_embedders.Colbert import Colbert
from .late_interaction_embedders.base import LateInteractionEmbedder
from .sparse_embedders.Splade import Splade
from .sparse_embedders.base import SparseEmbedder


class GeneralEmbedder:
    """
    Wrapper for hybrid embedding models (dense + sparse + late interaction).
    Uses a configurable dense embedder strategy.
    """

    _instance: Optional["GeneralEmbedder"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder = None,
        late_interaction_embedder: LateInteractionEmbedder = None,
    ):
        """
        :param dense_embedder: Dense embedder strategy.
        :param sparse_embedder: Identifier for the sparse embedding model.
        :param late_interaction_embedder: Identifier for the late interaction embedding model.
        """
        self._dense_embedder = dense_embedder
        self._sparse_embedder = sparse_embedder if sparse_embedder else Splade()
        self._late_interaction_embedder = (
            late_interaction_embedder if late_interaction_embedder else Colbert()
        )

    @property
    def dense_vector_name(self) -> str:
        return self._dense_embedder.model_name

    @property
    def sparse_vector_name(self) -> str:
        return self._sparse_embedder.model_name

    @property
    def late_vector_name(self) -> str:
        return self._late_interaction_embedder.model_name

    @property
    def dense_vector_size(self) -> int:
        return self._dense_embedder.vector_size

    @property
    def late_vector_size(self) -> int:
        return self._late_interaction_embedder.vector_size

    @staticmethod
    def get_instance(
        dense_embedder: DenseEmbedder,
    ) -> "GeneralEmbedder":

        if GeneralEmbedder._instance is None:
            with GeneralEmbedder._lock:
                if GeneralEmbedder._instance is None:
                    GeneralEmbedder._instance = GeneralEmbedder(
                        dense_embedder=dense_embedder,
                    )
        return GeneralEmbedder._instance

    @staticmethod
    def reset_instance() -> None:
        """
        Resets the singleton instance. Used for benchmark testing in order to
        reinitialize with different parameters.
        """
        GeneralEmbedder._instance = None

    def embed_dense(
        self,
        contents: Union[str, list[str]],
        source_file: Optional[Path] = None,
    ) -> Union[list[list[float]], list[float]]:
        """
        Embeds a list of documents or a single query using the configured dense embedding strategy.

        :param contents: A string or list of strings to be embedded.
        :param source_file: Optional source file information.
        :return: list of dense embedding vectors.
        """
        if isinstance(contents, str):
            return self._dense_embedder.embed_query(
                term_definition=contents,
            )
        else:
            return self._dense_embedder.embed_documents(
                texts=contents,
                source_file=source_file,
            )

    def embed_sparse(self, texts: list[str] | str) -> list:
        """
        Embeds a list of documents using the sparse embedding model.

        :param texts: A string or list of strings to be embedded.
        :return: list of sparse embedding vectors.
        """
        if isinstance(texts, str):
            return self._sparse_embedder.embed_query(query=texts)
        else:
            return self._sparse_embedder.embed_documents(documents=texts)

    def embed_late_interaction(self, texts: list[str] | str) -> list:
        """
        Embeds a list of documents using the late-interaction embedding model.

        :param texts: A string or list of strings to be embedded.
        :return: list of late-interaction embedding vectors.
        """
        if isinstance(texts, str):
            return self._late_interaction_embedder.embed_query(query=texts)
        else:
            return self._late_interaction_embedder.embed_documents(documents=texts)

    def embed_word(
        self,
        term_definition: str,
            user_interface_buffer: BackEndBuffer,
    ) -> dict:
        """
        This method takes a query string and computes its dense, sparse, and late interaction
        embeddings using their respective models. The embeddings are returned as a
        dictionary with keys corresponding to the model names.

        :param term_definition: Input query text (typically in "term: definition" format).
        :param user_interface_buffer: User interface buffer to check for restart signals.
        :return: A dictionary containing the embeddings.
        """
        if user_interface_buffer and user_interface_buffer.read_restart_signal():
            return {}

        dense_vec = self.embed_dense(
            contents=term_definition,
        )
        sparse_vec = self.embed_sparse(term_definition)
        late_vec = self.embed_late_interaction(term_definition)

        return {
            self._late_interaction_embedder.model_name: late_vec,
            self._sparse_embedder.model_name: sparse_vec,
            self._dense_embedder.model_name: dense_vec,
        }
