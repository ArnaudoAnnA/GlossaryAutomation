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

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from back_end.constants import CACHE_DIR_EMBEDDINGS


class CacheManager:
    """
    Manages caching of embeddings to optimize performance by reusing previously computed results.

    This class handles the saving, loading, and validation of embeddings and metadata.
    It is designed to store embeddings as numpy arrays in `.npy` files alongside their
    metadata in `.json` files within a pre-defined cache directory.

    :ivar CACHE_DIR_EMBEDDINGS: Directory used for storing cached embeddings and metadata.
    :type CACHE_DIR_EMBEDDINGS: Path
    """

    def _ensure_cache_dir(self) -> None:
        CACHE_DIR_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    def _compute_file_hash(self, path: Path) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_cache_path(self, dataset: str, model: str) -> Path:
        self._ensure_cache_dir()
        path = CACHE_DIR_EMBEDDINGS / f"{dataset}_{model.replace('-', '_')}"
        return path

    def _load_cached_metadata(self, cache_path: Path) -> Optional[dict]:
        json_path = cache_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        return None

    def _load_cached_embeddings(self, cache_path: Path):
        npy_path = cache_path.with_suffix(".npy")
        if not npy_path.exists():
            return None
        try:
            embeddings = np.load(npy_path, allow_pickle=True)
            result = [row.astype(np.float32).tolist() for row in embeddings]
            return result
        except Exception as e:
            print(f"[cache load] error: {e}")
            return None

    def save_embeddings(
        self,
        embeddings: list[list[float]],
        source_file: Path,
        model: str,
        vector_size: int,
        normalized: bool = False,
    ) -> None:
        """
        Save embeddings and metadata to cache.

        :param embeddings: List of embedding vectors to save.
        :param source_file: Path to the original source file used for embedding.
        :param model: The embedding model used.
        :param vector_size: The size of the embedding vectors.
        :param normalized: Whether the embeddings are normalized or not.
        """
        cache_key = ""
        if normalized:
            cache_key = "normalized_"
        cache_key += source_file.name.split("_")[0]
        cache_path = self._get_cache_path(cache_key, model)
        npy_path = cache_path.with_suffix(".npy")
        json_path = cache_path.with_suffix(".json")

        source_hash = self._compute_file_hash(path=source_file)

        # This try-except is necessary for late-interaction embeddings:
        # since each document has a different length, depending on the amount of
        # tokens, simply calling np.array on the list of embeddings will raise a
        # ValueError, since the resulting array will be non-rectangular. To solve this
        # issue, we can save the embeddings as an array of objects, where each object
        # is a numpy array of floats.
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
        except ValueError:
            embeddings_array = np.array(
                [np.array(e, dtype=np.float32) for e in embeddings], dtype=object
            )
        np.save(npy_path, embeddings_array)

        metadata = {
            "source_file": str(source_file),
            "source_hash": source_hash,
            "model": model,
            "vector_size": vector_size,
            "count": len(embeddings),
            "created_at": datetime.now().isoformat(),
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _validate_cache(
        self,
        cache_path: Path,
        source_file: Path,
        expected_count: int,
        model: str,
        vector_size: int,
    ) -> bool:
        """
        Validate the cached embeddings against the expected parameters.

        :param cache_path: Path to the cache file (without extension).
        :param source_file: Path to the original source file used for embedding.
        :param expected_count: Expected number of embeddings.
        :param model: The embedding model used.
        :param vector_size: The expected size of the embedding vectors.
        :return: True if the cache is valid, False otherwise.
        """
        metadata = self._load_cached_metadata(cache_path=cache_path)
        if metadata is None:
            return False

        if Path(metadata.get("source_file", "")).name != source_file.name:
            return False

        current_hash = self._compute_file_hash(source_file)
        if metadata.get("source_hash") != current_hash:
            return False

        if metadata.get("model") != model or metadata.get("vector_size") != vector_size:
            return False

        if metadata.get("count") != expected_count:
            return False

        return True

    def get_cached_embeddings(
        self, path: Path, model_name: str, model_size: int, expected_count: int
    ) -> Optional[list[list[float]]]:
        """
        Retrieve cached embeddings if available and valid. This function checks for
        pre-computed embeddings stored in a cache and validates their correctness
        based on provided parameters. If the cache is valid, returns the embeddings.

        :param path: Path to the source file used to generate embeddings.
        :type path: str
        :param model_name: Name of the model used for embedding generation.
        :type model_name: str
        :param model_size: Size of the embeddings generated by the model.
        :type model_size: int
        :param expected_count: Expected number of embeddings in the cache.
        :type expected_count: int
        :return: A 2D list of floats representing the cached embeddings if valid,
            or None if no valid cache is available.
        :rtype: Optional[list[list[float]]]
        """
        file_name = path.name
        cache_key = file_name.split("_")[0]
        cache_path = self._get_cache_path(cache_key, model_name)

        is_cache_valid = self._validate_cache(
            cache_path=cache_path,
            expected_count=expected_count,
            model=model_name,
            vector_size=model_size,
            source_file=path,
        )

        print(
            f"Cache validation for {cache_path}: {'valid' if is_cache_valid else 'invalid'}"
        )

        if is_cache_valid:
            embeddings = self._load_cached_embeddings(cache_path)
            return embeddings
        else:
            return None
