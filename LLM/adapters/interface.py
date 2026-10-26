from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Type
from pydantic import BaseModel


class LLMAdapterInterface(ABC):
    """Abstract interface for LLM adapters."""

    @abstractmethod
    def __init__(
        self,
        model_id: str,
        sys_prompt: str = "",
        shot_examples: list[dict[str, str]] | None = None,
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def format_history(raw_history: list[tuple[str, str]]) -> list[Any]:
        """
        :param raw_history: Each tuple in the list contains an input-output pair describing an interaction between the user and the llm."""
        pass

    @abstractmethod
    def make_a_query(
        self,
        prompt: str,
        response_schema: Type[BaseModel] | None = None,
        max_new_tokens: int = 200,
    ) -> str | Any:
        pass


class LLMProvider(Enum):
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"


def get_llm_adapter(
    model_id: str,
    model_provider: LLMProvider,
    sys_prompt: str,
    shot_examples: list[Any] | None,
) -> LLMAdapterInterface:
    if not isinstance(model_provider, LLMProvider):
        raise ValueError(f"Unknown provider: {model_provider}")

    # Deferred imports to prevent circular dependency
    if model_provider == LLMProvider.GOOGLE:
        from LLM.adapters.Gemini.GeminiAdapter import GeminiAdapter
        return GeminiAdapter(model_id=model_id, sys_prompt=sys_prompt, shot_examples=shot_examples)

    if model_provider == LLMProvider.HUGGINGFACE:
        from LLM.adapters.HuggingFaceAdapter import HuggingFaceAdapter
        return HuggingFaceAdapter(model_id=model_id, sys_prompt=sys_prompt, shot_examples=shot_examples)

    raise ValueError(f"Provider not implemented: {model_provider}")