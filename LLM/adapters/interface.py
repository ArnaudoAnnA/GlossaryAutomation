from abc import ABC, abstractmethod
from typing import Any, Type

from google.genai import Client
from google.genai.chats import Chat
from google.genai.types import Content
from pydantic import BaseModel

class LLMAdapterInterface(ABC):
    """Abstract interface for ModelWrapper."""

    @abstractmethod
    def __init__(
        self,
        sys_prompt: str,
        model_id: str = "gemini-3.5-flash-lite",
        shot_examples: list[Content] | None = None,
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def init_chat(
        sys_prompt: str,
        history: list[Content] | None,
        model_id: str = "gemini-3.5-flash-lite"
    ) -> tuple[Chat, Client]:
        pass

    @staticmethod
    @abstractmethod
    def format_history(raw_dictionary: list[dict[str, str]]) -> list[Content]:
        pass

    @abstractmethod
    def make_a_query(
        self,
        prompt: str,
        response_schema: Type[BaseModel] | None = None,
        max_new_tokens: int = 200,
    ) -> str | Any:
        pass