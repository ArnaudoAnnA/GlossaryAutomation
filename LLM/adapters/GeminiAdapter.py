#  Copyright (c) 2026. Anna Arnaudo / Politecnico di Torino
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
import time
from typing import Any, Type

from google.genai import errors, Client
from google.genai.chats import Chat
from google.genai.types import Content, GenerateContentConfig, Part
from pydantic import BaseModel

from LLM.adapters.interface import LLMAdapterInterface
from constants import MAX_ATTEMPTS, TEMPERATURE
from logging_debugging import log_thread_safe
from logging_debugging.logging_debugging import GLOBAL_LOG, LogType
from utilities_google_ai.APIKeysManager import APIKeysManager, NUM_API_KEYS
from utilities_google_ai.init_genai_client import init_genai_client_APIs

MODEL_ID = "gemini-3.5-flash-lite"


class GeminiAdapter(LLMAdapterInterface):

    def __init__(
            self,
            sys_prompt: str,
            model_id: str = MODEL_ID,
            shot_examples: list[tuple[str, str]] = None,
    ):
        # log_thread_safe(f"---\nUSING {model_id}\n---")
        self._model_id = model_id
        self._sys_prompt = sys_prompt
        self._chat, self._client = self.init_chat(
            sys_prompt=sys_prompt, history=self.format_history(shot_examples)
        )

    @staticmethod
    def get_instance(sys_prompt: str,
                     model_id: str = MODEL_ID,
                     shot_examples: list[tuple[str, str]] = None, ) -> "GeminiAdapter":
        return GeminiAdapter(sys_prompt, model_id, shot_examples)

    @staticmethod
    def init_content_config(sys_prompt: str, response_schema: Type[BaseModel] | None = None) -> GenerateContentConfig:
        return GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=TEMPERATURE,  # TODO check the impact of the temperature
        ) if not response_schema else GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=TEMPERATURE,  # TODO check the impact of the temperature
            response_mime_type="application/json",
            response_schema=response_schema,
        )

    @staticmethod
    def init_chat(
            sys_prompt: str, history: list[Content] | None, model_id: str = MODEL_ID
    ) -> tuple[Chat, Client]:
        client = init_genai_client_APIs()
        config = GeminiAdapter.init_content_config(sys_prompt=sys_prompt, )
        return (
            client.chats.create(model=model_id, history=history, config=config),
            client,
        )

    @staticmethod
    def format_history(raw_history: list[tuple[str, str]]) -> list[Content]:
        ret = []
        for d in raw_history:
            ret.append(Content(role="user", parts=[Part(d[0])]))
            ret.append(Content(role="model", parts=[Part(d[1])]))

        return ret

    def make_a_query(
            self,
            prompt: str,
            response_schema: Type[BaseModel] | None = None,
            schema_info: str | None = None,
            max_new_tokens: int = 200,
    ) -> str | Any:
        response = None
        last_exception: Exception = Exception(
            "response is None, but no exceptions have been raised when invoking self._chat._send_message()"
        )

        config = GeminiAdapter.init_content_config(sys_prompt=self._sys_prompt, response_schema=response_schema)

        for _ in range(NUM_API_KEYS * MAX_ATTEMPTS):

            try:
                response = self._chat.send_message(prompt, config=config)
                break
            except errors.ClientError as e:

                if (
                        e.code == 429 or e.code == 400
                ):  # Usage limit reached (429 RESOURCE_EXHAUSTED) or invalid API key (400 INVALID_ARGUMENT)
                    # Change API key and continue the loop
                    APIKeysManager.change_api_key()
                    self._chat, self._client = GeminiAdapter.init_chat(
                        self._sys_prompt,
                        model_id=self._model_id,
                        history=self._chat.get_history(),
                    )
                last_exception = e
            except errors.ServerError as e:
                log_thread_safe(GLOBAL_LOG, __file__, LogType.ERROR, f"[EXCEPTION] {e}")
                if e.code == 503:  # The model is overloaded. Please try again later.
                    time.sleep(0.3)
                last_exception = e

        if response is None:
            raise last_exception

        log_thread_safe(GLOBAL_LOG, __file__, LogType.LLM_OUTPUT,
                        "[ModelWrapper]\n"
                        f"PROMPT: {prompt}\n\n"
                        f"RESPONSE: {response.text if response and response.parts and response.text else ''}"
                        )

        if response_schema:
            return response.parsed

        return response.text if response and response.parts and response.text else ""
