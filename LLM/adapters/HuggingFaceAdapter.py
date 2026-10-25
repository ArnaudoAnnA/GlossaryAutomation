import json
from typing import Any, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

from LLM.adapters.interface import LLMAdapterInterface
from constants import TEMPERATURE
from logging_debugging import log_thread_safe
from logging_debugging.logging_debugging import GLOBAL_LOG, LogType

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


class HuggingFaceAdapter(LLMAdapterInterface):

    def __init__(
            self,
            sys_prompt: str,
            model_id: str = MODEL_ID,
            shot_examples: list[dict[str, str]] = None,
    ):
        self._model_id = model_id
        self._sys_prompt = sys_prompt
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        self._history = self.init_chat(sys_prompt=sys_prompt, history=shot_examples)

    @staticmethod
    def init_chat(
            sys_prompt: str, history: list[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        chat_history = [{"role": "system", "content": sys_prompt}]
        if history:
            chat_history.extend(history)
        return chat_history

    @staticmethod
    def format_history(raw_dictionary: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": d["role"], "content": d["content"]} for d in raw_dictionary]

    def make_a_query(
            self,
            prompt: str,
            response_schema: Type[BaseModel] | None = None,
            max_new_tokens: int = 200,
    ) -> str | Any:
        self._history.append({"role": "user", "content": prompt})

        input_ids = self._tokenizer.apply_chat_template(
            self._history, return_tensors="pt", add_generation_prompt=True
        ).to(self._model.device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=True if TEMPERATURE > 0 else False,
            pad_token_id=self._tokenizer.eos_token_id
        )

        response_text = self._tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        self._history.append({"role": "assistant", "content": response_text})

        log_thread_safe(GLOBAL_LOG, __file__, LogType.LLM_OUTPUT,
                        "[HuggingFaceAdapter]\n"
                        f"PROMPT: {prompt}\n\n"
                        f"RESPONSE: {response_text}"
                        )

        if response_schema:
            try:
                return response_schema.model_validate_json(response_text)
            except Exception as e:
                log_thread_safe(GLOBAL_LOG, __file__, LogType.ERROR, f"JSON Parsing error: {e}")
                return response_text

        return response_text