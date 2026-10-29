import os
from multiprocessing import Value

from dotenv import load_dotenv

from constants import ROOT_DIR
from logging_debugging import log_thread_safe
from logging_debugging.logging_debugging import GLOBAL_LOG, LogType

load_dotenv(ROOT_DIR / ".env")
NUM_API_KEYS = int(os.getenv("NUM_GOOGLE_API_KEYS"))


class APIKeysManager:
    _counter_api_keys = Value("i", 1)

    @staticmethod
    def get_api_key():
        with APIKeysManager._counter_api_keys.get_lock():
            name = f"GOOGLE_API_KEY_{APIKeysManager._counter_api_keys.value}"
        # log_thread_safe(f"[DEBUG]   Using API key named {name}\n")
        return os.getenv(name)

    @staticmethod
    def change_api_key():
        with APIKeysManager._counter_api_keys.get_lock():
            if APIKeysManager._counter_api_keys.value == NUM_API_KEYS:
                print("All API keys have been used, restart.")  # TODO remove
                APIKeysManager._counter_api_keys.value = 1
            else:
                APIKeysManager._counter_api_keys.value += 1
            log_thread_safe(GLOBAL_LOG, __file__, LogType.LLM_OUTPUT, f"Changed API key. Now using key #{APIKeysManager._counter_api_keys.value}")
