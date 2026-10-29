
from google import genai
from google.genai import errors, Client
from google.genai.types import HttpOptions

from logging_debugging.logging_debugging import GLOBAL_LOG, LogType, log_thread_safe
from utilities_google_ai.APIKeysManager import APIKeysManager, NUM_API_KEYS

# ------------------------- AIBooster setup ---------------------------------------------------------
PROJECT_ID = ""
REGION = ""
SERVICE_ACCOUNT = ""
LLM_PROXY_ENDPOINT = ""
# -------------------------------------------------------------------------------------------------------


def init_genai_client_APIs() -> Client:
    """
    This function attempts to create a ``genai.Client`` instance using available
    API keys managed by ``APIKeysManager``. It will iterate through the keys and
    try to instantiate the client until a valid one is found or exhausts all
    available keys. If no valid client can be created, the last exception raised
    is re-raised.

    :raises ClientError: If all attempts to instantiate a client using the
                         available API keys fail.
    :rtype: Client
    :return: An initialized instance of ``genai.Client``.
    """
    last_exception = None
    client = None
    for _ in range(NUM_API_KEYS):
        try:
            client = genai.Client(api_key=APIKeysManager.get_api_key(),)
            break
        except errors.ClientError as e:
            log_thread_safe(GLOBAL_LOG, __file__, LogType.ERROR, f"[EXCEPTION] {e}")
            last_exception = e
            APIKeysManager.change_api_key()

    if client is None:
        raise last_exception
    return client


