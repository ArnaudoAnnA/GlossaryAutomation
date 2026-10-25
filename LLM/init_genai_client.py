from enum import Enum, auto

import google.oauth2.id_token
from google import genai
from google.genai import errors, Client
from google.genai.types import HttpOptions

from globals.constants import MAX_ATTEMPTS
from logging_debugging.logging_debugging import GLOBAL_LOG, LogType, log_thread_safe
from .APIKeysManager import APIKeysManager, NUM_API_KEYS

# ------------------------- AIBooster setup ---------------------------------------------------------
PROJECT_ID = ""
REGION = ""
SERVICE_ACCOUNT = ""
LLM_PROXY_ENDPOINT = ""
# -------------------------------------------------------------------------------------------------------

def _init_genai_client_AIB(location: str = REGION, project: str = PROJECT_ID, llm_proxy_endpoint: str = LLM_PROXY_ENDPOINT) -> Client:
    """
    This function attempts to generate a Google GenAI Client that will query the referenced models on Google Cloud Platform.
    :param: region: the id of the region where the platform is hosted.
    :param: project: the id of the Google Cloud Platform project.
    :proxy_url: the string URL of the llm proxy.

    :rtype: Client
    :return: An initialized instance of ``genai.Client``.
    """

    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION, service_account=SERVICE_ACCOUNT)

    api_endpoint = f"{llm_proxy_endpoint}/google-llm"
    id_creds = google.oauth2.id_token.fetch_id_token_credentials(llm_proxy_endpoint)

    last_exception = None
    client = None
    for _ in range(MAX_ATTEMPTS):
        try:
            client = genai.Client(vertexai=True, location=location, project=project, credentials=id_creds, http_options=HttpOptions(base_url=api_endpoint, api_version="v1beta1"))
            break
        except errors.ClientError as e:
            log_thread_safe(GLOBAL_LOG, __file__, LogType.ERROR, f"[EXCEPTION] {e}")
            last_exception = e

    if client is None:
        raise last_exception
    return client


def _init_genai_client_APIs() -> Client:
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
            client = genai.Client(api_key=APIKeysManager.get_api_key())
            break
        except errors.ClientError as e:
            log_thread_safe(GLOBAL_LOG, __file__, LogType.ERROR, f"[EXCEPTION] {e}")
            last_exception = e
            APIKeysManager.change_api_key()

    if client is None:
        raise last_exception
    return client


class ClientMode(Enum):
    AIBOOSTER = auto()
    GOOGLE_AI_STUDIO_APIS = auto()


def init_genai_client(client_mode: ClientMode = ClientMode.GOOGLE_AI_STUDIO_APIS) -> Client:
    if client_mode == ClientMode.AIBOOSTER:
        return _init_genai_client_AIB()
    elif client_mode == ClientMode.GOOGLE_AI_STUDIO_APIS:
        return _init_genai_client_APIs()
    else:
        raise ValueError("Invalid client_mode: ", client_mode)
