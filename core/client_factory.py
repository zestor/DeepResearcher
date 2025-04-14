"""client_factory.py"""

from openai import OpenAI
from groq import Groq
from together import Together
from core.fireworks_client import FireworksAiClient
from core.config import (
    Service,
    OPENAI_API_KEY,
    FIREWORKS_API_KEY,
    GROQ_API_KEY,
    TOGETHER_API_KEY,
)


def get_client(use_service: Service):
    """
    Returns an instantiated client for the specified AI service.

    Args:
        use_service (Service): The service identifier.

    Raises:
        ValueError: If the service is unsupported.

    Returns:
        Any: The client instance.
    """
    service_mapping = {
        Service.DEEPSEEK: lambda: OpenAI(base_url="http://localhost:9001"),
        Service.GROQ: Groq,
        Service.OPENAI: OpenAI,
        Service.TOGETHER: Together,
        Service.FIREWORKS: FireworksAiClient,
    }
    if use_service not in service_mapping:
        raise ValueError(f"Unsupported service: {use_service}")
    client_class_or_factory = service_mapping[use_service]
    client = (
        client_class_or_factory()
        if callable(client_class_or_factory)
        else client_class_or_factory()
    )
    # Set API keys based on service
    api_key_mapping = {
        Service.DEEPSEEK: "nothing",
        Service.GROQ: GROQ_API_KEY,
        Service.OPENAI: OPENAI_API_KEY,
        Service.TOGETHER: TOGETHER_API_KEY,
        Service.FIREWORKS: FIREWORKS_API_KEY,
    }
    client.api_key = api_key_mapping[use_service]
    return client
