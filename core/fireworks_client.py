"""fireworks_client.py"""

import json
import requests
from openai.types.chat.chat_completion import ChatCompletion


class FireworksAiCompletions:
    """FireworksAiCompletions"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"

    def create(self, **kwargs) -> ChatCompletion:
        """
        Calls the Fireworks AI completions endpoint.

        Returns:
            ChatCompletion: The validated chat completion response.
        """
        if not self.api_key:
            raise ValueError(
                "API key is not set. Please set the api_key property on the FireworksAiClient."
            )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.request(
            "POST", self.url, headers=headers, data=json.dumps(kwargs), timeout=180
        )
        response.raise_for_status()
        return ChatCompletion.model_validate(response.json())


class FireworksAiChatClient:
    """FireworksAiChatClient"""

    def __init__(self, api_key: str):
        self.completions = FireworksAiCompletions(api_key)

    @property
    def api_key(self) -> str:
        """api_key"""
        return self.completions.api_key

    @api_key.setter
    def api_key(self, value: str):
        self.completions.api_key = value


class FireworksAiClient:
    """FireworksAiClient"""

    def __init__(self, api_key: str = None):
        from core.config import FIREWORKS_API_KEY

        if api_key is None:
            api_key = FIREWORKS_API_KEY
        self._api_key = api_key
        self.chat = FireworksAiChatClient(self._api_key)

    @property
    def api_key(self) -> str:
        """api_key"""
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value
        if self.chat:
            self.chat.api_key = value
