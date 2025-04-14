"""perplexity_client.py"""

import requests

from openai.types.chat.chat_completion import ChatCompletion

from core.llm_helpers import print_token_usage_details
from core.utilities import remove_think_text
from core.config import (
    Service,
    DEFAULT_PERPLEXITY_MODEL,
    PERPLEXITY_MODELS_WITH_SEARCH_CONTENT_SIZE,
    PERPLEXITY_SEARCH_CONTENT_SIZE,
    PERPLEXITY_API_KEY,
)


class PerplexityClient:
    """PerplexityClient"""

    def call_perplexity(self, query: str, recency: str = "month") -> str:
        """
        Calls the Perplexity AI API with the given query.
        Returns the text content from the model’s answer.
        """
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": DEFAULT_PERPLEXITY_MODEL,
            "messages": [
                {"role": "user", "content": query},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "search_recency_filter": recency,
            "stream": False,
        }

        if DEFAULT_PERPLEXITY_MODEL in PERPLEXITY_MODELS_WITH_SEARCH_CONTENT_SIZE:
            payload["web_search_options"] = {
                "search_context_size": PERPLEXITY_SEARCH_CONTENT_SIZE
            }

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)

            response.raise_for_status()
            data = response.json()
            chat_response = ChatCompletion.model_validate(data)

            print_token_usage_details(
                chat_response,
                Service.PERPLEXITY,
                DEFAULT_PERPLEXITY_MODEL,
                PERPLEXITY_SEARCH_CONTENT_SIZE,
            )
            retval = data["choices"][0]["message"]["content"]

            retval = remove_think_text(retval)

            joined_citations = "\n".join(
                [f"[{i+1}] {cite}" for i, cite in enumerate(data["citations"])]
            )
            citations = f"\n\nCitations:\n{joined_citations}"
            retval = retval + citations

            # print(f"* * *  Research Assistant Response  * * *\n\n{retval}\n\n")
            return retval
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error calling Perplexity API: {str(e)}"
