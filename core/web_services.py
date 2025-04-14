"""web_services.py"""

from core.firecrawl_client import FireCrawlClient
from core.perplexity_client import PerplexityClient


def web_search(query: str, recency: str = "month") -> str:
    """
    Calls the Perplexity AI API with the given query.

    Args:
        query (str): Question for perplexity to answer.
        recency (str): day, month, etc...

    Returns:
        str: Perplexity response
    """
    p = PerplexityClient()
    return p.call_perplexity(query, recency)


def call_web_content_retriever(url: str) -> str:
    """
    Calls FireCrawl to fetch content from a URL.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The scraped markdown content or an error message.
    """
    try:
        f = FireCrawlClient()
        return f.scrape_with_firecrawl(url)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error returning markdown data from {url}: {str(e)}"
