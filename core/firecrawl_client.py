"""program"""

import requests

from core.config import FIRECRAWL_API_KEY


class FireCrawlClient:
    """FireCrawlClient"""

    def scrape_with_firecrawl(self, url):
        """scrape_with_firecrawl"""
        endpoint = "https://api.firecrawl.dev/v1/scrape"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        }

        payload = {"url": url, "formats": ["markdown"], "timeout": 180000}

        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=180000
        )

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
