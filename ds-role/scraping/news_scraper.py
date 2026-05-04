import time
import urllib.parse
from typing import List, Dict

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
}

REQUEST_DELAY = 1
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2


class NewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _safe_get(self, url: str):
        for _ in range(MAX_RETRIES):
            try:
                return self.session.get(url, timeout=REQUEST_TIMEOUT)
            except Exception:
                time.sleep(1)
        return None

    def _fetch_rss(self, url: str) -> List[Dict]:
        results = []
        resp = self._safe_get(url)

        if not resp:
            return results

        soup = BeautifulSoup(resp.content, "xml")

        for item in soup.find_all("item"):
            link = item.link.text if item.link else ""

            results.append({
                "title": item.title.text if item.title else "",
                "url": link,
                "source": urllib.parse.urlparse(link).netloc,
                "published": item.pubDate.text if item.pubDate else "",
                "snippet": BeautifulSoup(
                    item.description.text if item.description else "",
                    "html.parser"
                ).get_text()
            })

        return results

    def search_google_news(self, query: str) -> List[Dict]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=id&gl=ID&ceid=ID:id"
        return self._fetch_rss(url)

    def search_bing_news(self, query: str) -> List[Dict]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/news/search?q={encoded}&format=rss&mkt=id-ID"
        return self._fetch_rss(url)

    def search_yahoo_news(self, query: str) -> List[Dict]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://news.search.yahoo.com/rss?p={encoded}"
        return self._fetch_rss(url)

    def fetch_content(self, url: str) -> str:
        resp = self._safe_get(url)
        if not resp:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # basic cleaning
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        return content

    def search_all(self, query: str) -> List[Dict]:
        data = []
        seen = set()

        def add(results):
            for r in results:
                if r["url"] and r["url"] not in seen:
                    seen.add(r["url"])
                    data.append(r)

        add(self.search_google_news(query))
        time.sleep(REQUEST_DELAY)

        add(self.search_bing_news(query))
        time.sleep(REQUEST_DELAY)

        add(self.search_yahoo_news(query))

        return data

    def collect(self, query: str, with_content: bool = False) -> List[Dict]:
        results = self.search_all(query)

        if with_content:
            for r in results:
                r["content"] = self.fetch_content(r["url"])
                time.sleep(REQUEST_DELAY)

        return results


# ENTRY POINT (IMPORTANT)
def run_scraper(query: str, with_content: bool = False) -> List[Dict]:
    scraper = NewsScraper()
    return scraper.collect(query, with_content)