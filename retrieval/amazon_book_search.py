import re

from tavily import TavilyClient

from config.settings import settings
from models.schemas import ProductDetails


class AmazonBookSearcher:
    def __init__(self) -> None:
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        self.base_domain = "amazon.com"

    # ------------------------------------------------------------------
    # Search & fetch
    # ------------------------------------------------------------------

    def search_books_by_genre(
        self,
        genre: str,
        exclude_titles: list[str] | None = None,
        max_results: int = 5,
    ) -> list[dict]:
        query = f"site:amazon.com best {genre} books customer reviews highly rated"
        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results * 2,
            include_raw_content=True,
            include_domains=["amazon.com"],
        )
        raw_results = response.get("results", [])

        filtered = []
        for r in raw_results:
            url = r.get("url", "")
            if "amazon.com" not in url:
                continue
            if "/dp/" not in url and "/gp/product/" not in url:
                continue
            if exclude_titles:
                title = r.get("title", "").lower()
                if any(ex.lower() in title or title in ex.lower()
                       for ex in exclude_titles):
                    continue
            filtered.append(r)

        if not filtered:
            raise ValueError(
                f"No Amazon book results found for genre: {genre}. "
                "Check TAVILY_API_KEY."
            )
        return filtered

    def fetch_book_details(self, amazon_url: str) -> dict:
        response = self.client.search(
            query=f"site:{amazon_url}",
            search_depth="advanced",
            max_results=1,
            include_raw_content=True,
        )
        if not response.get("results"):
            return {}
        raw = response["results"][0]
        return {
            "url": amazon_url,
            "title": raw.get("title", ""),
            "content": raw.get("content", ""),
            "raw_content": raw.get("raw_content", ""),
        }

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def extract_book_metadata(self, search_result: dict) -> dict | None:
        title = self.clean_amazon_title(search_result.get("title", ""))
        url = search_result.get("url", "")
        content = search_result.get("content", "")
        raw = search_result.get("raw_content", "") or content

        rating = self.extract_star_rating(raw)
        review_count = self.extract_review_count(raw)
        description = self.extract_description(raw, title)
        asin = self.extract_asin(url)

        if not title or not description:
            return None

        return {
            "title": title,
            "url": url,
            "asin": asin,
            "amazon_avg_rating": rating,
            "review_count": review_count,
            "description": description,
            "raw_snippet": content[:500],
        }

    def clean_amazon_title(self, title: str) -> str:
        if "|" in title:
            title = title.split("|")[0]
        for pattern in [
            r":\s*Amazon\.co\.uk:.*$",
            r":\s*Amazon\.com:.*$",
        ]:
            title = re.sub(pattern, "", title, flags=re.IGNORECASE)
        title = re.sub(r"^Amazon\.com:\s*", "", title, flags=re.IGNORECASE)
        title = re.sub(r":\s*Books$", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s*[-–]\s*Kindle [Ee]dition.*$", "", title)
        title = re.sub(r"\s*[-–]\s*[Pp]aperback.*$", "", title)
        title = re.sub(r":\s*by\s+[A-Z][^:]+$", "", title)
        title = title.strip()
        return title[:100]

    def extract_star_rating(self, text: str) -> float | None:
        patterns = [
            r"(\d+\.?\d*)\s+out\s+of\s+5\s+stars",
            r"(\d+\.?\d*)/5",
            r"Rated\s+(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+stars",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if 1.0 <= value <= 5.0:
                        return value
                except ValueError:
                    continue
        return None

    def extract_review_count(self, text: str) -> int | None:
        patterns = [
            r"([\d,]+)\s+(?:global\s+)?ratings",
            r"([\d,]+)\s+(?:customer\s+)?reviews",
            r"([\d,]+)\s+ratings",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None

    def extract_description(self, text: str, title: str) -> str | None:
        # Strategy 1: look for editorial section markers
        for marker in ["About the book", "Editorial description",
                        "From the Publisher", "Book Description"]:
            idx = text.lower().find(marker.lower())
            if idx != -1:
                snippet = text[idx + len(marker):idx + len(marker) + 400].strip()
                snippet = self._clean_description(snippet)
                if snippet and len(snippet) >= 50:
                    return snippet

        # Strategy 2: text following the title
        idx = text.lower().find(title.lower())
        if idx != -1 and idx + len(title) + 10 < len(text):
            snippet = text[idx + len(title):idx + len(title) + 400].strip()
            snippet = self._clean_description(snippet)
            if snippet and len(snippet) >= 50:
                return snippet

        # Strategy 3: sentences with book-describing words
        book_words = {"explores", "follows", "story of", "guide to", "introduces",
                      "reveals", "chronicles", "journey", "teaches", "examines",
                      "bestselling", "award-winning", "author"}
        sentences = re.split(r"(?<=[.!?])\s+", text)
        good_sentences = [
            s for s in sentences
            if any(w in s.lower() for w in book_words) and len(s) > 40
        ]
        if good_sentences:
            snippet = " ".join(good_sentences[:2])
            snippet = self._clean_description(snippet)
            if snippet and len(snippet) >= 50:
                return snippet[:400]

        # Strategy 4: longest clean paragraph
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        price_pattern = re.compile(r"[£$₦]\d|\bIn Stock\b|\bAdd to [Cc]art\b|\bBuy now\b",
                                   re.IGNORECASE)
        candidates = [
            p for p in paragraphs
            if len(p) >= 100 and not price_pattern.search(p)
        ]
        if candidates:
            snippet = self._clean_description(max(candidates, key=len))
            if snippet and len(snippet) >= 50:
                return snippet[:400]

        return None

    def extract_asin(self, url: str) -> str | None:
        match = re.search(r"/dp/([A-Z0-9]{10})", url)
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def find_book_for_user(
        self,
        preferred_genres: list[str],
        reviewed_titles: list[str],
        max_attempts: int = 3,
    ) -> ProductDetails:
        for genre in preferred_genres[:max_attempts]:
            try:
                raw_results = self.search_books_by_genre(
                    genre=genre,
                    exclude_titles=reviewed_titles,
                    max_results=5,
                )
            except ValueError:
                continue

            for raw_result in raw_results:
                metadata = self.extract_book_metadata(raw_result)
                if metadata is None:
                    continue

                title = metadata["title"]
                already_reviewed = any(
                    reviewed.lower() in title.lower() or title.lower() in reviewed.lower()
                    for reviewed in reviewed_titles
                )
                if already_reviewed:
                    continue

                return ProductDetails(
                    product_id=metadata["asin"] or f"amzn_{hash(title) % 99999}",
                    title=title,
                    category=genre,
                    description=metadata["description"],
                    metadata={
                        "amazon_avg_rating": metadata["amazon_avg_rating"],
                        "review_count": metadata["review_count"],
                        "amazon_url": metadata["url"],
                        "asin": metadata["asin"],
                        "source": "Amazon.com (via Tavily)",
                        "genre": genre,
                    },
                    image_url=None,
                )

        raise RuntimeError(
            f"Could not find a valid unseen Amazon book for user "
            f"after searching genres: {preferred_genres[:max_attempts]}. "
            "Tavily returned insufficient metadata."
        )

    def get_real_ground_truth(self, product: ProductDetails) -> float | None:
        return product.metadata.get("amazon_avg_rating")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_description(text: str) -> str:
        text = re.sub(r"[£$₦]\s*[\d,.]+", "", text)
        text = re.sub(r"\b(Add to [Cc]art|Buy now|In [Ss]tock|Free delivery)\b", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
