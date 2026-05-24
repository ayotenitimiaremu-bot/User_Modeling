import hashlib
import itertools
import random
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console

from models.schemas import PastInteraction, PersonalityTraits, ProductDetails, UserPersona

_console = Console()

GENRES = ["fiction", "non-fiction", "self-help", "biography",
          "science", "history", "mystery", "fantasy"]

_OCCUPATIONS = [
    "software engineer", "teacher", "accountant", "student",
    "doctor", "trader", "civil servant", "banker",
    "journalist", "entrepreneur",
]

_BOOKS_JSONL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/Books.jsonl"
)


def _stable_seed(s: str) -> int:
    """MD5-based seed — reproducible across Python sessions."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2 ** 32)


class GoodreadsLoader:
    """
    Loads Amazon Books Reviews 2023 data and builds UserPersona objects
    for the rest of the pipeline.
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.personas: dict[str, UserPersona] = {}
        self._genre_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, n_samples: int = 2000) -> pd.DataFrame:
        _console.print("[Loading] Fetching Amazon Books Reviews 2023 dataset...")
        try:
            df = self._load_real(n_samples)
        except Exception as exc:
            raise RuntimeError(
                "Amazon Reviews dataset failed to load. "
                "Check your internet connection. "
                "This system requires real data — no synthetic fallback."
            ) from exc
        self.df = df
        _console.print(
            f"[green][REAL DATA][/green] Loaded {len(df)} reviews "
            f"from {df['user_id'].nunique()} users"
        )
        return df

    def build_user_personas(self, df: Optional[pd.DataFrame] = None) -> dict[str, UserPersona]:
        if df is None:
            if self.df is None:
                raise RuntimeError("No data loaded. Call load() first.")
            df = self.df

        self.personas = {}
        has_date = "review_date" in df.columns

        for user_id, group in df.groupby("user_id"):
            user_id = str(user_id)
            reviews: list[str] = group["review_text"].tolist()
            ratings: list[float] = group["rating"].tolist()
            genres: list[str] = group["genre"].tolist()

            personality = self._infer_personality(reviews, ratings, genres)

            rng = random.Random(_stable_seed(user_id))
            age = rng.randint(20, 55)
            occupation = rng.choice(_OCCUPATIONS)

            history: list[PastInteraction] = []
            for _, row in group.iterrows():
                timestamp = None
                if has_date:
                    raw = row["review_date"]
                    timestamp = str(raw) if raw is not None and pd.notna(raw) else None

                history.append(
                    PastInteraction(
                        item_id=str(row["book_id"]),
                        item_title=str(row["book_title"]),
                        item_category=str(row["genre"]),
                        rating_given=float(row["rating"]),
                        review_text=str(row["review_text"])[:300],
                        timestamp=timestamp,
                    )
                )

            self.personas[user_id] = UserPersona(
                user_id=user_id,
                age=age,
                occupation=occupation,
                personality=personality,
                interaction_history=history,
            )

        return self.personas

    def get_persona(self, user_id: str) -> Optional[UserPersona]:
        if not self.personas:
            self.build_user_personas()
        return self.personas.get(user_id)

    def get_sample_personas(self, n: int = 3) -> list[UserPersona]:
        return list(self.personas.values())[:n]

    def get_preferred_genres(self, user_id: str) -> list[str]:
        persona = self.personas.get(user_id)
        if not persona:
            return []
        counts: dict[str, int] = {}
        for interaction in persona.interaction_history:
            counts[interaction.item_category] = counts.get(interaction.item_category, 0) + 1
        return sorted(counts, key=lambda g: counts[g], reverse=True)

    def get_reviewed_titles(self, user_id: str) -> list[str]:
        persona = self.personas.get(user_id)
        if not persona:
            return []
        return [i.item_title for i in persona.interaction_history]

    def get_sample_users_for_demo(self, n: int = 3) -> list[str]:
        """Return user IDs with the most interaction history, suitable for demo."""
        ranked = sorted(
            self.personas.items(),
            key=lambda kv: len(kv[1].interaction_history),
            reverse=True,
        )
        return [uid for uid, _ in ranked[:n]]

    def get_product_from_record(self, row: pd.Series) -> ProductDetails:
        review_title = (
            row.get("review_title", row["book_id"])
            if hasattr(row, "get")
            else row["book_id"]
        )
        return ProductDetails(
            product_id=str(row["book_id"]),
            title=str(review_title),
            category=str(row["genre"]),
            description=f"A {row['genre']} book. Review headline: {review_title}",
            metadata={"source": "Amazon Books Reviews 2023"},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_real(self, n_samples: int) -> pd.DataFrame:
        from datasets import load_dataset  # noqa: PLC0415

        dataset = load_dataset(
            "json", data_files=_BOOKS_JSONL, split="train", streaming=True
        )
        records = list(itertools.islice(dataset, n_samples))
        df = pd.DataFrame(records)

        # Keep asin — used to detect real data downstream.
        df["book_id"] = df["asin"].astype(str)
        df = df.rename(columns={
            "text": "review_text",
            "title": "review_title",
            "timestamp": "review_date",
        })

        df["genre"] = df["book_id"].map(self._get_genre)
        df["book_title"] = df["review_title"].fillna(df["book_id"])

        # Convert Unix-ms timestamp to date string if applicable.
        if df["review_date"].dtype in (np.int64, np.float64, object):
            converted = pd.to_datetime(df["review_date"], unit="ms", errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                df["review_date"] = converted.dt.strftime("%Y-%m-%d")
            else:
                df["review_date"] = df["review_date"].astype(str)

        # Quality filters.
        df = df.dropna(subset=["review_text", "user_id"])
        df = df[df["review_text"].str.len() >= 10]

        counts = df.groupby("user_id").size()
        valid_users = counts[counts >= 3].index
        df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

        if len(df) < 10:
            raise ValueError(
                f"Only {len(df)} rows after quality filtering — "
                "check your network connection or increase n_samples."
            )

        return df[["user_id", "book_id", "asin", "rating", "review_text",
                   "review_title", "book_title", "review_date", "genre"]]

    def _get_genre(self, asin: str) -> str:
        if asin not in self._genre_map:
            self._genre_map[asin] = GENRES[
                int(hashlib.md5(asin.encode()).hexdigest(), 16) % len(GENRES)
            ]
        return self._genre_map[asin]

    def _infer_personality(
        self,
        reviews: list[str],
        ratings: list[float],
        genres: list[str],
    ) -> PersonalityTraits:
        unique_genres = len(set(genres))
        openness = 3 if unique_genres >= 3 else (2 if unique_genres == 2 else 1)

        avg_review_length = float(np.mean([len(r.split()) for r in reviews]))
        conscientiousness = 3 if avg_review_length > 80 else (2 if avg_review_length > 30 else 1)

        pct_high = sum(1 for r in ratings if r >= 4) / len(ratings)
        extraversion = 3 if pct_high > 0.65 else (2 if pct_high > 0.40 else 1)

        std = float(np.std(ratings))
        agreeableness = 3 if std < 0.8 else (2 if std < 1.3 else 1)
        neuroticism = 3 if std > 1.5 else (2 if std > 1.0 else 1)

        return PersonalityTraits(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich import print as rprint
    from rich.table import Table

    loader = GoodreadsLoader()
    df = loader.load(n_samples=500)
    personas = loader.build_user_personas(df)

    table = Table(title="Sample User Personas — Amazon Reviews 2023")
    table.add_column("User ID")
    table.add_column("Age")
    table.add_column("Occupation")
    table.add_column("Openness")
    table.add_column("Avg Rating")
    table.add_column("Reviews")

    for uid, p in list(personas.items())[:5]:
        avg = sum(i.rating_given for i in p.interaction_history) / len(p.interaction_history)
        table.add_row(
            uid, str(p.age), p.occupation,
            str(p.personality.openness), f"{avg:.1f}",
            str(len(p.interaction_history)),
        )

    rprint(table)
    rprint(f"\nTotal users loaded: {len(personas)}")
    rprint("Data source: REAL (Amazon Books Reviews 2023)")
