import hashlib
import itertools
import random
from datetime import date, timedelta
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

# ---------------------------------------------------------------------------
# Per-profile review templates (5–8 each, matching tone and length target)
# ---------------------------------------------------------------------------

_REVIEW_TEMPLATES: dict[str, list[str]] = {
    "harsh_critic": [
        "Disappointing. The premise had potential but execution was poor. The pacing dragged in the middle sections and the conclusion felt rushed. Not worth the time.",
        "Overrated. Everyone seems to love this but I found it repetitive and the author's argument could have been made in half the pages. Save your money.",
        "Had high hopes but this let me down. The characters felt flat and the plot went nowhere interesting. Gave up halfway through and read the summary.",
        "I forced myself to finish this. The first chapter promised something fresh but the rest devolved into the same tired tropes we have seen a hundred times. The author clearly has talent but it is wasted here. Do not believe the five-star reviews.",
        "Two stars because there were a handful of genuinely interesting ideas scattered throughout, but they were buried under so much filler. The writing style grated on me from page one. I kept hoping it would improve. It did not.",
        "Not for me at all. I appreciate that others love this but I found the central argument unconvincing and the examples cherry-picked to the point of dishonesty. There are far better books on this topic.",
        "Another overhyped bestseller that fails to deliver. The author spends the first third telling you what they are going to say and the final third repeating what they said. Very little substance in between.",
        "Disappointing from an author I usually enjoy. The research is thin, the conclusions are obvious, and the self-congratulatory tone grows old fast. Expected much better and got much less.",
    ],
    "generous_reader": [
        "Loved every page! This book completely changed my perspective. The author writes with such clarity and warmth. Highly recommended!",
        "A wonderful read. Couldn't put it down. Perfect for anyone looking for inspiration.",
        "Absolutely brilliant. One of the best books I have read this year. Already bought copies for friends and family.",
        "This exceeded all my expectations. Beautiful writing, powerful message. I will be recommending this to everyone I know.",
        "Incredible. Moved me deeply. The kind of book that stays with you long after you finish.",
        "Five stars easily. A joy from the first page to the last. The author truly understands people.",
        "Amazing! Read it in one sitting. Cannot wait to read more from this author. A genuine gem.",
    ],
    "moderate_detail": [
        "Solid read with some real strengths and a few weaknesses. The first half kept me engaged but the pacing slowed considerably in the middle. Worth reading but manage expectations.",
        "There is good stuff here. The core ideas are sound and the writing is clean. But it could have been edited down by about a third without losing anything important.",
        "Three and a half stars rounded up. Enjoyable enough but forgettable. Good for a long flight but not something I will revisit.",
        "Mixed feelings on this one. Some chapters were genuinely insightful and others felt padded. The conclusion was stronger than I expected after a rocky middle section.",
        "Decent book. Not the revelation some reviewers claim but a competent and occasionally illuminating read. Worth picking up if the subject genuinely interests you.",
        "I liked it more than I expected after a slow start. The second half is significantly stronger. The author finds their voice around chapter six and does not let go.",
        "Good ideas, uneven execution. The author makes some genuinely original points but buries them in unnecessary context. Worth skimming even if not reading cover to cover.",
    ],
    "fantasy_fan": [
        "The world-building here is extraordinary. Every detail feels considered, the magic system has real internal logic, and the political intrigue keeps the stakes high throughout. A few pacing issues in the middle chapters but nothing that derails the experience. Highly recommended for fans of deep, immersive fantasy.",
        "I was gripped from the first page. The author has created a world I genuinely wanted to live in. The characters feel fully realised, the dialogue is sharp, and the plot twists landed hard every time. The ending leaves room for a sequel and I cannot wait.",
        "Beautiful and brutal in equal measure. This is the kind of fantasy that respects your intelligence and rewards careful reading. Some readers may find the slow burn frustrating but patience pays off enormously. One of the best in the genre I have read in years.",
        "A worthy entry in epic fantasy. The prose is lush without becoming purple and the author avoids most of the genre clichés. The magic system is inventive and the moral ambiguity of the main cast makes for genuinely interesting storytelling.",
        "Not quite perfect but very, very good. The mid-section loses momentum but the opening and closing sections are exceptional. The world feels lived-in in a way that lesser fantasy never achieves. Looking forward to whatever this author does next.",
        "Remarkable scope and execution. Building a world this coherent while keeping the human drama front and centre is genuinely hard to do. This author makes it look easy. The ending hit harder than I expected.",
    ],
    "practical_reader": [
        "Good practical advice backed by solid research. I finished this with a list of specific changes to make. That is all I ask of a book like this and it delivered cleanly.",
        "Worth reading for the frameworks alone. The author gives you tools you can apply immediately rather than vague inspiration. A few chapters felt like filler but the core content is excellent.",
        "Useful, clear, and well-organised. Not groundbreaking but a solid addition to the genre. Took notes throughout and will definitely revisit.",
        "The ideas are not new but this is one of the better presentations of them. The examples are well chosen and the writing is clear. Practical and immediately actionable.",
        "Helpful but could be shorter. About two-thirds of the book earns its place. The rest is repetition. Still, the parts that work really work and I applied them the same week.",
        "A decent return on investment for the time spent reading. The chapter on decision-making alone was worth the price. Would recommend to anyone early in their career or in a period of change.",
        "Practical and honest. The author does not oversell and that restraint builds real trust. I came away with three concrete habits I have already started. Rare for this genre.",
    ],
    "casual_rater": [
        "It was OK. Nothing special but I do not regret reading it.",
        "Some good parts, some dull parts. Pretty standard for this type of book.",
        "Read it in a weekend. Fine.",
        "Better than I expected actually. Would recommend to the right reader.",
        "Meh. Not bad but not great. Glad it was a library copy.",
        "Pretty good I suppose. Would not read again but glad I did once.",
        "Has its moments. Mostly average but harmless.",
        "Solid enough. Forgot most of it within a week but enjoyed it at the time.",
    ],
    "history_buff": [
        "Meticulously researched and compellingly written. The author draws on primary sources throughout and the footnotes alone are worth the price of admission. A few anachronistic framings that bothered me but these are minor complaints against an otherwise exceptional work of popular history.",
        "This is what popular history should look like. Rigorous without being dry, accessible without talking down to the reader. The author clearly spent years in the archives and it shows on every page. Mandatory reading for anyone serious about the period.",
        "A thorough and engaging account that handles a genuinely complex subject with appropriate nuance. I learned a great deal and found my prior assumptions challenged in productive ways. A few chapters felt overly long but the scholarship is impeccable throughout.",
        "Excellent overview with some genuinely fresh analysis in the later chapters. The narrative structure works well for a subject that could easily feel fragmented. Not for complete beginners but highly recommended for anyone with existing interest in the period.",
        "I have read everything I could find on this subject and this stands up well against the best of it. The author adds meaningful new perspective rather than just summarising existing scholarship. Four stars only because the final chapter felt rushed relative to the care given elsewhere.",
        "Compelling and well-paced for a topic that is easy to render dull in the wrong hands. The author's enthusiasm is infectious without ever becoming sloppy. One of the better entries in this increasingly crowded field.",
    ],
    "never_gives_5": [
        "Very good, not quite great. The author comes close to something exceptional several times without fully getting there. Still, four stars for a book that makes you think and mostly delivers on its promises.",
        "Impressive work. There are a couple of sections that could have used another editing pass and one argument that does not hold up well under scrutiny, but the overall quality is high. Almost five stars.",
        "Strong book. Would have been a five-star read if the final third matched the first two. As it is, a solid four. Well worth reading.",
        "Really enjoyed this. It earns nearly every page. The writing is polished and the ideas are interesting. One or two chapter conclusions felt like a stretch, which is why I cannot give the full five.",
        "Close to a masterpiece but not quite. There is a real intelligence behind this book and it shows. I just wish the middle section had been as tight as the opening. Four stars from me.",
        "Great read with a few rough edges. The research is impressive and the narrative moves well. Not quite the definitive treatment of the subject but easily the best available right now.",
        "Four stars. If the author ever writes a second edition with the weak middle chapters reconsidered, it becomes a five. As it stands it is still excellent and worth your time.",
    ],
}

_SYNTHETIC_USERS = [
    {"user_id": "synth_u001", "profile": "harsh_critic",    "avg_rating": 2.3, "rating_std": 0.8, "genres": ["fiction", "mystery"]},
    {"user_id": "synth_u002", "profile": "generous_reader", "avg_rating": 4.6, "rating_std": 0.5, "genres": ["self-help", "biography"]},
    {"user_id": "synth_u003", "profile": "moderate_detail", "avg_rating": 3.5, "rating_std": 1.2, "genres": ["science", "history", "non-fiction"]},
    {"user_id": "synth_u004", "profile": "fantasy_fan",     "avg_rating": 4.0, "rating_std": 0.9, "genres": ["fantasy", "fiction"]},
    {"user_id": "synth_u005", "profile": "practical_reader","avg_rating": 3.8, "rating_std": 0.7, "genres": ["self-help", "science"]},
    {"user_id": "synth_u006", "profile": "casual_rater",    "avg_rating": 3.2, "rating_std": 1.5, "genres": ["fiction", "mystery", "fantasy"]},
    {"user_id": "synth_u007", "profile": "history_buff",    "avg_rating": 4.1, "rating_std": 0.6, "genres": ["history", "biography"]},
    {"user_id": "synth_u008", "profile": "never_gives_5",   "avg_rating": 3.9, "rating_std": 0.8, "genres": ["science", "non-fiction", "biography"]},
]

_PREFIXES = ["", "Quick review: ", "After finishing this, ", "Honest opinion: ", "Just finished this. "]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _stable_seed(s: str) -> int:
    """MD5-based seed — reproducible across Python sessions (hash() is randomised)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


def _sentiment_from_rating(rating: float) -> str:
    if rating >= 4.0:
        return "positive"
    if rating >= 3.0:
        return "neutral"
    return "negative"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GoodreadsLoader:
    """
    Loads Amazon Books Reviews 2023 data (or falls back to synthetic) and
    builds UserPersona objects for the rest of the pipeline.
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.personas: dict[str, UserPersona] = {}
        self._genre_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, n_samples: int = 2000) -> pd.DataFrame:
        try:
            _console.print("[Loading] Fetching Amazon Books Reviews dataset...")
            df = self._load_real(n_samples)
            self.df = df
            _console.print(
                f"[green][REAL DATA][/green] Loaded {len(df)} reviews "
                f"from {df['user_id'].nunique()} users"
            )
            return df
        except Exception as exc:
            _console.print(
                f"[yellow][SYNTHETIC][/yellow] Real dataset unavailable ({exc}). "
                "Using synthetic data."
            )
            self.df = self.generate_synthetic_data(n_samples)
            return self.df

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
                if has_date:
                    raw = row["review_date"]
                    timestamp = str(raw) if raw is not None and pd.notna(raw) else None
                else:
                    timestamp = None

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

    def get_product_from_record(self, row: pd.Series) -> ProductDetails:
        review_title = row.get("review_title", row["book_id"]) if hasattr(row, "get") else row["book_id"]
        return ProductDetails(
            product_id=str(row["book_id"]),
            title=str(review_title),
            category=str(row["genre"]),
            description=f"A {row['genre']} book. Review headline: {review_title}",
            metadata={"source": "Amazon Books Reviews 2023"},
        )

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------

    def generate_synthetic_data(self, n: int = 2000) -> pd.DataFrame:
        rows: list[dict] = []
        base_date = date(2020, 1, 1)
        date_range = (date(2025, 1, 1) - base_date).days

        for user in _SYNTHETIC_USERS:
            rng_py = random.Random(_stable_seed(user["user_id"]))
            rng_np = np.random.default_rng(_stable_seed(user["user_id"]))

            n_reviews = rng_py.randint(15, 25)
            templates = _REVIEW_TEMPLATES[user["profile"]]

            for _ in range(n_reviews):
                genre = rng_py.choice(user["genres"])
                book_id = f"book_{genre[:3]}_{rng_py.randint(100, 999)}"

                rating_raw = float(rng_np.normal(user["avg_rating"], user["rating_std"]))
                rating = round(np.clip(rating_raw, 1.0, 5.0) * 2) / 2
                rating = float(max(1.0, min(5.0, rating)))
                if user["profile"] == "never_gives_5":
                    rating = min(4.5, rating)

                template = rng_py.choice(templates)
                prefix = rng_py.choice(_PREFIXES)
                review_text = (prefix + template).strip()

                review_title = f"{genre.capitalize()} read"
                review_date = (base_date + timedelta(days=rng_py.randint(0, date_range - 1))).isoformat()

                rows.append({
                    "user_id": user["user_id"],
                    "book_id": book_id,
                    "rating": rating,
                    "review_text": review_text,
                    "review_title": review_title,
                    "book_title": f"{genre.capitalize()} Book",
                    "review_date": review_date,
                    "genre": genre,
                })

        df = pd.DataFrame(rows)
        _console.print(
            f"[SYNTHETIC] Generated {len(df)} synthetic reviews "
            f"for {len(_SYNTHETIC_USERS)} users"
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_real(self, n_samples: int) -> pd.DataFrame:
        from datasets import load_dataset  # noqa: PLC0415

        _BOOKS_JSONL = (
            "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
            "/resolve/main/raw/review_categories/Books.jsonl"
        )
        dataset = load_dataset("json", data_files=_BOOKS_JSONL, split="train", streaming=True)

        records = list(itertools.islice(dataset, n_samples))
        df = pd.DataFrame(records)

        # Keep asin in place — used later to detect real vs synthetic data.
        df["book_id"] = df["asin"].astype(str)
        df = df.rename(columns={
            "text": "review_text",
            "title": "review_title",
            "timestamp": "review_date",
        })

        # Assign genre deterministically from book_id (asin).
        df["genre"] = df["book_id"].map(self._get_genre)

        # Use review headline as book title proxy.
        df["book_title"] = df["review_title"].fillna(df["book_id"])

        # Convert timestamp (likely Unix ms) to date string.
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

        if len(df) < 100:
            raise ValueError(f"Only {len(df)} rows after filtering — not enough data.")

        return df[["user_id", "book_id", "asin", "rating", "review_text",
                   "review_title", "book_title", "review_date", "genre"]]

    def _get_genre(self, asin: str) -> str:
        if asin not in self._genre_map:
            self._genre_map[asin] = GENRES[int(hashlib.md5(asin.encode()).hexdigest(), 16) % len(GENRES)]
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

    table = Table(title="Sample User Personas")
    table.add_column("User ID")
    table.add_column("Age")
    table.add_column("Occupation")
    table.add_column("Openness")
    table.add_column("Avg Rating")
    table.add_column("Reviews")

    for uid, p in list(personas.items())[:3]:
        avg = sum(i.rating_given for i in p.interaction_history) / len(p.interaction_history)
        table.add_row(
            uid,
            str(p.age),
            p.occupation,
            str(p.personality.openness),
            f"{avg:.1f}",
            str(len(p.interaction_history)),
        )

    rprint(table)
    rprint("\nSample full persona:")
    rprint(list(personas.values())[0].model_dump())
    rprint(f"\nData source: {'REAL' if 'asin' in df.columns else 'SYNTHETIC'}")
