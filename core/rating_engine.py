from typing import Optional

import numpy as np

from models.schemas import PastInteraction, UserPersona


def compute_rating_prior(persona: UserPersona) -> dict:
    history = persona.interaction_history

    if not history:
        return {
            "mean_rating": 3.0,
            "std_rating": 1.0,
            "never_gives_5": False,
            "never_gives_1": False,
            "harsh_rater": False,
            "generous_rater": False,
            "rating_distribution": {"1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0, "5": 0.0},
            "category_biases": {},
        }

    ratings = [i.rating_given for i in history]
    mean_r = float(np.mean(ratings))
    std_r = float(np.std(ratings))
    total = len(ratings)

    # Rating distribution as percentages per whole-star bucket
    distribution = {}
    for star in range(1, 6):
        count = sum(1 for r in ratings if int(round(r)) == star)
        distribution[str(star)] = round(count / total, 4)

    # Per-category average
    cat_ratings: dict[str, list[float]] = {}
    for i in history:
        cat_ratings.setdefault(i.item_category, []).append(i.rating_given)
    category_biases = {cat: float(np.mean(vals)) for cat, vals in cat_ratings.items()}

    return {
        "mean_rating": round(mean_r, 4),
        "std_rating": round(std_r, 4),
        "never_gives_5": not any(r >= 5.0 for r in ratings),
        "never_gives_1": not any(r <= 1.0 for r in ratings),
        "harsh_rater": mean_r < 2.5,
        "generous_rater": mean_r > 4.0,
        "rating_distribution": distribution,
        "category_biases": category_biases,
    }


def calibrate_rating(
    raw_rating: float,
    historical_avg: float,
    historical_std: float,
    never_gives_5: bool = False,
    never_gives_1: bool = False,
    harsh_rater: bool = False,
    generous_rater: bool = False,
) -> float:
    # 1. Hard caps for users who never hit the extremes
    if never_gives_5 and raw_rating >= 5.0:
        raw_rating = 4.5

    if never_gives_1 and raw_rating <= 1.0:
        raw_rating = 1.5

    # 2. Regression to the mean — prevents the LLM drifting to extremes
    weight = 0.3
    raw_rating = (weight * historical_avg) + ((1 - weight) * raw_rating)

    # 3. Harsh rater: pull high scores back down
    if harsh_rater and raw_rating > 4.0:
        raw_rating = raw_rating * 0.85

    # 4. Generous rater: nudge very low scores up slightly
    if generous_rater and raw_rating < 2.0:
        raw_rating = raw_rating * 1.15

    # 5. Clip and round to nearest 0.5
    raw_rating = max(1.0, min(5.0, raw_rating))
    return round(raw_rating * 2) / 2


def compute_importance_score(
    interaction: PastInteraction,
    target_category: str,
    current_timestamp: Optional[str] = None,
) -> float:
    score = 0.0

    # Category relevance (40%)
    category_match = 1.0 if interaction.item_category == target_category else 0.0
    score += 0.4 * category_match

    # Engagement depth via review length (30%)
    review_length = len(interaction.review_text.split())
    engagement = min(review_length / 100, 1.0)
    score += 0.3 * engagement

    # Base participation score (30%)
    score += 0.3

    return min(score, 1.0)


def filter_relevant_memory(
    interactions: list[PastInteraction],
    target_category: str,
    threshold: float = 0.4,
    max_items: int = 5,
) -> list[PastInteraction]:
    if not interactions:
        return []

    # Index of the three most recent interactions for recency bonus
    recent_indices = set(range(max(0, len(interactions) - 3), len(interactions)))

    scored: list[tuple[float, PastInteraction]] = []
    for idx, interaction in enumerate(interactions):
        score = compute_importance_score(interaction, target_category)
        if idx in recent_indices:
            score = min(score + 0.2, 1.0)
        if score >= threshold:
            scored.append((score, interaction))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [interaction for _, interaction in scored[:max_items]]
