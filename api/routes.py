import numpy as np
from fastapi import APIRouter, HTTPException

from config.settings import settings
from core.review_generator import ReviewGenerator
from data.amazon_reviews_2023 import GoodreadsLoader
from models.schemas import ProductDetails, ReviewOutput, ReviewRequest, UserPersona
from retrieval.amazon_book_search import AmazonBookSearcher

router = APIRouter()
generator = ReviewGenerator()
book_searcher = AmazonBookSearcher()

_loader: GoodreadsLoader | None = None


def _get_loader() -> GoodreadsLoader:
    global _loader
    if _loader is None:
        _loader = GoodreadsLoader()
        _loader.load(200)
        _loader.build_user_personas()
    return _loader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_personality_summary(persona: UserPersona) -> str:
    p = persona.personality
    traits = []

    if p.openness == 3:
        traits.append("reads widely across genres")
    elif p.openness == 1:
        traits.append("sticks to familiar genres")

    if p.conscientiousness == 3:
        traits.append("writes detailed reviews")
    elif p.conscientiousness == 1:
        traits.append("brief reviewer")

    if p.neuroticism == 3:
        traits.append("critical rater")
    elif p.extraversion == 3:
        traits.append("enthusiastic reader")

    avg = sum(i.rating_given for i in persona.interaction_history)
    avg /= max(len(persona.interaction_history), 1)

    if avg > 4.0:
        traits.append(f"generous rater (avg {avg:.1f}★)")
    elif avg < 3.0:
        traits.append(f"harsh rater (avg {avg:.1f}★)")
    else:
        traits.append(f"moderate rater (avg {avg:.1f}★)")

    return f"{persona.age}-year-old {persona.occupation} who " + ", ".join(traits) + "."


# ---------------------------------------------------------------------------
# POST /reviews/generate
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=ReviewOutput)
def generate_review(request: ReviewRequest) -> ReviewOutput:
    try:
        return generator.generate_review(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /reviews/batch
# ---------------------------------------------------------------------------

@router.post("/batch", response_model=list[ReviewOutput])
def batch_generate(requests: list[ReviewRequest]) -> list[ReviewOutput]:
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Max 10 requests per batch")
    results = []
    for req in requests:
        try:
            results.append(generator.generate_review(req))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return results


# ---------------------------------------------------------------------------
# GET /reviews/sample-request
# ---------------------------------------------------------------------------

@router.get("/sample-request")
def sample_request() -> dict:
    loader = _get_loader()
    persona = loader.get_sample_personas(1)[0]
    row = loader.df.iloc[0]
    product = loader.get_product_from_record(row)
    request = ReviewRequest(persona=persona, product=product)
    return {
        "note": "Copy this structure and modify for your use case",
        "request": request.model_dump(),
    }


# ---------------------------------------------------------------------------
# GET /reviews/sample-personas
# ---------------------------------------------------------------------------

@router.get("/sample-personas", response_model=list[UserPersona])
def sample_personas() -> list[UserPersona]:
    return _get_loader().get_sample_personas(3)


# ---------------------------------------------------------------------------
# GET /reviews/live-demo
# ---------------------------------------------------------------------------

@router.get("/live-demo")
def live_demo() -> dict:
    # 1. Load real users
    try:
        loader = GoodreadsLoader()
        df = loader.load(n_samples=2000)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Dataset unavailable. Check internet connection.")

    personas = loader.build_user_personas(df)
    demo_user_ids = loader.get_sample_users_for_demo(n=3)

    if not demo_user_ids:
        raise HTTPException(status_code=503, detail="No qualifying users found in dataset.")

    # 2. For each demo user, find a real Amazon book and generate a review
    results = []
    for user_id in demo_user_ids:
        persona = personas[user_id]
        preferred_genres = loader.get_preferred_genres(user_id)
        reviewed_titles = loader.get_reviewed_titles(user_id)

        try:
            unseen_book = book_searcher.find_book_for_user(
                preferred_genres=preferred_genres,
                reviewed_titles=reviewed_titles,
            )
        except RuntimeError as exc:
            results.append({
                "user_id": user_id,
                "error": str(exc),
                "status": "skipped",
            })
            continue

        ground_truth = book_searcher.get_real_ground_truth(unseen_book)

        context = {
            "platform": "Amazon Books",
            "book_url": unseen_book.metadata.get("amazon_url"),
            "real_review_count": unseen_book.metadata.get("review_count"),
            "ground_truth_rating": ground_truth,
        }

        request = ReviewRequest(persona=persona, product=unseen_book, context=context)
        output = generator.generate_review(request)

        rating_delta = (
            round(abs(output.star_rating - ground_truth), 2)
            if ground_truth is not None
            else None
        )

        history_count = len(persona.interaction_history)
        avg_hist_rating = round(
            sum(i.rating_given for i in persona.interaction_history) / max(history_count, 1), 2
        )

        results.append({
            "user_id": user_id,
            "status": "success",
            "user_summary": {
                "age": persona.age,
                "occupation": persona.occupation,
                "personality": persona.personality.model_dump(),
                "books_reviewed_historically": history_count,
                "preferred_genres": preferred_genres,
                "historical_avg_rating": avg_hist_rating,
                "personality_summary": build_personality_summary(persona),
            },
            "amazon_book": {
                "title": unseen_book.title,
                "genre": unseen_book.category,
                "description": unseen_book.description,
                "amazon_url": unseen_book.metadata.get("amazon_url"),
                "amazon_avg_rating": ground_truth,
                "amazon_review_count": unseen_book.metadata.get("review_count"),
                "asin": unseen_book.metadata.get("asin"),
                "data_source": "Amazon.com via Tavily search",
            },
            "simulated_review": {
                "review_text": output.review_text,
                "star_rating": output.star_rating,
                "tone": output.tone,
                "reasoning_trace": output.reasoning_trace,
                "mindset_update": output.mindset_update,
                "confidence": output.confidence,
            },
            "evaluation": {
                "ground_truth_amazon_rating": ground_truth,
                "simulated_rating": output.star_rating,
                "absolute_error": rating_delta,
                "within_half_star": (
                    rating_delta <= 0.5 if rating_delta is not None else None
                ),
                "within_one_star": (
                    rating_delta <= 1.0 if rating_delta is not None else None
                ),
                "note": (
                    "Ground truth is Amazon's actual average customer rating from real buyers"
                    if ground_truth is not None
                    else "Amazon rating not found on this page — delta comparison unavailable"
                ),
            },
        })

    # 3. Aggregate evaluation
    successful = [r for r in results if r.get("status") == "success"]
    deltas = [
        r["evaluation"]["absolute_error"]
        for r in successful
        if r["evaluation"]["absolute_error"] is not None
    ]

    aggregate: dict = {
        "users_attempted": len(demo_user_ids),
        "users_succeeded": len(successful),
        "users_with_ground_truth": len(deltas),
    }
    if deltas:
        aggregate.update({
            "mean_absolute_error": round(sum(deltas) / len(deltas), 3),
            "rmse": round(float(np.sqrt(np.mean([d ** 2 for d in deltas]))), 3),
            "within_half_star_pct": round(
                sum(1 for d in deltas if d <= 0.5) / len(deltas) * 100, 1
            ),
            "within_one_star_pct": round(
                sum(1 for d in deltas if d <= 1.0) / len(deltas) * 100, 1
            ),
        })

    return {
        "pipeline": {
            "description": "Real Amazon users → Tavily book search → LLM review simulation → accuracy eval",
            "dataset": "Amazon Reviews 2023 (McAuley-Lab)",
            "book_source": "Amazon.com via Tavily",
            "llm": settings.MAIN_MODEL,
            "ground_truth": "Amazon customer avg rating",
        },
        "aggregate_evaluation": aggregate,
        "user_results": results,
    }
