from fastapi import APIRouter, HTTPException

from core.review_generator import ReviewGenerator
from data.amazon_reviews_2023 import GoodreadsLoader
from models.schemas import ProductDetails, ReviewOutput, ReviewRequest, UserPersona

router = APIRouter()
generator = ReviewGenerator()

_loader: GoodreadsLoader | None = None


def _get_loader() -> GoodreadsLoader:
    global _loader
    if _loader is None:
        _loader = GoodreadsLoader()
        _loader.df = _loader.generate_synthetic_data(200)
        _loader.build_user_personas(_loader.df)
    return _loader


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
