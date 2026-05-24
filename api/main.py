from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings

app = FastAPI(
    title="Task A — User Review Modeling API",
    description="""
Generates simulated user reviews and ratings based on user persona and product details.

## How to Get Started

**New to this API?** Hit `GET /reviews/sample-request` first.
It returns a complete working request you can copy and modify.

**Have your own users?** Build a `UserPersona` from your user data
and `POST` to `/reviews/generate`.

**Testing?** Hit `GET /reviews/sample-personas` to see
pre-built personas from the Amazon Reviews 2023 dataset.
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/reviews")


@app.get("/")
def root() -> dict:
    return {
        "message": "Task A — User Review Modeling API",
        "docs": "/docs",
        "sample_request": "/reviews/sample-request",
        "sample_personas": "/reviews/sample-personas",
    }


@app.post("/analyze-product")
async def analyze_product(request: dict) -> dict:
    return {
        "safe_options": [
            {
                "platform": "Jumia",
                "price": 285000,
                "seller": "Jumia Official Store",
                "trust_score": 95,
                "location": "Lagos",
                "delivery": "2-3 days",
                "url": "https://jumia.com.ng"
            }
        ],
        "scam_warnings": [
            {
                "platform": "Jiji",
                "price": 120000,
                "seller": "QuickDeals NG",
                "trust_score": 24,
                "warnings": [
                    "Account created 3 days ago",
                    "Price 58% below market average",
                    "1 scam report on Nairaland"
                ],
                "location": "Unverified"
            }
        ],
        "savings": 45000
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.MAIN_MODEL,
        "dataset": "Amazon Reviews 2023",
    }
