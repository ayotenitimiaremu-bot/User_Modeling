from fastapi import FastAPI

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

app.include_router(router, prefix="/reviews")


@app.get("/")
def root() -> dict:
    return {
        "message": "Task A — User Review Modeling API",
        "docs": "/docs",
        "sample_request": "/reviews/sample-request",
        "sample_personas": "/reviews/sample-personas",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.MAIN_MODEL,
        "dataset": "Amazon Reviews 2023",
    }
