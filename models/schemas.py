from typing import Literal, Optional

from pydantic import BaseModel, Field


class PersonalityTraits(BaseModel):
    openness: int = Field(ge=1, le=3)
    conscientiousness: int = Field(ge=1, le=3)
    extraversion: int = Field(ge=1, le=3)
    agreeableness: int = Field(ge=1, le=3)
    neuroticism: int = Field(ge=1, le=3)


class PastInteraction(BaseModel):
    item_id: str
    item_title: str
    item_category: str
    rating_given: float = Field(ge=1.0, le=5.0)
    review_text: str
    timestamp: Optional[str] = None


class UserPersona(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_001",
                "age": 28,
                "occupation": "banker",
                "personality": {
                    "openness": 2,
                    "conscientiousness": 3,
                    "extraversion": 2,
                    "agreeableness": 2,
                    "neuroticism": 1,
                },
                "interaction_history": [
                    {
                        "item_id": "book_001",
                        "item_title": "Atomic Habits",
                        "item_category": "self-help",
                        "rating_given": 4.5,
                        "review_text": "Very practical book with actionable advice",
                        "timestamp": "2024-01-15",
                    }
                ],
                "inferred_preferences": {},
            }
        }
    }

    user_id: str
    age: int = Field(ge=13, le=90)
    occupation: str
    personality: PersonalityTraits
    interaction_history: list[PastInteraction] = []
    inferred_preferences: dict = {}


class ProductDetails(BaseModel):
    product_id: str
    title: str
    category: str
    description: str
    metadata: dict
    image_url: Optional[str] = None


class ReviewRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "persona": {
                    "user_id": "user_001",
                    "age": 28,
                    "occupation": "banker",
                    "personality": {
                        "openness": 2,
                        "conscientiousness": 3,
                        "extraversion": 2,
                        "agreeableness": 2,
                        "neuroticism": 1,
                    },
                    "interaction_history": [
                        {
                            "item_id": "book_001",
                            "item_title": "Atomic Habits",
                            "item_category": "self-help",
                            "rating_given": 4.5,
                            "review_text": "Very practical book with actionable advice",
                            "timestamp": "2024-01-15",
                        }
                    ],
                    "inferred_preferences": {},
                },
                "product": {
                    "product_id": "shoe_001",
                    "title": "Clarks Desert Boot",
                    "category": "shoes",
                    "description": "Classic suede ankle boot with crepe sole",
                    "metadata": {"brand": "Clarks", "price": 85000, "currency": "NGN"},
                },
                "context": None,
            }
        }
    }

    persona: UserPersona
    product: ProductDetails
    context: Optional[dict] = None


class ReviewOutput(BaseModel):
    review_text: str
    star_rating: float = Field(ge=1.0, le=5.0)
    tone: Literal["positive", "negative", "neutral", "mixed"]
    reasoning_trace: str
    confidence: float = Field(ge=0.0, le=1.0)
    mindset_update: str
