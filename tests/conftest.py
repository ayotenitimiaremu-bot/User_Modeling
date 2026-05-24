import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (calls real APIs or network)"
    )


from models.schemas import (
    PastInteraction,
    PersonalityTraits,
    ProductDetails,
    ReviewRequest,
    UserPersona,
)


@pytest.fixture
def personality():
    return PersonalityTraits(
        openness=2, conscientiousness=3,
        extraversion=2, agreeableness=3, neuroticism=1,
    )


@pytest.fixture
def past_interaction():
    return PastInteraction(
        item_id="book_001",
        item_title="Atomic Habits",
        item_category="self-help",
        rating_given=4.5,
        review_text="Very practical book with actionable advice that I found easy to apply daily.",
        timestamp="2024-01-15",
    )


@pytest.fixture
def persona(personality):
    rows = [
        ("self-help", 4.5, "Very practical and well written. Clear advice throughout the whole book."),
        ("self-help", 4.0, "Good read. Solid takeaways worth implementing right away."),
        ("fiction",   3.5, "Decent story. Some slow parts but overall enjoyable enough for the weekend."),
        ("history",   4.0, "Fascinating perspective. Well researched and surprisingly accessible."),
        ("fiction",   3.0, "Average. Had real potential but did not fully deliver in the end unfortunately."),
    ]
    interactions = [
        PastInteraction(
            item_id=f"book_{i:03d}",
            item_title=f"Book {i}",
            item_category=cat,
            rating_given=rating,
            review_text=text,
            timestamp=f"2024-0{i}-01",
        )
        for i, (cat, rating, text) in enumerate(rows, 1)
    ]
    return UserPersona(
        user_id="test_user_001",
        age=28,
        occupation="banker",
        personality=personality,
        interaction_history=interactions,
    )


@pytest.fixture
def product():
    return ProductDetails(
        product_id="shoe_001",
        title="Clarks Desert Boot",
        category="shoes",
        description="Classic suede ankle boot with crepe sole",
        metadata={"brand": "Clarks", "price": 85000, "currency": "NGN"},
    )


@pytest.fixture
def review_request(persona, product):
    return ReviewRequest(persona=persona, product=product)


@pytest.fixture
def sample_persona():
    personality = PersonalityTraits(
        openness=2, conscientiousness=2, extraversion=2,
        agreeableness=3, neuroticism=2,
    )
    rows = [
        ("self-help", 3.5, "Useful concepts but could have been shorter. Good practical tips throughout."),
        ("fiction",   4.0, "Enjoyed this more than expected. Well paced and engaging all the way through."),
        ("history",   3.0, "Decent overview. Nothing groundbreaking but informative enough for the topic."),
        ("self-help", 4.0, "Solid advice. Took notes and have already applied a few ideas at work."),
        ("mystery",   2.5, "Started well but lost me halfway through. The ending was a real disappointment."),
    ]
    interactions = [
        PastInteraction(
            item_id=f"book_{i:03d}",
            item_title=f"Book {i}",
            item_category=cat,
            rating_given=rating,
            review_text=text,
            timestamp=f"2024-0{i}-15",
        )
        for i, (cat, rating, text) in enumerate(rows, 1)
    ]
    return UserPersona(
        user_id="sample_user_accountant",
        age=30,
        occupation="accountant",
        personality=personality,
        interaction_history=interactions,
    )


@pytest.fixture
def sample_product():
    return ProductDetails(
        product_id="deep_work_001",
        title="Deep Work",
        category="self-help",
        description="Rules for focused success in a distracted world.",
        metadata={"author": "Cal Newport", "pages": 296},
    )


@pytest.fixture
def sample_request(sample_persona, sample_product):
    return ReviewRequest(persona=sample_persona, product=sample_product)
