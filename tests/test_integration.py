"""
Integration tests — hit the real Groq API and LangSmith tracing.
Run with:  pytest tests/test_integration.py -v
Skip with: pytest tests/ -v -m "not integration"
"""
import pytest

from core.review_generator import ReviewGenerator
from models.schemas import (
    PastInteraction,
    PersonalityTraits,
    ProductDetails,
    ReviewOutput,
    ReviewRequest,
    UserPersona,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Module-scoped fixtures — one ReviewGenerator instance, one API call for
# the standard pipeline tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generator():
    return ReviewGenerator()


@pytest.fixture(scope="module")
def base_persona():
    personality = PersonalityTraits(
        openness=2, conscientiousness=3, extraversion=2, agreeableness=3, neuroticism=1,
    )
    history = [
        PastInteraction(
            item_id="b1", item_title="Atomic Habits", item_category="self-help",
            rating_given=4.5,
            review_text="Very practical with actionable advice I could apply the same week.",
            timestamp="2024-01-10",
        ),
        PastInteraction(
            item_id="b2", item_title="Deep Work", item_category="self-help",
            rating_given=4.0,
            review_text="Solid read. Changed how I structure my mornings for the better.",
            timestamp="2024-02-05",
        ),
        PastInteraction(
            item_id="b3", item_title="1984", item_category="fiction",
            rating_given=3.5,
            review_text="Classic but dense. Worth reading once if you have the patience.",
            timestamp="2024-03-01",
        ),
    ]
    return UserPersona(
        user_id="integration_user_001",
        age=28,
        occupation="software engineer",
        personality=personality,
        interaction_history=history,
    )


@pytest.fixture(scope="module")
def self_help_product():
    return ProductDetails(
        product_id="book_habit",
        title="The Power of Habit",
        category="self-help",
        description="Why we do what we do in life and business, and how to change.",
        metadata={"author": "Charles Duhigg", "pages": 371},
    )


@pytest.fixture(scope="module")
def review_result(generator, base_persona, self_help_product):
    """One real API call shared across all TestFullPipeline tests."""
    return generator.generate_review(ReviewRequest(persona=base_persona, product=self_help_product))


# ---------------------------------------------------------------------------
# TestFullPipeline — verifies the 6-step pipeline produces a valid output
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_returns_review_output_type(self, review_result):
        assert isinstance(review_result, ReviewOutput)

    def test_review_text_is_non_empty(self, review_result):
        assert isinstance(review_result.review_text, str)
        assert len(review_result.review_text.strip()) > 20

    def test_star_rating_in_bounds(self, review_result):
        assert 1.0 <= review_result.star_rating <= 5.0

    def test_star_rating_is_half_star_increment(self, review_result):
        assert (review_result.star_rating * 2) % 1 == pytest.approx(0.0)

    def test_tone_is_valid(self, review_result):
        assert review_result.tone in {"positive", "negative", "neutral", "mixed"}

    def test_confidence_in_bounds(self, review_result):
        assert 0.0 <= review_result.confidence <= 1.0

    def test_reasoning_trace_is_non_empty(self, review_result):
        assert len(review_result.reasoning_trace.strip()) > 0

    def test_mindset_update_is_non_empty(self, review_result):
        assert len(review_result.mindset_update.strip()) > 0


# ---------------------------------------------------------------------------
# TestPersonaBehavior — one call each, tests different persona edge cases
# ---------------------------------------------------------------------------

class TestPersonaBehavior:
    @staticmethod
    def _persona(ratings, uid):
        history = [
            PastInteraction(
                item_id=f"b{i}", item_title=f"Book {i}", item_category="self-help",
                rating_given=r, review_text="Decent read, had some useful ideas.",
            )
            for i, r in enumerate(ratings)
        ]
        return UserPersona(
            user_id=uid,
            age=30,
            occupation="teacher",
            personality=PersonalityTraits(
                openness=2, conscientiousness=2, extraversion=2,
                agreeableness=2, neuroticism=2,
            ),
            interaction_history=history,
        )

    @staticmethod
    def _product(uid="mindset"):
        return ProductDetails(
            product_id=f"p_{uid}",
            title="Mindset",
            category="self-help",
            description="The new psychology of success by Carol Dweck.",
            metadata={"author": "Carol Dweck", "pages": 288},
        )

    def test_harsh_rater_produces_valid_output(self, generator):
        request = ReviewRequest(
            persona=self._persona([1.5, 2.0, 1.5, 2.0, 2.5], "harsh_user"),
            product=self._product(),
        )
        result = generator.generate_review(request)
        assert isinstance(result, ReviewOutput)
        assert 1.0 <= result.star_rating <= 5.0

    def test_generous_rater_produces_valid_output(self, generator):
        request = ReviewRequest(
            persona=self._persona([4.5, 5.0, 4.5, 4.0, 5.0], "generous_user"),
            product=self._product(),
        )
        result = generator.generate_review(request)
        assert isinstance(result, ReviewOutput)
        assert 1.0 <= result.star_rating <= 5.0

    def test_empty_history_does_not_crash(self, generator):
        persona = UserPersona(
            user_id="no_history_user",
            age=22,
            occupation="student",
            personality=PersonalityTraits(
                openness=2, conscientiousness=2, extraversion=2,
                agreeableness=2, neuroticism=2,
            ),
        )
        product = ProductDetails(
            product_id="p_sapiens",
            title="Sapiens",
            category="history",
            description="A brief history of humankind.",
            metadata={"author": "Yuval Noah Harari", "pages": 443},
        )
        result = generator.generate_review(ReviewRequest(persona=persona, product=product))
        assert isinstance(result, ReviewOutput)
        assert len(result.review_text.strip()) > 0
