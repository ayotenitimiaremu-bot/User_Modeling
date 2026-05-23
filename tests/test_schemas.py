import pytest
from pydantic import ValidationError

from models.schemas import (
    PastInteraction,
    PersonalityTraits,
    ProductDetails,
    ReviewOutput,
    ReviewRequest,
    UserPersona,
)


class TestPersonalityTraits:
    def test_valid(self, personality):
        assert personality.openness == 2
        assert personality.conscientiousness == 3
        assert personality.neuroticism == 1

    def test_boundary_min(self):
        t = PersonalityTraits(openness=1, conscientiousness=1, extraversion=1,
                              agreeableness=1, neuroticism=1)
        assert t.openness == 1

    def test_boundary_max(self):
        t = PersonalityTraits(openness=3, conscientiousness=3, extraversion=3,
                              agreeableness=3, neuroticism=3)
        assert t.neuroticism == 3

    @pytest.mark.parametrize("field,value", [
        ("openness", 0), ("openness", 4),
        ("conscientiousness", 0), ("conscientiousness", 4),
        ("extraversion", 0), ("extraversion", 4),
        ("agreeableness", -1), ("neuroticism", 5),
    ])
    def test_out_of_range_raises(self, field, value):
        data = {f: 2 for f in
                ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")}
        data[field] = value
        with pytest.raises(ValidationError):
            PersonalityTraits(**data)


class TestPastInteraction:
    def test_valid(self, past_interaction):
        assert past_interaction.item_id == "book_001"
        assert past_interaction.rating_given == 4.5
        assert past_interaction.timestamp == "2024-01-15"

    def test_timestamp_is_optional(self):
        i = PastInteraction(
            item_id="b1", item_title="A Book", item_category="fiction",
            rating_given=3.0, review_text="OK read.",
        )
        assert i.timestamp is None

    @pytest.mark.parametrize("rating", [0.9, 5.1, 0.0, 6.0, -1.0])
    def test_rating_out_of_range_raises(self, rating):
        with pytest.raises(ValidationError):
            PastInteraction(item_id="b", item_title="T", item_category="fiction",
                            rating_given=rating, review_text="text")

    @pytest.mark.parametrize("rating", [1.0, 2.5, 3.0, 4.5, 5.0])
    def test_valid_boundary_ratings(self, rating):
        i = PastInteraction(item_id="b", item_title="T", item_category="fiction",
                            rating_given=rating, review_text="text")
        assert i.rating_given == rating


class TestUserPersona:
    def test_valid(self, persona):
        assert persona.user_id == "test_user_001"
        assert persona.age == 28
        assert len(persona.interaction_history) == 5

    def test_defaults_empty_history_and_preferences(self, personality):
        p = UserPersona(user_id="u", age=25, occupation="student", personality=personality)
        assert p.interaction_history == []
        assert p.inferred_preferences == {}

    @pytest.mark.parametrize("age", [12, 91, 0, -1])
    def test_age_out_of_range_raises(self, age, personality):
        with pytest.raises(ValidationError):
            UserPersona(user_id="u", age=age, occupation="teacher", personality=personality)

    @pytest.mark.parametrize("age", [13, 90])
    def test_age_boundary_valid(self, age, personality):
        p = UserPersona(user_id="u", age=age, occupation="teacher", personality=personality)
        assert p.age == age

    def test_inferred_preferences_accepts_dict(self, personality):
        p = UserPersona(user_id="u", age=30, occupation="teacher",
                        personality=personality,
                        inferred_preferences={"genre": "fiction", "length": "short"})
        assert p.inferred_preferences["genre"] == "fiction"


class TestProductDetails:
    def test_valid(self, product):
        assert product.product_id == "shoe_001"
        assert product.title == "Clarks Desert Boot"
        assert product.metadata["brand"] == "Clarks"

    def test_image_url_optional(self):
        p = ProductDetails(product_id="p1", title="Item", category="shoes",
                           description="desc", metadata={})
        assert p.image_url is None

    def test_image_url_set(self):
        p = ProductDetails(product_id="p1", title="Item", category="shoes",
                           description="desc", metadata={},
                           image_url="https://example.com/img.jpg")
        assert p.image_url == "https://example.com/img.jpg"

    def test_metadata_accepts_nested_dict(self):
        p = ProductDetails(product_id="p1", title="T", category="c",
                           description="d",
                           metadata={"specs": {"color": "brown", "size": 42}})
        assert p.metadata["specs"]["color"] == "brown"


class TestReviewRequest:
    def test_valid(self, review_request, persona, product):
        assert review_request.persona.user_id == persona.user_id
        assert review_request.product.product_id == product.product_id

    def test_context_optional(self, review_request):
        assert review_request.context is None

    def test_context_accepts_dict(self, persona, product):
        r = ReviewRequest(persona=persona, product=product,
                          context={"time_of_day": "morning", "platform": "mobile"})
        assert r.context["time_of_day"] == "morning"


class TestReviewOutput:
    @pytest.mark.parametrize("tone", ["positive", "negative", "neutral", "mixed"])
    def test_valid_tones(self, tone):
        r = ReviewOutput(review_text="Good.", star_rating=4.0, tone=tone,
                         reasoning_trace="trace", confidence=0.8,
                         mindset_update="update")
        assert r.tone == tone

    @pytest.mark.parametrize("tone", ["excellent", "POSITIVE", "bad", "ok", ""])
    def test_invalid_tone_raises(self, tone):
        with pytest.raises(ValidationError):
            ReviewOutput(review_text="x", star_rating=3.0, tone=tone,
                         reasoning_trace="t", confidence=0.5, mindset_update="u")

    @pytest.mark.parametrize("rating", [0.9, 5.1, 0.0, 6.0])
    def test_star_rating_out_of_range_raises(self, rating):
        with pytest.raises(ValidationError):
            ReviewOutput(review_text="x", star_rating=rating, tone="neutral",
                         reasoning_trace="t", confidence=0.5, mindset_update="u")

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -1.0])
    def test_confidence_out_of_range_raises(self, confidence):
        with pytest.raises(ValidationError):
            ReviewOutput(review_text="x", star_rating=3.0, tone="neutral",
                         reasoning_trace="t", confidence=confidence, mindset_update="u")

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_boundary_valid(self, confidence):
        r = ReviewOutput(review_text="x", star_rating=3.0, tone="neutral",
                         reasoning_trace="t", confidence=confidence, mindset_update="u")
        assert r.confidence == confidence
