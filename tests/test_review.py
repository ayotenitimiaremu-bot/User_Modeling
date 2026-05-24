import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import PromptTemplate

from config.prompts import (
    MINDSET_UPDATE_PROMPT,
    PERSONA_SUMMARY_PROMPT,
    REASONING_PROMPT,
    REVIEW_GENERATION_PROMPT,
)
from core.rating_engine import calibrate_rating, compute_importance_score
from core.review_generator import ReviewGenerator
from data.amazon_reviews_2023 import GoodreadsLoader
from models.schemas import PastInteraction, ReviewOutput, UserPersona


# ---------------------------------------------------------------------------
# Sample inputs for prompt rendering tests
# ---------------------------------------------------------------------------

_PERSONA_INPUTS = {
    "age": 28, "occupation": "banker",
    "openness": 2, "conscientiousness": 3,
    "extraversion": 2, "agreeableness": 2, "neuroticism": 1,
    "interaction_history_text": "- Atomic Habits (self-help): 4.5/5 — 'Very practical...'",
}

_REASONING_INPUTS = {
    "persona_summary": "A detail-oriented 28-year-old banker who prefers practical books.",
    "product_title": "Atomic Habits",
    "product_category": "self-help",
    "product_description": "A guide to building habits.",
    "product_metadata": "{'author': 'James Clear', 'pages': 320}",
    "relevant_history": "- Atomic Habits: 4.5/5 — 'Very practical'",
    "context": "{'time': 'unspecified', 'platform': 'general'}",
}

_REVIEW_GEN_INPUTS = {
    "persona_summary": "A 28-year-old banker.",
    "reasoning_json": '{"predicted_rating": 4.0, "reasoning_summary": "Good fit"}',
    "product_title": "Atomic Habits",
    "product_category": "self-help",
    "avg_review_length": 50,
    "conscientiousness": 3,
}

_MINDSET_INPUTS = {
    "current_preferences": "{}",
    "product_title": "Atomic Habits",
    "product_category": "self-help",
    "star_rating": 4.0,
    "tone": "positive",
}


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestPromptRendering:
    def test_persona_summary_renders_without_error(self):
        text = PERSONA_SUMMARY_PROMPT.format(**_PERSONA_INPUTS)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_persona_summary_contains_key_values(self):
        text = PERSONA_SUMMARY_PROMPT.format(**_PERSONA_INPUTS)
        assert "28" in text
        assert "banker" in text
        assert "Atomic Habits" in text

    def test_reasoning_renders_without_error(self):
        text = REASONING_PROMPT.format(**_REASONING_INPUTS)
        assert isinstance(text, str)

    def test_reasoning_contains_product_info(self):
        text = REASONING_PROMPT.format(**_REASONING_INPUTS)
        assert "Atomic Habits" in text
        assert "self-help" in text
        assert "James Clear" in text

    def test_reasoning_contains_json_schema(self):
        text = REASONING_PROMPT.format(**_REASONING_INPUTS)
        assert "gut_reaction" in text
        assert "predicted_rating" in text
        assert "reasoning_summary" in text
        assert "{" in text and "}" in text

    def test_review_generation_renders_without_error(self):
        text = REVIEW_GENERATION_PROMPT.format(**_REVIEW_GEN_INPUTS)
        assert isinstance(text, str)

    def test_review_generation_contains_word_count(self):
        text = REVIEW_GENERATION_PROMPT.format(**_REVIEW_GEN_INPUTS)
        assert "50" in text

    def test_review_generation_contains_json_schema(self):
        text = REVIEW_GENERATION_PROMPT.format(**_REVIEW_GEN_INPUTS)
        assert "review_text" in text
        assert "star_rating" in text
        assert "confidence" in text

    def test_mindset_update_renders_without_error(self):
        text = MINDSET_UPDATE_PROMPT.format(**_MINDSET_INPUTS)
        assert isinstance(text, str)

    def test_mindset_update_contains_product_and_rating(self):
        text = MINDSET_UPDATE_PROMPT.format(**_MINDSET_INPUTS)
        assert "Atomic Habits" in text
        assert "4.0" in text
        assert "positive" in text

    def test_mindset_update_contains_json_schema(self):
        text = MINDSET_UPDATE_PROMPT.format(**_MINDSET_INPUTS)
        assert "mindset_update" in text
        assert "updated_preferences" in text

    def test_all_prompts_have_input_variables(self):
        assert len(PERSONA_SUMMARY_PROMPT.input_variables) > 0
        assert len(REASONING_PROMPT.input_variables) > 0
        assert len(REVIEW_GENERATION_PROMPT.input_variables) > 0
        assert len(MINDSET_UPDATE_PROMPT.input_variables) > 0

    def test_persona_summary_input_variables(self):
        expected = {"age", "occupation", "openness", "conscientiousness",
                    "extraversion", "agreeableness", "neuroticism",
                    "interaction_history_text"}
        assert expected.issubset(set(PERSONA_SUMMARY_PROMPT.input_variables))


# ---------------------------------------------------------------------------
# ReviewGenerator._try_parse_json
# ---------------------------------------------------------------------------

class TestTryParseJson:
    def test_valid_json_object(self):
        result = ReviewGenerator._try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_embedded_in_prose(self):
        result = ReviewGenerator._try_parse_json(
            'Sure, here is the result:\n{"key": "value"}\nHope that helps.'
        )
        assert result == {"key": "value"}

    def test_nested_json(self):
        result = ReviewGenerator._try_parse_json('{"outer": {"inner": 42}}')
        assert result["outer"]["inner"] == 42

    def test_json_with_numbers(self):
        result = ReviewGenerator._try_parse_json('{"rating": 4.5, "count": 3}')
        assert result["rating"] == 4.5
        assert result["count"] == 3

    def test_pure_invalid_text_returns_none(self):
        assert ReviewGenerator._try_parse_json("not json at all") is None

    def test_empty_string_returns_none(self):
        assert ReviewGenerator._try_parse_json("") is None

    def test_partial_json_returns_none(self):
        assert ReviewGenerator._try_parse_json('{"key": ') is None

    def test_json_array_not_matched_as_dict(self):
        result = ReviewGenerator._try_parse_json('[1, 2, 3]')
        assert result is None


# ---------------------------------------------------------------------------
# ReviewGenerator._safe_json_invoke  (uses FakeListChatModel)
# ---------------------------------------------------------------------------

class TestSafeJsonInvoke:
    _simple = PromptTemplate(input_variables=["text"], template="{text}")

    def _gen(self, responses):
        g = ReviewGenerator.__new__(ReviewGenerator)
        g.main_llm = FakeListChatModel(responses=responses)
        g.fast_llm = FakeListChatModel(responses=responses)
        return g

    def test_returns_dict_on_clean_json(self):
        g = self._gen(['{"result": "ok", "score": 0.9}'])
        result = g._safe_json_invoke(self._simple, g.main_llm, {"text": "go"})
        assert result == {"result": "ok", "score": 0.9}

    def test_succeeds_when_json_embedded_in_text(self):
        g = self._gen(['Here you go: {"key": "value"} enjoy.'])
        result = g._safe_json_invoke(self._simple, g.main_llm, {"text": "go"})
        assert result == {"key": "value"}

    def test_retries_and_succeeds_on_second_attempt(self):
        g = self._gen(["totally not json", '{"key": "recovered"}'])
        result = g._safe_json_invoke(self._simple, g.main_llm, {"text": "go"})
        assert result == {"key": "recovered"}

    def test_raises_after_two_consecutive_failures(self):
        g = self._gen(["bad response one", "bad response two"])
        with pytest.raises(ValueError, match="JSON parsing failed"):
            g._safe_json_invoke(self._simple, g.main_llm, {"text": "go"})

    def test_strict_instruction_appended_on_retry(self):
        g = self._gen(["not json", '{"ok": true}'])
        result = g._safe_json_invoke(self._simple, g.main_llm, {"text": "original"})
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# ReviewGenerator.generate_review  (full pipeline with FakeListChatModel)
# ---------------------------------------------------------------------------

_REASONING = json.dumps({
    "gut_reaction": "Practical and useful",
    "will_notice": "Actionable steps",
    "disappointments": "Nothing major",
    "impressions": "Well structured",
    "comparison": "Better than similar books",
    "predicted_rating": 4.0,
    "reasoning_summary": "Good fit for a detail-oriented banker",
})
_REVIEW = json.dumps({
    "review_text": "Really enjoyed this. Clear advice I could apply the same week.",
    "star_rating": 4.0,
    "tone": "positive",
    "confidence": 0.85,
})
_MINDSET = json.dumps({
    "mindset_update": "User is now more open to self-help and productivity content.",
    "updated_preferences": "Prefers practical non-fiction with actionable takeaways",
})


def _make_generator():
    g = ReviewGenerator.__new__(ReviewGenerator)
    g.fast_llm = FakeListChatModel(responses=[
        "A detail-oriented 28-year-old banker who favours practical reads.",
        _MINDSET,
    ])
    g.main_llm = FakeListChatModel(responses=[_REASONING, _REVIEW])
    return g


class TestGenerateReview:
    def test_returns_review_output_type(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert isinstance(result, ReviewOutput)

    def test_star_rating_is_half_star_increment(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert (result.star_rating * 2) % 1 == pytest.approx(0.0)

    def test_star_rating_in_valid_bounds(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert 1.0 <= result.star_rating <= 5.0

    def test_tone_is_valid_literal(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert result.tone in {"positive", "negative", "neutral", "mixed"}

    def test_confidence_in_bounds(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert 0.0 <= result.confidence <= 1.0

    def test_review_text_is_non_empty_string(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert isinstance(result.review_text, str)
        assert len(result.review_text) > 0

    def test_reasoning_trace_is_string(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert isinstance(result.reasoning_trace, str)

    def test_mindset_update_is_string(self, review_request):
        result = _make_generator().generate_review(review_request)
        assert isinstance(result.mindset_update, str)
        assert len(result.mindset_update) > 0

    def test_invalid_tone_normalised_to_neutral(self, review_request):
        g = ReviewGenerator.__new__(ReviewGenerator)
        g.fast_llm = FakeListChatModel(responses=["Persona summary.", _MINDSET])
        g.main_llm = FakeListChatModel(responses=[
            _REASONING,
            json.dumps({"review_text": "OK", "star_rating": 3.0,
                        "tone": "SUPERB", "confidence": 0.7}),
        ])
        result = g.generate_review(review_request)
        assert result.tone == "neutral"

    def test_confidence_clamped_when_above_1(self, review_request):
        g = ReviewGenerator.__new__(ReviewGenerator)
        g.fast_llm = FakeListChatModel(responses=["Persona summary.", _MINDSET])
        g.main_llm = FakeListChatModel(responses=[
            _REASONING,
            json.dumps({"review_text": "Great!", "star_rating": 4.0,
                        "tone": "positive", "confidence": 99.0}),
        ])
        result = g.generate_review(review_request)
        assert result.confidence <= 1.0

    def test_confidence_clamped_when_below_0(self, review_request):
        g = ReviewGenerator.__new__(ReviewGenerator)
        g.fast_llm = FakeListChatModel(responses=["Persona summary.", _MINDSET])
        g.main_llm = FakeListChatModel(responses=[
            _REASONING,
            json.dumps({"review_text": "Bad.", "star_rating": 2.0,
                        "tone": "negative", "confidence": -5.0}),
        ])
        result = g.generate_review(review_request)
        assert result.confidence >= 0.0

    def test_empty_history_persona_does_not_crash(self, personality, product):
        from models.schemas import ReviewRequest, UserPersona
        empty_persona = UserPersona(user_id="empty", age=25, occupation="student",
                                    personality=personality)
        request = ReviewRequest(persona=empty_persona, product=product)
        result = _make_generator().generate_review(request)
        assert isinstance(result, ReviewOutput)


# ---------------------------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------------------------

def test_schemas_validate(sample_persona):
    assert isinstance(sample_persona, UserPersona)
    p = sample_persona.personality
    for attr in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        assert 1 <= getattr(p, attr) <= 3
    for interaction in sample_persona.interaction_history:
        assert 1.0 <= interaction.rating_given <= 5.0


def test_rating_calibration():
    result = calibrate_rating(5.0, 3.5, 0.8, never_gives_5=True)
    assert result <= 4.5

    result = calibrate_rating(4.5, 2.0, 0.5, harsh_rater=True)
    assert result < 4.5

    result = calibrate_rating(6.0, 3.0, 0.5)
    assert result <= 5.0


def test_importance_scoring():
    interaction = PastInteraction(
        item_id="b1", item_title="Nike Air Max",
        item_category="shoes", rating_given=4.0,
        review_text="Great fit and comfortable for all-day wear. Very stylish design.",
    )
    score = compute_importance_score(interaction, "shoes")
    assert score > 0.4

    score_mismatch = compute_importance_score(interaction, "books")
    assert score_mismatch < score


@pytest.mark.integration
def test_review_generator_integration(sample_request):
    """Makes a real Groq API call — requires GROQ_API_KEY in .env."""
    generator = ReviewGenerator()
    result = generator.generate_review(sample_request)
    assert isinstance(result, ReviewOutput)
    assert 1.0 <= result.star_rating <= 5.0
    assert result.tone in ["positive", "negative", "neutral", "mixed"]
    assert len(result.review_text) > 20
