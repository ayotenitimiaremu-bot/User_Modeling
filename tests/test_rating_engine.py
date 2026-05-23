import pytest

from core.rating_engine import (
    calibrate_rating,
    compute_importance_score,
    compute_rating_prior,
    filter_relevant_memory,
)
from models.schemas import PastInteraction, PersonalityTraits, UserPersona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _personality():
    return PersonalityTraits(openness=2, conscientiousness=2, extraversion=2,
                             agreeableness=2, neuroticism=2)


def _make_interaction(item_id="b1", category="fiction", rating=3.0,
                      review_text="word " * 10):
    return PastInteraction(
        item_id=item_id, item_title=f"Title {item_id}",
        item_category=category, rating_given=rating,
        review_text=review_text.strip(),
    )


def _make_persona(ratings, categories=None, word_counts=None):
    categories = categories or ["fiction"] * len(ratings)
    word_counts = word_counts or [10] * len(ratings)
    interactions = [
        _make_interaction(
            item_id=f"b{i}",
            category=cat,
            rating=r,
            review_text=" ".join(["word"] * wc),
        )
        for i, (r, cat, wc) in enumerate(zip(ratings, categories, word_counts))
    ]
    return UserPersona(user_id="u", age=30, occupation="teacher",
                       personality=_personality(),
                       interaction_history=interactions)


# ---------------------------------------------------------------------------
# compute_rating_prior
# ---------------------------------------------------------------------------

class TestComputeRatingPrior:
    def test_empty_history_returns_defaults(self):
        p = UserPersona(user_id="u", age=25, occupation="teacher",
                        personality=_personality())
        prior = compute_rating_prior(p)
        assert prior["mean_rating"] == 3.0
        assert prior["std_rating"] == 1.0
        assert prior["never_gives_5"] is False
        assert prior["never_gives_1"] is False
        assert prior["harsh_rater"] is False
        assert prior["generous_rater"] is False
        assert prior["category_biases"] == {}

    def test_mean_and_std_computed_correctly(self):
        prior = compute_rating_prior(_make_persona([2.0, 4.0, 3.0]))
        assert prior["mean_rating"] == pytest.approx(3.0, abs=0.01)

    def test_harsh_rater_flag(self):
        prior = compute_rating_prior(_make_persona([1.0, 2.0, 1.5, 2.0, 2.0]))
        assert prior["harsh_rater"] is True
        assert prior["generous_rater"] is False

    def test_generous_rater_flag(self):
        prior = compute_rating_prior(_make_persona([4.5, 5.0, 4.5, 4.0, 4.5]))
        assert prior["generous_rater"] is True
        assert prior["harsh_rater"] is False

    def test_never_gives_5(self):
        prior = compute_rating_prior(_make_persona([1.0, 2.0, 3.0, 4.0, 4.5]))
        assert prior["never_gives_5"] is True

    def test_gives_5_sets_flag_false(self):
        prior = compute_rating_prior(_make_persona([3.0, 5.0]))
        assert prior["never_gives_5"] is False

    def test_never_gives_1(self):
        prior = compute_rating_prior(_make_persona([2.0, 3.0, 4.0, 5.0]))
        assert prior["never_gives_1"] is True

    def test_gives_1_sets_flag_false(self):
        prior = compute_rating_prior(_make_persona([1.0, 4.0]))
        assert prior["never_gives_1"] is False

    def test_category_biases_computed(self):
        prior = compute_rating_prior(_make_persona(
            [4.0, 4.0, 2.0, 2.0],
            categories=["fiction", "fiction", "mystery", "mystery"],
        ))
        assert prior["category_biases"]["fiction"] == pytest.approx(4.0)
        assert prior["category_biases"]["mystery"] == pytest.approx(2.0)

    def test_single_category_in_biases(self):
        prior = compute_rating_prior(_make_persona([3.0, 4.0, 5.0]))
        assert "fiction" in prior["category_biases"]

    def test_rating_distribution_has_five_buckets(self):
        prior = compute_rating_prior(_make_persona([1.0, 3.0, 5.0]))
        assert set(prior["rating_distribution"].keys()) == {"1", "2", "3", "4", "5"}

    def test_rating_distribution_sums_to_one(self):
        prior = compute_rating_prior(_make_persona([1.0, 2.0, 3.0, 4.0, 5.0]))
        total = sum(prior["rating_distribution"].values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_rating_distribution_correct_percentages(self):
        prior = compute_rating_prior(_make_persona([1.0, 1.0, 3.0, 5.0, 5.0]))
        dist = prior["rating_distribution"]
        assert dist["1"] == pytest.approx(0.4, abs=0.001)
        assert dist["5"] == pytest.approx(0.4, abs=0.001)
        assert dist["2"] == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# calibrate_rating
# ---------------------------------------------------------------------------

class TestCalibrateRating:
    def test_never_gives_5_caps_at_4_5(self):
        result = calibrate_rating(5.0, 4.0, 0.3, never_gives_5=True)
        assert result <= 4.5

    def test_never_gives_1_floors_at_1_5(self):
        result = calibrate_rating(1.0, 2.0, 0.3, never_gives_1=True)
        assert result >= 1.5

    def test_regression_to_mean_pulls_low_rating_up(self):
        # raw=1, hist_avg=4 → blended should be above 1
        result = calibrate_rating(1.0, 4.0, 0.5)
        assert result > 1.0

    def test_regression_to_mean_pulls_high_rating_down(self):
        # raw=5, hist_avg=2 → blended should be below 5
        result = calibrate_rating(5.0, 2.0, 0.5)
        assert result < 5.0

    def test_harsh_rater_reduces_high_scores(self):
        normal = calibrate_rating(4.5, 3.0, 0.5, harsh_rater=False)
        harsh = calibrate_rating(4.5, 3.0, 0.5, harsh_rater=True)
        assert harsh <= normal

    def test_harsh_rater_does_not_affect_low_scores(self):
        normal = calibrate_rating(2.0, 2.0, 0.3, harsh_rater=False)
        harsh = calibrate_rating(2.0, 2.0, 0.3, harsh_rater=True)
        assert normal == harsh

    def test_generous_rater_raises_very_low_scores(self):
        normal = calibrate_rating(1.5, 3.0, 0.5, generous_rater=False)
        generous = calibrate_rating(1.5, 3.0, 0.5, generous_rater=True)
        assert generous >= normal

    def test_generous_rater_does_not_affect_high_scores(self):
        normal = calibrate_rating(4.0, 4.0, 0.3, generous_rater=False)
        generous = calibrate_rating(4.0, 4.0, 0.3, generous_rater=True)
        assert normal == generous

    @pytest.mark.parametrize("raw", [-10.0, 0.0, 1.0, 3.0, 5.0, 7.0, 20.0])
    def test_output_always_clipped_to_1_5(self, raw):
        result = calibrate_rating(raw, 3.0, 0.5)
        assert 1.0 <= result <= 5.0

    @pytest.mark.parametrize("raw", [1.0, 1.7, 2.3, 2.9, 3.6, 4.1, 4.9, 5.0])
    def test_output_is_half_star_increment(self, raw):
        result = calibrate_rating(raw, 3.0, 0.5)
        assert (result * 2) % 1 == pytest.approx(0.0), \
            f"calibrate_rating({raw}) = {result} is not a 0.5 increment"

    def test_stable_input_unchanged_direction(self):
        # When raw equals historical avg, result should be close to it
        result = calibrate_rating(3.0, 3.0, 0.5)
        assert result == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# compute_importance_score
# ---------------------------------------------------------------------------

class TestComputeImportanceScore:
    def test_category_match_adds_to_score(self):
        i = _make_interaction(category="self-help")
        match = compute_importance_score(i, "self-help")
        no_match = compute_importance_score(i, "fantasy")
        assert match - no_match == pytest.approx(0.4, abs=0.001)

    def test_score_bounded_between_0_and_1(self):
        for cat in ["self-help", "fiction", "mystery", "zzzunknown"]:
            i = _make_interaction(category="self-help")
            s = compute_importance_score(i, cat)
            assert 0.0 <= s <= 1.0

    def test_longer_review_gives_higher_score(self):
        short = _make_interaction(review_text="ok")
        long = _make_interaction(review_text=" ".join(["word"] * 200))
        assert compute_importance_score(long, "fiction") >= \
               compute_importance_score(short, "fiction")

    def test_engagement_caps_at_100_words(self):
        at_100 = _make_interaction(review_text=" ".join(["w"] * 100))
        at_500 = _make_interaction(review_text=" ".join(["w"] * 500))
        assert compute_importance_score(at_100, "fiction") == \
               compute_importance_score(at_500, "fiction")

    def test_base_score_always_present(self):
        i = _make_interaction(category="fiction", review_text="word")
        # min score = 0.3 (base) + 0 (no cat match) + tiny engagement
        score = compute_importance_score(i, "mystery")
        assert score >= 0.3


# ---------------------------------------------------------------------------
# filter_relevant_memory
# ---------------------------------------------------------------------------

class TestFilterRelevantMemory:
    def test_empty_list_returns_empty(self):
        assert filter_relevant_memory([], "fiction") == []

    def test_category_matches_included(self):
        interactions = [
            _make_interaction("b1", "fiction", 4.0),
            _make_interaction("b2", "mystery", 3.0),
            _make_interaction("b3", "fiction", 5.0),
        ]
        results = filter_relevant_memory(interactions, "fiction", threshold=0.3)
        result_ids = {r.item_id for r in results}
        assert "b1" in result_ids
        assert "b3" in result_ids

    def test_respects_max_items(self):
        interactions = [_make_interaction(f"b{i}", "fiction", 3.0) for i in range(10)]
        results = filter_relevant_memory(interactions, "fiction", max_items=3)
        assert len(results) <= 3

    def test_threshold_excludes_low_scoring_items(self):
        # Short review + no category match → score ~0.3 (base only)
        interactions = [_make_interaction("b1", "mystery", 3.0, review_text="ok")]
        results = filter_relevant_memory(interactions, "fiction", threshold=0.8)
        assert results == []

    def test_recency_bonus_keeps_recent_items_at_top(self):
        # 6 identical interactions except position; last 3 get +0.2 bonus
        interactions = [
            _make_interaction(f"b{i}", "fiction", 3.0, review_text=" ".join(["w"] * 10))
            for i in range(6)
        ]
        results = filter_relevant_memory(interactions, "fiction", threshold=0.3, max_items=6)
        result_ids = [r.item_id for r in results]
        # Most recent 3 (b3, b4, b5) should appear
        assert "b5" in result_ids
        assert "b4" in result_ids
        assert "b3" in result_ids

    def test_sorted_by_score_descending(self):
        long_review = " ".join(["word"] * 200)
        short_review = "ok"
        interactions = [
            _make_interaction("low", "mystery", 3.0, review_text=short_review),
            _make_interaction("high", "fiction", 4.0, review_text=long_review),
        ]
        results = filter_relevant_memory(interactions, "fiction", threshold=0.3, max_items=2)
        assert results[0].item_id == "high"

    def test_returns_past_interaction_objects(self):
        interactions = [_make_interaction("b1", "fiction", 4.0)]
        results = filter_relevant_memory(interactions, "fiction", threshold=0.0)
        assert isinstance(results[0], PastInteraction)
