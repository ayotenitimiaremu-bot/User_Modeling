import pytest
import pandas as pd

from data.amazon_reviews_2023 import GoodreadsLoader
from models.schemas import ProductDetails, UserPersona


# ---------------------------------------------------------------------------
# Module-scoped fixture — generate once, reuse across all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loader_with_data():
    loader = GoodreadsLoader()
    df = loader.generate_synthetic_data(200)
    loader.df = df
    personas = loader.build_user_personas(df)
    return loader, df, personas


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------

class TestGenerateSyntheticData:
    def test_returns_dataframe(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert isinstance(df, pd.DataFrame)

    def test_has_exactly_eight_users(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert df["user_id"].nunique() == 8

    def test_required_columns_present(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        required = {"user_id", "book_id", "rating", "review_text",
                    "review_title", "book_title", "review_date", "genre"}
        assert required.issubset(set(df.columns))

    def test_no_asin_column(self):
        # asin is the real-data detection column; synthetic must NOT have it
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert "asin" not in df.columns

    def test_ratings_in_valid_range(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert df["rating"].between(1.0, 5.0).all()

    def test_ratings_are_half_star_increments(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert ((df["rating"] * 2) % 1 == 0).all()

    def test_no_null_review_text(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert df["review_text"].notna().all()
        assert (df["review_text"].str.len() > 0).all()

    def test_no_null_user_id(self):
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert df["user_id"].notna().all()

    def test_genre_values_are_valid(self):
        from data.amazon_reviews_2023 import GENRES
        df = GoodreadsLoader().generate_synthetic_data(100)
        assert df["genre"].isin(GENRES).all()

    def test_reproducible_across_calls(self):
        l1, l2 = GoodreadsLoader(), GoodreadsLoader()
        df1 = l1.generate_synthetic_data(50)
        df2 = l2.generate_synthetic_data(50)
        assert df1["rating"].tolist() == df2["rating"].tolist()
        assert df1["user_id"].tolist() == df2["user_id"].tolist()

    def test_never_gives_5_user_has_no_5_stars(self):
        df = GoodreadsLoader().generate_synthetic_data(200)
        u8_ratings = df[df["user_id"] == "synth_u008"]["rating"]
        assert len(u8_ratings) > 0
        assert (u8_ratings > 4.5).sum() == 0

    def test_each_user_has_minimum_reviews(self):
        df = GoodreadsLoader().generate_synthetic_data(200)
        counts = df.groupby("user_id").size()
        assert (counts >= 10).all()


# ---------------------------------------------------------------------------
# build_user_personas
# ---------------------------------------------------------------------------

class TestBuildUserPersonas:
    def test_returns_dict(self, loader_with_data):
        _, _, personas = loader_with_data
        assert isinstance(personas, dict)

    def test_all_values_are_user_personas(self, loader_with_data):
        _, _, personas = loader_with_data
        assert all(isinstance(v, UserPersona) for v in personas.values())

    def test_all_8_synthetic_users_present(self, loader_with_data):
        _, _, personas = loader_with_data
        assert len(personas) == 8

    def test_persona_has_interaction_history(self, loader_with_data):
        _, _, personas = loader_with_data
        for p in personas.values():
            assert len(p.interaction_history) >= 3

    def test_persona_age_in_valid_range(self, loader_with_data):
        _, _, personas = loader_with_data
        for p in personas.values():
            assert 20 <= p.age <= 55

    def test_persona_occupation_is_non_empty_string(self, loader_with_data):
        _, _, personas = loader_with_data
        for p in personas.values():
            assert isinstance(p.occupation, str)
            assert len(p.occupation) > 0

    def test_personality_values_in_1_to_3(self, loader_with_data):
        _, _, personas = loader_with_data
        for p in personas.values():
            for attr in ("openness", "conscientiousness", "extraversion",
                         "agreeableness", "neuroticism"):
                val = getattr(p.personality, attr)
                assert 1 <= val <= 3, f"{p.user_id}.{attr} = {val} out of range"

    def test_review_text_truncated_to_300_chars(self, loader_with_data):
        _, _, personas = loader_with_data
        for p in personas.values():
            for interaction in p.interaction_history:
                assert len(interaction.review_text) <= 300

    def test_harsh_critic_has_low_extraversion(self, loader_with_data):
        _, _, personas = loader_with_data
        critic = personas.get("synth_u001")
        assert critic is not None
        # avg_rating=2.3 → pct_high < 40% → extraversion = 1
        assert critic.personality.extraversion == 1

    def test_generous_reader_has_high_extraversion(self, loader_with_data):
        _, _, personas = loader_with_data
        generous = personas.get("synth_u002")
        assert generous is not None
        # avg_rating=4.6 → pct_high > 65% → extraversion = 3
        assert generous.personality.extraversion == 3

    def test_reproducible_age_and_occupation(self):
        loader = GoodreadsLoader()
        df = loader.generate_synthetic_data(100)
        p1 = loader.build_user_personas(df)
        p2 = loader.build_user_personas(df)
        for uid in p1:
            assert p1[uid].age == p2[uid].age
            assert p1[uid].occupation == p2[uid].occupation

    def test_personas_stored_on_instance(self, loader_with_data):
        loader, _, personas = loader_with_data
        assert loader.personas is personas


# ---------------------------------------------------------------------------
# get_persona / get_sample_personas
# ---------------------------------------------------------------------------

class TestGetPersona:
    def test_returns_correct_persona(self, loader_with_data):
        loader, _, _ = loader_with_data
        p = loader.get_persona("synth_u001")
        assert p is not None
        assert p.user_id == "synth_u001"

    def test_unknown_id_returns_none(self, loader_with_data):
        loader, _, _ = loader_with_data
        assert loader.get_persona("ghost_user") is None


class TestGetSamplePersonas:
    def test_returns_correct_count(self, loader_with_data):
        loader, _, _ = loader_with_data
        assert len(loader.get_sample_personas(3)) == 3

    def test_returns_fewer_if_not_enough(self, loader_with_data):
        loader, _, _ = loader_with_data
        samples = loader.get_sample_personas(100)
        assert len(samples) <= 8

    def test_all_items_are_user_personas(self, loader_with_data):
        loader, _, _ = loader_with_data
        for p in loader.get_sample_personas(3):
            assert isinstance(p, UserPersona)


# ---------------------------------------------------------------------------
# get_product_from_record
# ---------------------------------------------------------------------------

class TestGetProductFromRecord:
    def test_returns_product_details(self, loader_with_data):
        loader, df, _ = loader_with_data
        row = df.iloc[0]
        product = loader.get_product_from_record(row)
        assert isinstance(product, ProductDetails)

    def test_product_id_matches_book_id(self, loader_with_data):
        loader, df, _ = loader_with_data
        row = df.iloc[0]
        product = loader.get_product_from_record(row)
        assert product.product_id == str(row["book_id"])

    def test_category_matches_genre(self, loader_with_data):
        loader, df, _ = loader_with_data
        row = df.iloc[0]
        product = loader.get_product_from_record(row)
        assert product.category == str(row["genre"])

    def test_description_contains_genre(self, loader_with_data):
        loader, df, _ = loader_with_data
        row = df.iloc[0]
        product = loader.get_product_from_record(row)
        assert row["genre"] in product.description

    def test_source_in_metadata(self, loader_with_data):
        loader, df, _ = loader_with_data
        product = loader.get_product_from_record(df.iloc[0])
        assert "source" in product.metadata
