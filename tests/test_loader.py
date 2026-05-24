"""
Loader tests — all tests here hit the real Amazon Reviews 2023 dataset
over the network and are therefore marked @pytest.mark.integration.

Run with: pytest tests/test_loader.py -v
Skip with: pytest tests/ -m "not integration"
"""
import pytest

from data.amazon_reviews_2023 import GENRES, GoodreadsLoader
from models.schemas import ProductDetails, UserPersona

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def loader_with_data():
    loader = GoodreadsLoader()
    df = loader.load(300)
    personas = loader.build_user_personas(df)
    return loader, df, personas


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_returns_dataframe(self, loader_with_data):
        _, df, _ = loader_with_data
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, loader_with_data):
        _, df, _ = loader_with_data
        required = {"user_id", "book_id", "asin", "rating", "review_text",
                    "book_title", "review_date", "genre"}
        assert required.issubset(set(df.columns))

    def test_is_real_data(self, loader_with_data):
        _, df, _ = loader_with_data
        assert "asin" in df.columns

    def test_ratings_in_valid_range(self, loader_with_data):
        _, df, _ = loader_with_data
        assert df["rating"].between(1.0, 5.0).all()

    def test_no_null_review_text(self, loader_with_data):
        _, df, _ = loader_with_data
        assert df["review_text"].notna().all()
        assert (df["review_text"].str.len() >= 10).all()

    def test_no_null_user_id(self, loader_with_data):
        _, df, _ = loader_with_data
        assert df["user_id"].notna().all()

    def test_genre_values_are_valid(self, loader_with_data):
        _, df, _ = loader_with_data
        assert df["genre"].isin(GENRES).all()

    def test_each_user_has_minimum_reviews(self, loader_with_data):
        _, df, _ = loader_with_data
        counts = df.groupby("user_id").size()
        assert (counts >= 3).all()


# ---------------------------------------------------------------------------
# build_user_personas()
# ---------------------------------------------------------------------------

class TestBuildUserPersonas:
    def test_returns_dict(self, loader_with_data):
        _, _, personas = loader_with_data
        assert isinstance(personas, dict)

    def test_all_values_are_user_personas(self, loader_with_data):
        _, _, personas = loader_with_data
        assert all(isinstance(v, UserPersona) for v in personas.values())

    def test_at_least_one_persona(self, loader_with_data):
        _, _, personas = loader_with_data
        assert len(personas) >= 1

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

    def test_personas_stored_on_instance(self, loader_with_data):
        loader, _, personas = loader_with_data
        assert loader.personas is personas

    def test_reproducible_age_and_occupation(self, loader_with_data):
        loader, df, _ = loader_with_data
        p1 = loader.build_user_personas(df)
        p2 = loader.build_user_personas(df)
        for uid in p1:
            assert p1[uid].age == p2[uid].age
            assert p1[uid].occupation == p2[uid].occupation


# ---------------------------------------------------------------------------
# get_persona / get_sample_personas
# ---------------------------------------------------------------------------

class TestGetPersona:
    def test_returns_correct_persona(self, loader_with_data):
        loader, _, personas = loader_with_data
        first_id = list(personas.keys())[0]
        p = loader.get_persona(first_id)
        assert p is not None
        assert p.user_id == first_id

    def test_unknown_id_returns_none(self, loader_with_data):
        loader, _, _ = loader_with_data
        assert loader.get_persona("ghost_user_xyz_999") is None


class TestGetSamplePersonas:
    def test_returns_correct_count(self, loader_with_data):
        loader, _, _ = loader_with_data
        assert len(loader.get_sample_personas(3)) == 3

    def test_returns_fewer_if_not_enough(self, loader_with_data):
        loader, _, personas = loader_with_data
        samples = loader.get_sample_personas(10000)
        assert len(samples) <= len(personas)

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
        product = loader.get_product_from_record(df.iloc[0])
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
