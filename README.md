# Task A — User Review Modeling

## What This Does

Simulates authentic user reviews and star ratings for products, based on a user persona and product details.

Given a `UserPersona` (age, occupation, Big Five personality, review history) and a `ProductDetails` object, the system:

1. Builds a natural-language summary of the persona
2. Filters the persona's history for relevant past reviews
3. Runs a reasoning chain to predict how the user would react
4. Generates a review in the user's voice
5. Calibrates the star rating against the persona's historical behaviour
6. Updates the persona's inferred preferences after each review

LLM calls go to **Groq** (`llama-3.3-70b-versatile` for reasoning/generation, `llama-3.1-8b-instant` for summaries). All runs are traced in **LangSmith**.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY (and optionally LANGCHAIN_API_KEY) to .env
```

---

## Run the API

```bash
uvicorn api.main:app --reload
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## How to Get Request Data

### Option 1 — Use Sample Data (Quickest Start)

```bash
# Get a complete working request you can copy and modify
curl http://localhost:8000/reviews/sample-request

# Get 3 pre-built personas
curl http://localhost:8000/reviews/sample-personas
```

Take the `request` object from `/sample-request`, swap in your own product details, and POST it to `/reviews/generate`.

---

### Option 2 — Build from Amazon Reviews 2023 Dataset (Testing)

```bash
python data/amazon_reviews_2023.py
```

This prints a Rich table of sample personas built from the Amazon Books Reviews 2023 dataset (falls back to synthetic profiles when offline). Use these personas as starting points or for batch testing.

---

### Option 3 — Build Your Own Persona (Production Use)

Construct a `UserPersona` manually from your own user data:

```json
{
  "user_id": "user_123",
  "age": 34,
  "occupation": "teacher",
  "personality": {
    "openness": 3,
    "conscientiousness": 2,
    "extraversion": 1,
    "agreeableness": 3,
    "neuroticism": 2
  },
  "interaction_history": [
    {
      "item_id": "book_001",
      "item_title": "Atomic Habits",
      "item_category": "self-help",
      "rating_given": 4.5,
      "review_text": "Very practical advice, changed my morning routine completely.",
      "timestamp": "2024-03-01"
    }
  ]
}
```

#### Personality Traits Guide

All traits are scored **1–3** (simplified from the standard OCEAN 1–5 scale):

| Trait | 1 | 2 | 3 |
|---|---|---|---|
| **Openness** | Sticks to one genre, dislikes surprises | Tries new genres occasionally | Reads everything, loves unusual picks |
| **Conscientiousness** | Short, vague reviews ("It was OK") | Moderate detail, a few specifics | Long, structured reviews with examples |
| **Extraversion** | Rarely gives high ratings, reserved | Moderate spread of ratings | Frequently enthusiastic, many 4–5 stars |
| **Agreeableness** | High rating variance, critical | Moderate consistency | Very consistent, rarely extreme |
| **Neuroticism** | Low variance, steady rater | Moderate sensitivity | High variance, mood-driven ratings |

#### Required Fields

| Field | Type | Notes |
|---|---|---|
| `user_id` | string | Any unique identifier |
| `age` | int | 13–90 |
| `occupation` | string | Free text |
| `personality` | object | All five traits, each 1–3 |
| `interaction_history` | list | At least 1 item recommended; empty list is valid |

---

## API Endpoints

### `POST /reviews/generate`

Generate a single review.

```bash
curl -X POST http://localhost:8000/reviews/generate \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Response:
```json
{
  "review_text": "I read this over a weekend and it genuinely changed how I work...",
  "star_rating": 4.5,
  "tone": "positive",
  "reasoning_trace": "User has a history of rewarding practical self-help...",
  "confidence": 0.82,
  "mindset_update": "More open to productivity-focused non-fiction going forward."
}
```

### `POST /reviews/batch`

Generate up to 10 reviews in one call (processed sequentially to respect Groq rate limits).

```bash
curl -X POST http://localhost:8000/reviews/batch \
  -H "Content-Type: application/json" \
  -d '[{"persona": {...}, "product": {...}}, {"persona": {...}, "product": {...}}]'
```

### `GET /reviews/sample-request`

Returns a complete, copy-pasteable `ReviewRequest` built from a real Amazon Reviews 2023 persona and product. Start here if you are new to the API.

### `GET /reviews/sample-personas`

Returns 3 pre-built `UserPersona` objects from the Amazon Reviews 2023 dataset.

### `GET /health`

```json
{"status": "ok", "model": "llama-3.3-70b-versatile", "dataset": "Amazon Reviews 2023"}
```

---

## Evaluation

### Text Quality

```python
from evaluation.text_quality import ReviewQualityEvaluator

evaluator = ReviewQualityEvaluator()

# ROUGE scores against a reference review
scores = evaluator.evaluate_rouge(generated="...", reference="...")

# BERTScore across a batch (downloads distilbert-base-uncased on first run)
scores = evaluator.evaluate_bertscore(generated_list=[...], reference_list=[...])

# Full report — runs all metrics and prints a Rich table
report = evaluator.full_report(
    generated=[...],
    references=[...],
    generated_tones=[...],
    expected_tones=[...],
    historical_avg_lengths=[50, 60, 45],
)
```

### Rating Accuracy

```python
from evaluation.rating_accuracy import RatingAccuracyEvaluator

evaluator = RatingAccuracyEvaluator()

# Standard error metrics
rmse = evaluator.compute_rmse(predicted=[4.0, 3.5], actual=[4.5, 3.0])
mae  = evaluator.compute_mae(predicted=[4.0, 3.5], actual=[4.5, 3.0])

# Tolerance fractions
within_half = evaluator.compute_within_half_star(predicted, actual)
within_one  = evaluator.compute_within_one_star(predicted, actual)

# Behavioural fidelity — checks rating drift, review length drift,
# and never_gives_5 violations against persona history
fidelity = evaluator.behavioural_fidelity_score(persona, generated_reviews)

# Full report — all metrics + Rich table
report = evaluator.full_report(predicted, actual, persona, generated_reviews)
```

---

## Run Tests

```bash
# Unit tests only (no API calls)
pytest tests/ -m "not integration" -v

# Integration tests (calls real Groq API — requires .env)
pytest tests/test_integration.py -v

# All tests including integration
pytest tests/ -v
```

---

## Project Structure

```
task-a-user-modeling/
├── api/
│   ├── main.py                 # FastAPI app
│   └── routes.py               # Endpoints
├── config/
│   ├── prompts.py              # LangChain PromptTemplates
│   └── settings.py             # Pydantic Settings (reads .env)
├── core/
│   ├── rating_engine.py        # Rating calibration & memory filtering
│   └── review_generator.py     # 6-step LLM pipeline
├── data/
│   └── amazon_reviews_2023.py  # Loader + 8 synthetic user profiles
├── evaluation/
│   ├── rating_accuracy.py      # RMSE, MAE, fidelity score
│   └── text_quality.py         # ROUGE, BERTScore, tone accuracy
├── models/
│   └── schemas.py              # Pydantic v2 data contracts
└── tests/
    ├── conftest.py
    ├── test_integration.py     # Real Groq API calls
    ├── test_loader.py
    ├── test_rating_engine.py
    ├── test_review.py
    └── test_schemas.py
```
"# User_Modeling" 
