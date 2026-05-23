import json
import re
from typing import Optional

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config.prompts import (
    MINDSET_UPDATE_PROMPT,
    PERSONA_SUMMARY_PROMPT,
    REASONING_PROMPT,
    REVIEW_GENERATION_PROMPT,
)
from config.settings import settings
from core.rating_engine import calibrate_rating, compute_rating_prior, filter_relevant_memory
from models.schemas import ReviewOutput, ReviewRequest


_VALID_TONES = {"positive", "negative", "neutral", "mixed"}


class ReviewGenerator:
    def __init__(self) -> None:
        self.main_llm = ChatGroq(
            model=settings.MAIN_MODEL,
            temperature=0.7,
            groq_api_key=settings.GROQ_API_KEY,
        )
        self.fast_llm = ChatGroq(
            model=settings.FAST_MODEL,
            temperature=0.3,
            groq_api_key=settings.GROQ_API_KEY,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_review(self, request: ReviewRequest) -> ReviewOutput:
        history = request.persona.interaction_history

        # ── Step 1: persona summary (plain text) ───────────────────────
        history_text = "\n".join(
            f"- {i.item_title} ({i.item_category}): {i.rating_given}/5"
            f" — '{i.review_text[:100]}...'"
            for i in history
        ) or "No prior reviews."

        persona_summary: str = (
            PERSONA_SUMMARY_PROMPT
            | self.fast_llm
            | StrOutputParser()
        ).invoke({
            "age": request.persona.age,
            "occupation": request.persona.occupation,
            "openness": request.persona.personality.openness,
            "conscientiousness": request.persona.personality.conscientiousness,
            "extraversion": request.persona.personality.extraversion,
            "agreeableness": request.persona.personality.agreeableness,
            "neuroticism": request.persona.personality.neuroticism,
            "interaction_history_text": history_text,
        })

        # ── Step 2: relevant history + statistics ───────────────────────
        prior = compute_rating_prior(request.persona)
        relevant = filter_relevant_memory(history, request.product.category)
        if not relevant:
            relevant = list(history)[-3:]

        relevant_text = "\n".join(
            f"- {i.item_title}: {i.rating_given}/5 — '{i.review_text[:100]}'"
            for i in relevant
        ) or "No relevant history."

        avg_review_length = (
            int(np.mean([len(i.review_text.split()) for i in history])) if history else 50
        )
        historical_avg = prior["mean_rating"]
        historical_std = prior["std_rating"]

        # ── Step 3: reasoning chain (JSON) ─────────────────────────────
        context_str = str(request.context or {"time": "unspecified", "platform": "general"})

        reasoning: dict = self._safe_json_invoke(
            REASONING_PROMPT,
            self.main_llm,
            {
                "persona_summary": persona_summary,
                "product_title": request.product.title,
                "product_category": request.product.category,
                "product_description": request.product.description,
                "product_metadata": str(request.product.metadata),
                "relevant_history": relevant_text,
                "context": context_str,
            },
        )

        # ── Step 4: review generation (JSON) ───────────────────────────
        review_data: dict = self._safe_json_invoke(
            REVIEW_GENERATION_PROMPT,
            self.main_llm,
            {
                "persona_summary": persona_summary,
                "reasoning_json": json.dumps(reasoning, indent=2),
                "product_title": request.product.title,
                "product_category": request.product.category,
                "avg_review_length": avg_review_length,
                "conscientiousness": request.persona.personality.conscientiousness,
            },
        )

        # ── Step 5: mindset update (JSON) ──────────────────────────────
        current_prefs = (
            str(request.persona.inferred_preferences)
            if request.persona.inferred_preferences
            else "No prior preferences recorded"
        )
        raw_rating = float(review_data.get("star_rating", reasoning.get("predicted_rating", 3.0)))

        mindset: dict = self._safe_json_invoke(
            MINDSET_UPDATE_PROMPT,
            self.fast_llm,
            {
                "current_preferences": current_prefs,
                "product_title": request.product.title,
                "product_category": request.product.category,
                "star_rating": raw_rating,
                "tone": review_data.get("tone", "neutral"),
            },
        )

        # ── Step 6: assemble ReviewOutput ──────────────────────────────
        calibrated = calibrate_rating(
            raw_rating=raw_rating,
            historical_avg=historical_avg,
            historical_std=historical_std,
            never_gives_5=prior["never_gives_5"],
            never_gives_1=prior["never_gives_1"],
            harsh_rater=prior["harsh_rater"],
            generous_rater=prior["generous_rater"],
        )

        tone = review_data.get("tone", "neutral").lower()
        if tone not in _VALID_TONES:
            tone = "neutral"

        confidence = float(review_data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return ReviewOutput(
            review_text=review_data.get("review_text", ""),
            star_rating=calibrated,
            tone=tone,
            reasoning_trace=reasoning.get("reasoning_summary", ""),
            confidence=confidence,
            mindset_update=mindset.get("mindset_update", ""),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_json_invoke(
        self,
        prompt: PromptTemplate,
        llm: ChatGroq,
        inputs: dict,
    ) -> dict:
        """
        Invoke a prompt → LLM chain and parse JSON from the response.
        On failure, extracts any JSON block with regex, then retries once
        with a stricter instruction appended before giving up.
        """
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke(inputs)

        # Attempt 1: direct parse
        parsed = self._try_parse_json(raw)
        if parsed is not None:
            return parsed

        # Attempt 2: retry with stricter instruction
        strict_inputs = dict(inputs)
        for key in reversed(list(strict_inputs.keys())):
            if isinstance(strict_inputs[key], str):
                strict_inputs[key] += "\n\nIMPORTANT: Return ONLY the JSON object, no other text."
                break

        raw2 = chain.invoke(strict_inputs)
        parsed2 = self._try_parse_json(raw2)
        if parsed2 is not None:
            return parsed2

        raise ValueError(
            f"JSON parsing failed after retry. "
            f"Last raw output (first 300 chars): {raw2[:300]}"
        )

    @staticmethod
    def _try_parse_json(text: str) -> Optional[dict]:
        """Try direct parse, then regex-extract the first JSON object block."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        return None


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich import print as rprint

    from data.amazon_reviews_2023 import GoodreadsLoader
    from models.schemas import ProductDetails, ReviewRequest

    loader = GoodreadsLoader()
    df = loader.load(200)
    personas = loader.build_user_personas(df)
    persona = list(personas.values())[0]

    product = ProductDetails(
        product_id="test_001",
        title="Atomic Habits",
        category="self-help",
        description="A guide to building good habits and breaking bad ones",
        metadata={"author": "James Clear", "pages": 320},
    )

    generator = ReviewGenerator()
    request = ReviewRequest(persona=persona, product=product)
    result = generator.generate_review(request)

    rprint("\n[bold green]Generated Review:[/bold green]")
    rprint(result.model_dump())
