from langchain_core.prompts import PromptTemplate

PERSONA_SUMMARY_PROMPT = PromptTemplate(
    input_variables=[
        "age", "occupation",
        "openness", "conscientiousness", "extraversion",
        "agreeableness", "neuroticism",
        "interaction_history_text",
    ],
    template="""You are summarising a user profile for a recommendation system.

USER PROFILE:
- Age: {age}
- Occupation: {occupation}
- Personality (Big Five, scale 1-3):
  * Openness to new things: {openness}/3
  * Conscientiousness/detail-oriented: {conscientiousness}/3
  * Extraversion/social energy: {extraversion}/3
  * Agreeableness/trust in others: {agreeableness}/3
  * Neuroticism/price sensitivity: {neuroticism}/3

THEIR RECENT REVIEWS:
{interaction_history_text}

Write a 3-sentence natural language summary of this person's reviewing style and preferences. Be specific about patterns you notice.""",
)

REASONING_PROMPT = PromptTemplate(
    input_variables=[
        "persona_summary", "product_title", "product_category",
        "product_description", "product_metadata",
        "relevant_history", "context",
    ],
    template="""You are simulating how a real person would think before writing a review. Do not write the review yet — only reason.

WHO THIS PERSON IS:
{persona_summary}

WHAT THEY ARE REVIEWING:
Title: {product_title}
Category: {product_category}
Description: {product_description}
Details: {product_metadata}

THEIR RELEVANT PAST REVIEWS IN THIS CATEGORY:
{relevant_history}

CURRENT CONTEXT:
{context}

Think through these questions step by step:
1. What is their first gut reaction to this product?
2. Based on their past behaviour, what will they notice first?
3. What specifically might disappoint them?
4. What specifically might impress them?
5. Compared to things they have reviewed before, is this better or worse? By how much?
6. What star rating is this person likely to give (1-5, consider their historical average)?

Return ONLY a JSON object:
{{
  "gut_reaction": "string",
  "will_notice": "string",
  "disappointments": "string",
  "impressions": "string",
  "comparison": "string",
  "predicted_rating": 0.0,
  "reasoning_summary": "string"
}}""",
)

REVIEW_GENERATION_PROMPT = PromptTemplate(
    input_variables=[
        "persona_summary", "reasoning_json", "product_title",
        "product_category", "avg_review_length", "conscientiousness",
    ],
    template="""Using the reasoning below, write an authentic product review as this user would write it.

USER PROFILE:
{persona_summary}

PRE-REVIEW REASONING:
{reasoning_json}

PRODUCT: {product_title} ({product_category})

WRITING INSTRUCTIONS:
- Target length: approximately {avg_review_length} words (match this user's historical average)
- If conscientiousness is {conscientiousness}/3:
  * 3 = detailed, structured, specific examples
  * 2 = moderate detail, some specifics
  * 1 = brief, casual, general impressions
- The rating from the reasoning is your guide — match tone to rating
- Write in first person
- Sound like a real person, not marketing copy
- Do NOT mention that you are an AI

Return ONLY a JSON object:
{{
  "review_text": "string",
  "star_rating": 0.0,
  "tone": "positive|negative|neutral|mixed",
  "confidence": 0.0
}}""",
)

MINDSET_UPDATE_PROMPT = PromptTemplate(
    input_variables=[
        "current_preferences", "product_title",
        "product_category", "star_rating", "tone",
    ],
    template="""A user just reviewed '{product_title}' ({product_category}) and gave it {star_rating} stars with a {tone} tone.

Their current preference state: {current_preferences}

In 1-2 sentences, how does this interaction update their preferences? What are they more or less likely to want next?

Return ONLY a JSON object:
{{
  "mindset_update": "string",
  "updated_preferences": "string"
}}""",
)
