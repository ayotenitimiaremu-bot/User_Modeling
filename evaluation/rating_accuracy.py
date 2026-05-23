from math import sqrt
from statistics import mean

from rich.console import Console
from rich.table import Table

from core.rating_engine import compute_rating_prior
from models.schemas import ReviewOutput, UserPersona


class RatingAccuracyEvaluator:
    def compute_rmse(
        self, predicted: list[float], actual: list[float]
    ) -> float:
        return sqrt(mean((p - a) ** 2 for p, a in zip(predicted, actual)))

    def compute_mae(
        self, predicted: list[float], actual: list[float]
    ) -> float:
        return mean(abs(p - a) for p, a in zip(predicted, actual))

    def compute_within_half_star(
        self, predicted: list[float], actual: list[float]
    ) -> float:
        if not predicted:
            return 0.0
        return sum(abs(p - a) <= 0.5 for p, a in zip(predicted, actual)) / len(predicted)

    def compute_within_one_star(
        self, predicted: list[float], actual: list[float]
    ) -> float:
        if not predicted:
            return 0.0
        return sum(abs(p - a) <= 1.0 for p, a in zip(predicted, actual)) / len(predicted)

    def behavioural_fidelity_score(
        self,
        persona: UserPersona,
        generated_reviews: list[ReviewOutput],
    ) -> dict:
        hist_ratings = [i.rating_given for i in persona.interaction_history]
        gen_ratings = [r.star_rating for r in generated_reviews]

        hist_mean = mean(hist_ratings) if hist_ratings else 3.0
        gen_mean = mean(gen_ratings) if gen_ratings else 3.0
        mean_drift = abs(hist_mean - gen_mean)

        hist_lengths = [
            len(i.review_text.split()) for i in persona.interaction_history
        ]
        gen_lengths = [len(r.review_text.split()) for r in generated_reviews]
        avg_hist_len = mean(hist_lengths) if hist_lengths else 1
        avg_gen_len = mean(gen_lengths) if gen_lengths else 0
        length_drift = abs(avg_hist_len - avg_gen_len) / max(avg_hist_len, 1)

        prior = compute_rating_prior(persona)
        violation = (
            any(r.star_rating == 5.0 for r in generated_reviews)
            if prior["never_gives_5"]
            else False
        )

        fidelity = (
            1.0
            - (0.4 * min(mean_drift, 2.0) / 2.0)
            - (0.4 * min(length_drift, 1.0))
            - (0.2 * float(violation))
        )

        return {
            "mean_rating_drift":    mean_drift,
            "length_drift_pct":     length_drift * 100,
            "behavioural_violation": violation,
            "fidelity_score":       max(fidelity, 0.0),
        }

    def full_report(
        self,
        predicted: list[float],
        actual: list[float],
        persona: UserPersona,
        generated_reviews: list[ReviewOutput],
    ) -> dict:
        report = {
            "rmse":               self.compute_rmse(predicted, actual),
            "mae":                self.compute_mae(predicted, actual),
            "within_half_star":   self.compute_within_half_star(predicted, actual),
            "within_one_star":    self.compute_within_one_star(predicted, actual),
            **self.behavioural_fidelity_score(persona, generated_reviews),
        }

        # ── Rich table ────────────────────────────────────────────────────
        table = Table(title="Rating Accuracy Evaluation — Amazon Reviews 2023")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right", style="green")

        labels = {
            "rmse":                  "RMSE",
            "mae":                   "MAE",
            "within_half_star":      "Within ±0.5 Stars",
            "within_one_star":       "Within ±1.0 Star",
            "mean_rating_drift":     "Mean Rating Drift",
            "length_drift_pct":      "Length Drift (%)",
            "behavioural_violation": "Behavioural Violation",
            "fidelity_score":        "Fidelity Score",
        }
        for key, label in labels.items():
            value = report[key]
            if isinstance(value, bool):
                table.add_row(label, "Yes" if value else "No")
            else:
                table.add_row(label, f"{value:.4f}")

        Console().print(table)
        return report
