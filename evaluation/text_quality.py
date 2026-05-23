from statistics import mean
from typing import Optional

from rich.console import Console
from rich.table import Table
from rouge_score import rouge_scorer


class ReviewQualityEvaluator:
    def __init__(self) -> None:
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def evaluate_rouge(self, generated: str, reference: str) -> dict:
        scores = self.scorer.score(reference, generated)
        return {
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall":    scores["rouge1"].recall,
            "rouge1_f1":        scores["rouge1"].fmeasure,
            "rouge2_f1":        scores["rouge2"].fmeasure,
            "rougeL_f1":        scores["rougeL"].fmeasure,
        }

    def evaluate_bertscore(
        self,
        generated_list: list[str],
        reference_list: list[str],
    ) -> dict:
        from bert_score import score as bert_score  # lazy — heavy import

        P, R, F1 = bert_score(
            generated_list,
            reference_list,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall":    R.mean().item(),
            "bertscore_f1":        F1.mean().item(),
        }

    def evaluate_tone_accuracy(
        self,
        generated_tones: list[str],
        expected_tones: list[str],
    ) -> float:
        if not generated_tones:
            return 0.0
        matches = sum(g == e for g, e in zip(generated_tones, expected_tones))
        return matches / len(generated_tones)

    def evaluate_length_consistency(
        self,
        generated_reviews: list[str],
        historical_avg_lengths: list[int],
    ) -> float:
        if not generated_reviews:
            return 0.0
        diffs = [
            abs(len(gen.split()) - hist_avg) / max(hist_avg, 1)
            for gen, hist_avg in zip(generated_reviews, historical_avg_lengths)
        ]
        return 1.0 - mean(diffs)

    def full_report(
        self,
        generated: list[str],
        references: list[str],
        generated_tones: list[str],
        expected_tones: list[str],
        historical_avg_lengths: Optional[list[int]] = None,
    ) -> dict:
        rouge_scores = [
            self.evaluate_rouge(g, r) for g, r in zip(generated, references)
        ]
        avg_rouge = {
            k: mean(s[k] for s in rouge_scores) for k in rouge_scores[0]
        }

        bert = self.evaluate_bertscore(generated, references)
        tone_acc = self.evaluate_tone_accuracy(generated_tones, expected_tones)

        length_consistency: Optional[float] = None
        if historical_avg_lengths:
            length_consistency = self.evaluate_length_consistency(
                generated, historical_avg_lengths
            )

        report = {**avg_rouge, **bert, "tone_accuracy": tone_acc}
        if length_consistency is not None:
            report["length_consistency"] = length_consistency

        # ── Rich table ────────────────────────────────────────────────────
        table = Table(title="Review Quality Evaluation — Amazon Reviews 2023")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right", style="green")

        labels = {
            "rouge1_precision":   "ROUGE-1 Precision",
            "rouge1_recall":      "ROUGE-1 Recall",
            "rouge1_f1":          "ROUGE-1 F1",
            "rouge2_f1":          "ROUGE-2 F1",
            "rougeL_f1":          "ROUGE-L F1",
            "bertscore_precision": "BERTScore Precision",
            "bertscore_recall":   "BERTScore Recall",
            "bertscore_f1":       "BERTScore F1",
            "tone_accuracy":      "Tone Accuracy",
            "length_consistency": "Length Consistency",
        }
        for key, label in labels.items():
            if key in report:
                table.add_row(label, f"{report[key]:.4f}")

        Console().print(table)
        return report
