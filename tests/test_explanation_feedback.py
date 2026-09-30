import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llm_router.config import RoutingSettings
from llm_router.explanation import (
    ExplanationGenerator, canonical_explanation, describe_request, reason_clause,
)
from llm_router.feedback import FeedbackStore, ask_feedback
from llm_router.routing import RouteDecision


class FakeTokenizer:
    def apply_chat_template(self, messages, **options):
        assert options["enable_thinking"] is False
        return messages[-1]["content"]


def decision(reason="classifier", label="E2B", confidence=.9, margin=.7,
             category="general", difficulty=2):
    return RouteDecision(label, "test/model", category, difficulty, confidence, margin,
                         {"E2B": .9, "E4B": .07, "12B": .03}, reason)


class ExplanationFeedbackTests(unittest.TestCase):
    def test_category_and_difficulty_shape_reason(self):
        settings = RoutingSettings()
        recipe = decision()
        self.assertEqual(describe_request("general", 2), "простой общий вопрос")
        self.assertEqual(canonical_explanation(recipe, settings),
                         "Выбрана модель E2B, потому что это простой общий вопрос.")
        medium = decision(label="E4B", category="programming", difficulty=4)
        self.assertIn("задача по программированию средней сложности",
                      reason_clause(medium, settings))
        hard = decision(label="12B", category="mathematics", difficulty=8)
        self.assertIn("сложная математическая задача", reason_clause(hard, settings))
        low = decision("low_confidence", "E4B", .5, .3)
        self.assertIn("низкая уверенность", reason_clause(low, settings))

    def test_generated_sentence_and_grounded_fallback(self):
        selected = decision()
        generator = ExplanationGenerator(
            model_loader=lambda: (object(), FakeTokenizer()),
            generate_text=lambda model, tokenizer, prompt:
                "Выбрана модель E2B, потому что это простой общий вопрос.",
        )
        self.assertEqual(generator.explain(selected),
                         "Выбрана модель E2B, потому что это простой общий вопрос.")
        generator.generate_text = lambda *args: (
            "Выбрана модель E2B, потому что уверенность выше порога."
        )
        self.assertEqual(generator.explain(selected),
                         canonical_explanation(selected, RoutingSettings()))
        generator.model_loader = lambda: (_ for _ in ()).throw(RuntimeError("GPU unavailable"))
        self.assertEqual(generator.explain(selected),
                         canonical_explanation(selected, RoutingSettings()))

    def test_model_receives_category_and_difficulty(self):
        captured = {}
        def generate(model, tokenizer, prompt):
            captured["prompt"] = prompt
            return "Выбрана модель E2B, потому что это простой общий вопрос."
        ExplanationGenerator(
            model_loader=lambda: (object(), FakeTokenizer()), generate_text=generate
        ).explain(decision())
        self.assertIn("Категория general", captured["prompt"])
        self.assertIn("сложность 2/9", captured["prompt"])

    def test_feedback_is_local_and_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.sqlite3"
            store = FeedbackStore(path)
            values = iter(["н", "E4B", "слишком простая модель"])
            saved = ask_feedback("Вопрос", decision(), "Причина.", store,
                                 input_fn=lambda prompt: next(values), output_fn=lambda text: None)
            self.assertTrue(saved)
            with closing(sqlite3.connect(path)) as connection:
                row = connection.execute(
                    """SELECT q.text, f.rating, f.expected_model_label, f.comment
                    FROM feedback f JOIN responses r ON r.id=f.response_id
                    JOIN routing_decisions d ON d.id=r.decision_id
                    JOIN requests q ON q.id=d.request_id"""
                ).fetchone()
            self.assertEqual(row, ("Вопрос", -1, "E4B", "слишком простая модель"))
            self.assertFalse(ask_feedback("Вопрос", decision(), "Причина.", store,
                                          input_fn=lambda prompt: "", output_fn=lambda text: None))


if __name__ == "__main__":
    unittest.main()
