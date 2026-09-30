import unittest

from llm_router.config import RoutingSettings
from llm_router.routing import Router, analyze_request, difficulty_model
from router import maybe_reuse_loaded_e4b, handle_question, BENCHMARK_MARKER
from llm_router.config import MODEL_IDS


class FakeClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_probabilities(self, text):
        return self.probabilities


class RoutingTests(unittest.TestCase):
    def route(self, probabilities, text="Что такое цикл for в Python?"):
        return Router(FakeClassifier(probabilities)).route(text)

    def test_confident_prediction(self):
        decision = self.route({"E2B": 0.90, "E4B": 0.07, "12B": 0.03})
        self.assertEqual(decision.model_label, "E2B")
        self.assertAlmostEqual(decision.confidence, 0.90)
        self.assertAlmostEqual(decision.margin, 0.83)
        self.assertEqual(decision.category, "programming")
        self.assertEqual(decision.reason, "classifier")

    def test_exact_threshold_selects_predicted_model(self):
        decision = self.route({"E4B": 0.60, "E2B": 0.35, "12B": 0.05})
        self.assertEqual(decision.model_label, "E4B")
        self.assertEqual(decision.reason, "classifier")

    def test_below_threshold_escalates_one_tier(self):
        decision = self.route({"E2B": 0.59, "E4B": 0.35, "12B": 0.06})
        self.assertEqual(decision.model_label, "E4B")
        self.assertEqual(decision.reason, "low_confidence")
        decision = self.route({"E4B": 0.59, "E2B": 0.31, "12B": 0.10})
        self.assertEqual(decision.model_label, "12B")

    def test_borderline_hard_task_keeps_12b(self):
        decision = self.route({"E4B": 0.64, "12B": 0.30, "E2B": 0.06},
                              "Докажи корректность алгоритма Дейкстры")
        self.assertEqual(decision.model_label, "12B")
        self.assertEqual(decision.predicted_label, "E4B")
        self.assertEqual(decision.reason, "hard_request_borderline")

    def test_benchmark_metadata_cases(self):
        examples = [
            ("Дай простой рецепт борща на две порции.", "general", 2),
            ("Вычисли медиану чисел 4, 7, 7, 10, 12, 20.", "mathematics", 2),
            ("Напиши на Python простой калькулятор.", "programming", 4),
            ("Спроектируй миграцию монолита в сервисы с двойной записью.", "systems", 7),
            ("Предложи индексы для таблицы событий с миллиардом строк.", "databases", 7),
            ("Докажи корректность алгоритма Дейкстры.", "algorithms", 7),
            ("Разработай метод обнаружения дрейфа данных в рекомендательной модели.", "systems", 7),
            ("Проанализируй поиск ближайших соседей в миллиарде векторов.", "systems", 7),
        ]
        for text, category, minimum in examples:
            with self.subTest(text=text):
                result = analyze_request(text)
                self.assertEqual(result.category, category)
                self.assertGreaterEqual(result.difficulty, minimum)

    def test_reuse_only_short_translation_when_e4b_loaded(self):
        class Manager:
            def __init__(self, loaded):
                self.loaded = loaded
                self.calls = 0
            def get_loaded_models(self):
                self.calls += 1
                return [{"key": model} for model in self.loaded]
        recommended = self.route({"E2B": .75, "E4B": .2, "12B": .05},
                                 "Переведи короткую фразу на английский")
        manager = Manager([MODEL_IDS["E4B"]])
        used = maybe_reuse_loaded_e4b(recommended, manager, "Переведи короткую фразу на английский")
        self.assertEqual(used.model_label, "E4B")
        self.assertEqual(used.predicted_label, "E2B")
        self.assertEqual(used.reason, "reuse_loaded_e4b")
        self.assertEqual(manager.calls, 1)
        self.assertEqual(maybe_reuse_loaded_e4b(recommended, manager, "Переведи короткую фразу на английский", False), recommended)
        self.assertEqual(manager.calls, 1)
        self.assertEqual(maybe_reuse_loaded_e4b(recommended, Manager([MODEL_IDS["12B"]]), "Переведи короткую фразу на английский"), recommended)
        self.assertEqual(maybe_reuse_loaded_e4b(recommended,
                         Manager([MODEL_IDS["E2B"], MODEL_IDS["E4B"]]),
                         "Переведи короткую фразу на английский"), recommended)
        ordinary = self.route({"E2B": .8, "E4B": .15, "12B": .05}, "Дай рецепт борща")
        self.assertEqual(maybe_reuse_loaded_e4b(ordinary, manager, "Дай рецепт борща"), ordinary)
        self.assertEqual(manager.calls, 1)
        indirect = self.route({"E2B": .8, "E4B": .15, "12B": .05},
                              "Как по-немецки вежливо спросить, свободен ли этот столик?")
        self.assertEqual(maybe_reuse_loaded_e4b(indirect, manager,
                         "Как по-немецки вежливо спросить, свободен ли этот столик?"), indirect)
        self.assertEqual(manager.calls, 1)

    def test_integrated_reuse_reports_actual_model(self):
        import contextlib
        import io
        import json
        class Manager:
            last_prepare_seconds = .1
            last_request_seconds = .2
            def get_loaded_models(self):
                return [{"key": MODEL_IDS["E4B"]}]
            def chat(self, model, question):
                self.used = model
                return {"output": [{"type": "message", "content": "Готово"}], "stats": {}}
            def extract_answer(self, result):
                return result["output"][0]["content"]
        class Explainer:
            def explain(self, decision):
                return f"Выбрана модель {decision.model_label}, потому что это простой запрос на перевод."
        manager = Manager()
        router = Router(FakeClassifier({"E2B": .75, "E4B": .20, "12B": .05}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            handle_question("Переведи короткую фразу на английский", router,
                            Explainer(), manager, benchmark_json=True)
        self.assertEqual(manager.used, MODEL_IDS["E4B"])
        self.assertIn("Предсказанная модель: E2B", output.getvalue())
        payload = json.loads(output.getvalue().split(BENCHMARK_MARKER)[-1])
        self.assertEqual(payload["decision"]["predicted_label"], "E2B")
        self.assertEqual(payload["decision"]["model_label"], "E4B")

    def test_low_margin_does_not_override_confident_model(self):
        decision = self.route({"E4B": 0.66, "E2B": 0.32, "12B": 0.02})
        self.assertEqual(decision.model_label, "E4B")
        self.assertEqual(decision.reason, "classifier")

    def test_difficulty_does_not_override_confident_prediction(self):
        decision = self.route(
            {"E2B": 0.95, "E4B": 0.03, "12B": 0.02},
            "Спроектируй отказоустойчивую распределенную систему хранения",
        )
        self.assertEqual(decision.model_label, "E2B")
        self.assertGreaterEqual(decision.difficulty, 6)

    def test_low_confidence_at_maximum_tier(self):
        decision = self.route({"12B": 0.50, "E4B": 0.35, "E2B": 0.15})
        self.assertEqual(decision.model_label, "12B")
        self.assertEqual(decision.reason, "low_confidence_max_tier")

    def test_category_and_boundaries(self):
        self.assertEqual(analyze_request("Переведи текст на английский").category, "translation")
        self.assertEqual(analyze_request("Посчитай 25 процентов от 200").category, "mathematics")
        self.assertEqual([difficulty_model(n) for n in (2, 3, 5, 6)],
                         ["E2B", "E4B", "E4B", "12B"])

    def test_invalid_input_and_settings(self):
        with self.assertRaises(ValueError):
            self.route({"E2B": 1.0, "E4B": 0.0, "12B": 0.0}, "  ")
        with self.assertRaises(ValueError):
            RoutingSettings(confidence_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
