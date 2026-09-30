"""Command-line entry point for the local LLM router."""

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, replace
from time import perf_counter

from ..explanation import ExplanationGenerator
from ..feedback import FeedbackStore, ask_feedback
from ..models import ModelManager
from ..config import MODEL_IDS
from ..settings import AppSettings
from ..i18n import category_name
from ..routing import Router

BENCHMARK_MARKER = "ROUTER_BENCHMARK_JSON: "


def maybe_reuse_loaded_e4b(decision, manager, question, enabled=True, model_ids=None):
    """Reuse E4B only for short translations recommended to E2B."""
    model_ids = model_ids or MODEL_IDS
    if not enabled or decision.model_label != "E2B":
        return decision
    if decision.category != "translation" or decision.difficulty > 2:
        return decision
    if len(question) > 160 or not re.match(r"^\s*(?:переведи|передай на|translate)\b", question, re.IGNORECASE):
        return decision
    loaded = manager.get_loaded_models()
    if any(item["key"] == model_ids["E2B"] for item in loaded):
        return decision
    if any(item["key"] == model_ids["E4B"] for item in loaded):
        return replace(decision, model_label="E4B", model_id=model_ids["E4B"],
                       reason="reuse_loaded_e4b")
    return decision


def handle_question(question, router, explainer, manager, benchmark_json=False,
                    reuse_loaded_e4b=True, event_store=None, collect_feedback_labels=False, language="ru", model_ids=None):
    started = perf_counter()
    decision = router.route(question)
    decision = maybe_reuse_loaded_e4b(decision, manager, question, reuse_loaded_e4b, model_ids)
    routed = perf_counter()
    explanation = explainer.explain(decision)
    explained = perf_counter()
    store = None if benchmark_json else (event_store if event_store is not None else FeedbackStore())
    decision_id = None
    if store is not None:
        try:
            _, decision_id = store.record_request(question, decision, explanation)
        except (sqlite3.Error, OSError, RuntimeError) as error:
            print(f"{'Could not save request to local database' if language == 'en' else 'Не удалось сохранить запрос в локальной базе'}: {error}")

    print("\n" + ("Probabilities:" if language == "en" else "Вероятности:"))
    for label, probability in sorted(
        decision.probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {label}: {probability:.2%}")
    print(f"\n{'Category' if language == 'en' else 'Категория'}: {category_name(language, decision.category)}")
    print(f"{'Difficulty' if language == 'en' else 'Сложность'}: {decision.difficulty}/9")
    print(f"{'Prediction confidence' if language == 'en' else 'Уверенность предсказания'}: {decision.confidence:.2%}")
    print(f"Margin: {decision.margin:.2%}")
    if decision.predicted_label and decision.predicted_label != decision.model_label:
        print(f"{'Predicted model' if language == 'en' else 'Предсказанная модель'}: {decision.predicted_label}")
    print(f"{'Selected model' if language == 'en' else 'Выбрана модель'}: {decision.model_label}")
    print(f"{'Reason' if language == 'en' else 'Причина'}: {explanation}")
    print(f"LM Studio model: {decision.model_id}\n")

    try:
        result = manager.chat(decision.model_id, question)
    except Exception as error:
        if decision_id is not None:
            try:
                store.record_response(decision_id, error=error, timings={
                    "routing": routed - started,
                    "explanation": explained - routed,
                    "total": perf_counter() - started,
                    "model_prepare": manager.last_prepare_seconds,
                    "model_request": manager.last_request_seconds,
                })
            except (sqlite3.Error, OSError, RuntimeError) as storage_error:
                print(f"{'Could not save error to local database' if language == 'en' else 'Не удалось сохранить ошибку в локальной базе'}: {storage_error}")
        raise
    chatted = perf_counter()
    answer = manager.extract_answer(result)
    print(("Answer:" if language == "en" else "Ответ:") + "\n")
    print(answer)
    response_id = None
    if decision_id is not None:
        try:
            response_id = store.record_response(decision_id, response=result, answer=answer,
                                                timings={
                                                    "routing": routed - started,
                                                    "explanation": explained - routed,
                                                    "model_prepare": manager.last_prepare_seconds,
                                                    "model_request": manager.last_request_seconds,
                                                    "total": chatted - started,
                                                })
        except (sqlite3.Error, OSError, RuntimeError) as error:
            print(f"{'Could not save response to local database' if language == 'en' else 'Не удалось сохранить ответ в локальной базе'}: {error}")

    stats = result.get("stats", {})
    print("\n" + "=" * 50)
    print("Statistics:" if language == "en" else "Статистика:")
    print("=" * 50)
    print(f"  Input tokens:      {stats.get('input_tokens', 'N/A')}")
    print(f"  Output tokens:     {stats.get('total_output_tokens', 'N/A')}")
    print(f"  Reasoning tokens:  {stats.get('reasoning_output_tokens', 'N/A')}")
    print(f"  TTFT:              {stats.get('time_to_first_token_seconds', 'N/A')} s")
    print(f"  Generation speed:  {stats.get('tokens_per_second', 'N/A')} tok/s")
    print("=" * 50)

    if benchmark_json:
        payload = {
            "question": question,
            "decision": asdict(decision),
            "explanation": explanation,
            "lm_studio_response": result,
            "timings_seconds": {
                "routing": routed - started,
                "explanation": explained - routed,
                "lm_studio_total": chatted - explained,
                "model_prepare": manager.last_prepare_seconds,
                "model_request": manager.last_request_seconds,
                "total_in_process": chatted - started,
            },
        }
        print(BENCHMARK_MARKER + json.dumps(payload, ensure_ascii=False))
    else:
        if response_id is not None:
            try:
                ask_feedback(question, decision, explanation, store=store,
                             response_id=response_id, propose_label=collect_feedback_labels, language=language)
            except (sqlite3.Error, OSError, RuntimeError) as error:
                print(f"{'Could not save feedback to local database' if language == 'en' else 'Не удалось сохранить feedback в локальной базе'}: {error}")
        else:
            print("Feedback unavailable: response was not saved locally." if language == "en" else "Feedback недоступен: ответ не сохранён в локальной базе.")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Локальный роутер запросов / Local request router")
    parser.add_argument(
        "--benchmark-json", action="store_true", help="Один запрос с полными метриками / One request with full metrics"
    )
    parser.add_argument(
        "--reuse-loaded-e4b", action=argparse.BooleanOptionalAction, default=None,
        help="Повторно использовать загруженную E4B для перевода / Reuse loaded E4B for translation",
    )
    parser.add_argument(
        "--collect-feedback-labels", action="store_true",
        help="Предлагать метки из отзывов / Queue feedback labels for review",
    )
    parser.add_argument("--lang", choices=("ru", "en"), default=None,
                        help="Язык / Language: ru or en")
    args = parser.parse_args(argv)
    app_settings = AppSettings.load()
    language = args.lang or app_settings.language
    router = Router(model_ids=app_settings.model_ids)
    explainer = ExplanationGenerator(router.settings, language=language)
    manager = ModelManager(app_settings.lm_studio_url, language)
    while True:
        try:
            question = input("Request (Enter to exit): " if language == "en" else "Запрос (Enter — выход): ").strip()
        except EOFError:
            return
        if not question:
            return
        handle_question(question, router, explainer, manager, args.benchmark_json,
                        (app_settings.reuse_loaded_e4b if args.reuse_loaded_e4b is None else args.reuse_loaded_e4b), collect_feedback_labels=(args.collect_feedback_labels or app_settings.collect_feedback_labels),
                        language=language, model_ids=app_settings.model_ids)
        if args.benchmark_json:
            return


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
