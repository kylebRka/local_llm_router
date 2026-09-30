"""Ask for feedback and optionally queue it for reviewed training."""

from .event_store import EventStore
from .config import MODEL_ORDER


class FeedbackStore(EventStore):
    """Compatibility facade; real responses should pass their response_id."""

    def record(self, question, decision, explanation, rating, expected_model=None,
               comment='', response_id=None, propose_label=False, language="ru"):
        if response_id is None:
            _, decision_id = self.record_request(question, decision, explanation)
            response_id = self.record_response(decision_id, response={'stats': {}}, answer=None)
        return self.record_feedback(response_id, rating, expected_model, comment, propose_label)


def ask_feedback(question, decision, explanation, store=None, input_fn=input,
                 output_fn=print, response_id=None, propose_label=False, language="ru"):
    """Ask once after a successful response; EOF skips feedback."""
    store = store if store is not None else FeedbackStore()
    try:
        response = input_fn('Was the route suitable? [y/n/Enter to skip]: ' if language == 'en' else 'Маршрут подошёл? [д/н/Enter — пропустить]: ').strip().lower()
    except EOFError:
        return False
    if not response:
        return False
    if response not in ('д', 'да', 'y', 'yes', 'н', 'нет', 'n', 'no'):
        output_fn('Rating not recognized; skipping.' if language == 'en' else 'Оценка не распознана, пропускаю.')
        return False
    rating = 1 if response in ('д', 'да', 'y', 'yes') else -1
    expected_model = None
    comment = ''
    if rating == -1:
        try:
            correction = input_fn('Which model would be better? [E2B/E4B/12B/Enter]: ' if language == 'en' else 'Какая модель лучше? [E2B/E4B/12B/Enter]: ').strip().upper()
            if correction in MODEL_ORDER:
                expected_model = correction
            comment = input_fn('Comment [Enter to skip]: ' if language == 'en' else 'Комментарий [Enter — пропустить]: ').strip()
        except EOFError:
            pass
    if response_id is not None:
        store.record_feedback(response_id, rating, expected_model, comment, propose_label)
    else:
        store.record(question, decision, explanation, rating, expected_model, comment,
                     propose_label=propose_label)
    output_fn('Feedback saved locally.' if language == 'en' else 'Feedback сохранён локально.')
    if propose_label and rating == -1 and expected_model is None:
        output_fn('No model specified: feedback saved; add a label with real_data.py approve.' if language == 'en' else 'Модель не указана: отзыв сохранён, метку можно добавить через real_data.py approve.')
    if propose_label:
        counts = store.counts()
        output_fn((f"Suggested labels awaiting review: {counts['pending_labels']}. Review: python real_data.py list; train after 20 approved examples per model." if language == 'en' else f"Предложенных меток ждут проверки: {counts['pending_labels']}. Проверка: python real_data.py list; обучение — после 20 подтверждённых примеров на модель."))
    return True
