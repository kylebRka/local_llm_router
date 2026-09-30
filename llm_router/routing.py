"""Question analysis and model selection; no LM Studio side effects."""

import re
from dataclasses import dataclass
from typing import Mapping

from .classifier import ModelClassifier
from .config import MODEL_IDS, MODEL_ORDER, RoutingSettings

# Ordered by specificity: a SQL comparison is about databases, not generic analysis.
CATEGORY_PATTERNS = (
    ('translation', r'\b(перевед\w*|перевод\w*|translate|translation|передай на (английск\w*|испанск\w*|французск\w*|немецк\w*)|по-немецки|английск\w* вариант|разниц\w* в тоне)\b'),
    ('databases', r'\b(sql|postgresql|database|schema|foreign key|query plan|indexing|реляцион\w*|первичн\w* ключ\w*|баз[аыуе] данн\w*|денормализац\w*|n\+1|схем[аыуе] баз\w*|таблиц[аыуе] заказ\w*|таблиц[аыуе] событ\w*|идемпотентност\w* запис\w* заказ\w*)\b'),
    ('algorithms', r'\b(algorithm|dijkstra|shortest path|дейкстр\w*|кратчайш\w* пут\w*|корректност\w* алгоритм\w*)\b'),
    ('mathematics', r'\b(fraction|percentage|average|median|equation|integral|probability|calculate|math|convergence|iterative method|numerical|bayesian|stochastic|optimization|proof|theorem|медиан\w*|посчитай|вычисли|уравнени\w*|интегрир\w*|интеграл\w*|производн\w*|вероятност\w*|процент\w*|погрешност\w*|теорем\w*|индукци\w*|ж[её]стк\w* систем\w*|численн\w* решени\w*|площад\w* треугольник\w*|скидк\w*|сумм\w* неч[её]тн\w*|сколько осталось|сколько плиток)\b'),
    ('systems', r'\b(architecture|distributed|multi-region|fault-tolerant|scalability|outage|consistency|streaming|api|redis|к[еэ]широван\w*|сервис\w*|согласованност\w*|очеред\w*|балансировщик\w*|прокси|трафик\w*|плат[её]ж\w*|шардировани\w*|репликаци\w*|микросервис\w*|монолит\w*|распредел[её]нн\w*|raft|paxos|сервер\w*|рекомендательн\w* модел\w*|вектор\w*|чат для 50 тысяч)\b'),
    ('science', r'\b(photosynthesis|evaporation|gravity|magnetism|science|water cycle|фотосинтез\w*|испарени\w*|радуг\w*|давлени\w* воздух\w*|бактери\w*|вирус\w*|растени\w*|фасол\w*|теплов\w* расширени\w*|наблюдательн\w* исследовани\w*|контролируем\w* эксперимент\w*|металлическ\w* ложк\w*|схем\w* эксперимент\w*|солнечн\w* свет\w*|зим\w* день\w*)\b'),
    ('writing', r'\b(write a greeting|draft a note|invitation|announcement|apology|caption|email|cover letter|newsletter|speech|user guide|status report|paragraph|письм\w*|сообщени\w* преподавател\w*|приглашени\w*|объявлени\w*|заметк\w*|ответ покупател\w*|заголовк\w*|отредактируй|перепиши фраз\w*|сократи до|план выступлени\w*|резюме|сочинени\w*|стать\w*|эссе)\b'),
    ('programming', r'\b(python|javascript|typescript|code|function|script|programming|debug|calculator|код\w*|функци\w*|скрипт\w*|indexerror|генератор\w*|библиотек\w*|программ\w*|debug|тест\w* для функци\w*)\b'),
    ('analysis', r'\b(analyze|compare|evaluate|trade-offs|causal|risk analysis|decision framework|survey design|experiment|проанализируй|сравни|оцени|разбери|критери\w*|плюс\w* и минус\w*|риск\w*|метрик\w* качества|конверсии\w*|как оценить|план проверк\w*|компромисс\w*|analyse|analyze|compare)\b'),
)

HIGH_PATTERNS = (
    r'\b(derive|prove|proof|rigorous|formal argument|stability analysis|constrained optimization|sensitivity analysis|stochastic|multiple comparisons|competing explanations|failure modes|failure recovery|under load|causal impact|selection bias|regime changes|multi-year|production solution|large-scale|distributed|zero-downtime)\b',
    r'\b(architect|design a distributed|multi-region|regional outage|zero-downtime|failure modes|rollback|sharding|спроектируй|шардировани\w*|отказоустойчив\w*|распредел[её]нн\w*|микросервис\w*|миллиард\w*|миллион\w* транзакц\w*|50 тысяч одновременн\w*|raft|paxos|сетев\w* разделени\w*)\b',
    r'\b(докажи|выведи услови\w* устойчивост\w*|ж[её]стк\w* систем\w*|дрейф\w* данн\w*|причинн\w* эффект\w*|утечк\w* памят\w*|без остановк\w*|двойн\w* запис\w*|без простоя|дублировани\w* плат[её]ж\w*|redis-кластер\w*|тр[её]х регион\w*)\b',
    r'\b(разработай метод|плат[её]жн\w* платформ\w*|многорегиональн\w*|production|восстановлени\w* кластера|конкурентн\w* доступ\w*)\b',
)
MEDIUM_PATTERNS = (
    r'\b(solve|calculate|worked example|draft|polished|professional|analyze|compare|evaluate|decision framework|two-step|error handling|working code|write a python|create a python)\b',
    r'\b(write a python|create a python|working code|implement|compare|analyze|evaluate|сравни|разниц\w*|проанализируй|оцени|реализуй|напиши\s+(?:на\s+\w+\s+)?(?:скрипт|функци\w*|программ\w*|калькулятор)|покажи\s+(?:функци\w*|sql)|предложи тест\w*|стратеги\w*)\b',
    r'\b(sql|индексаци\w*|миграци\w*|гипотез\w*|эксперимент\w*|опыт\w*|оптимизир\w*|тест\w* для функци\w*|api без дублировани\w*|калькулятор|напиши на python|индекс\w* postgresql|ограничить поток|план проверк\w*|предложи метрик\w*|риск\w* для запуск\w*|плюс\w* и минус\w*|как оценить|согласованност\w*)\b',
)
SIMPLE_PATTERNS = (r'\b(переведи|translate|how many|what is|define|explain in simple terms|сколько|что такое|дай определение|объясни простыми словами)\b',)


@dataclass(frozen=True)
class RequestAnalysis:
    category: str
    difficulty: int


@dataclass(frozen=True)
class RouteDecision:
    model_label: str
    model_id: str
    category: str
    difficulty: int
    confidence: float
    margin: float
    probabilities: Mapping[str, float]
    reason: str
    predicted_label: str | None = None


def analyze_request(text: str) -> RequestAnalysis:
    """Independent topic and task-complexity estimates (0–9)."""
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError('Запрос не должен быть пустым')
    category = next((name for name, pattern in CATEGORY_PATTERNS if re.search(pattern, normalized)), 'general')
    high_hits = sum(bool(re.search(pattern, normalized)) for pattern in HIGH_PATTERNS)
    medium_hits = sum(bool(re.search(pattern, normalized)) for pattern in MEDIUM_PATTERNS)
    if high_hits:
        difficulty = min(9, 6 + high_hits + (medium_hits > 0))
    elif medium_hits:
        difficulty = min(5, 3 + medium_hits)
    elif any(re.search(pattern, normalized) for pattern in SIMPLE_PATTERNS):
        difficulty = 1
    else:
        difficulty = 2
    return RequestAnalysis(category, difficulty)


def difficulty_model(difficulty: int) -> str:
    if not 0 <= difficulty <= 9:
        raise ValueError('difficulty must be between 0 and 9')
    return MODEL_ORDER[0 if difficulty <= 2 else 1 if difficulty <= 5 else 2]


class Router:
    def __init__(self, classifier=None, settings=None, model_ids=None):
        self.classifier = classifier if classifier is not None else ModelClassifier()
        self.settings = settings if settings is not None else RoutingSettings()
        self.model_ids = dict(model_ids) if model_ids is not None else dict(MODEL_IDS)

    def route(self, text: str) -> RouteDecision:
        analysis = analyze_request(text)
        probabilities = self.classifier.predict_probabilities(text)
        if set(probabilities) != set(MODEL_ORDER):
            raise ValueError('Classifier classes must match configured model labels')
        if any(not 0 <= value <= 1 for value in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 0.01:
            raise ValueError('Classifier probabilities must sum to 1')
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)
        predicted = ranked[0]
        confidence = probabilities[predicted]
        margin = confidence - probabilities[ranked[1]]
        model_index = MODEL_ORDER.index(predicted)
        reason = 'classifier'
        if confidence < self.settings.confidence_threshold:
            if model_index < len(MODEL_ORDER) - 1:
                model_index += 1
                reason = 'low_confidence'
            else:
                reason = 'low_confidence_max_tier'
        elif confidence < 0.65 and analysis.difficulty >= 7 and model_index == 1:
            # Keep the previous conservative route for hard, borderline E4B tasks.
            model_index = 2
            reason = 'hard_request_borderline'
        label = MODEL_ORDER[model_index]
        return RouteDecision(label, self.model_ids[label], analysis.category, analysis.difficulty,
                             confidence, margin, probabilities, reason, predicted)
